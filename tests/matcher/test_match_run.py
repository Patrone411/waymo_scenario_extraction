from pathlib import Path
import os

from run_matching import _first_window
from osc2_parser.parser import OSCProgram
from scenario_matching.matching.post.plan import build_block_plans
from scenario_matching.matching.engine import MatchEngine
from scenario_matching.harness import HarnessConfig
from scenario_matching.analysis_stats.stats_windows import (
    max_possible_windows,
    count_windows,
)
from scenario_extraction.parquet_source import AzureParquetSource
from azure_results_writer import AzureResultsWriter


# ---------------------------------------------------------
# Test configuration
# ---------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_EXTRACTION = REPO_ROOT / "scenario_extraction"

OSC_FILE = (
    SCENARIO_EXTRACTION
    / "osc2_parser"
    / "osc"
    / "change_lane.osc"
)

# komplette Shard 00000, nicht nur die erste Szene
BASE_PREFIX = "parquet/matcher_ci_test/00000"

RESULTS_CONTAINER = "results"
RESULTS_PREFIX = "results/matcher_ci_test"

RUN_ID = "ci-azure-full-shard"
SHARD_INDEX = 0

EXPECTED_MIN_HITS = 1  # mind. ein Hit erwartet, exakter Wert je nach Shard-Inhalt anpassen


def test_full_shard_matching_and_azure_upload():

    # -----------------------------------------------------
    # 1. OSC parsen
    # -----------------------------------------------------

    assert OSC_FILE.exists(), (
        f"Test-OSC file not found: {OSC_FILE}"
    )

    old_cwd = Path.cwd()

    try:
        os.chdir(SCENARIO_EXTRACTION)

        prog = OSCProgram(
            osc_path="osc2_parser/osc/change_lane.osc"
        ).compile()

        assert prog is not None
        assert "top" in prog.constraints_by_scenario

        scn = prog.constraints_by_scenario["top"]

        # -----------------------------------------------------
        # 2. Calls / Plans erzeugen
        # -----------------------------------------------------

        calls = []

        dur_by_label = getattr(
            prog,
            "block_durations",
            {},
        ) or {}

        for c in prog.calls:

            c2 = dict(c)

            lbl = c2.get("block_label")

            if lbl in dur_by_label and c2.get("duration") is None:
                c2["block_duration"] = dur_by_label[lbl]

            c2.setdefault(
                "block_overlap",
                "start",
            )

            calls.append(c2)

        plans = build_block_plans(
            calls,
            fps=10,
        )

        for plan in plans.values():
            plan.collect_block_windows = True

        # -----------------------------------------------------
        # 3. MatchEngine
        #    (kein first_window_only -> volle Shard-Verarbeitung,
        #    damit alle Hits/Fenster wie im echten Batch-Run erfasst werden)
        # -----------------------------------------------------

        cfg = HarnessConfig(
            fps=10,
            exact_lanes=prog.min_lanes,
            debug_match=False,
            debug_segments=False,
            debug_checks=False,
        )

        cfg.use_sed = True
        cfg.debug_pcs = False
        cfg.first_window_only = False

        engine = MatchEngine(
            cfg=cfg,
            scn_constraints=scn,
            calls=calls,
        )

        # -----------------------------------------------------
        # 4. Azure ParquetSource (komplette Shard)
        # -----------------------------------------------------

        src = AzureParquetSource(
            account_name=os.environ["AZURE_STORAGE_ACCOUNT"],
            account_key=os.environ["AZURE_STORAGE_KEY"],
            container="parquets",
            base_prefix=BASE_PREFIX,
            min_lanes=prog.min_lanes,
        )

        # -----------------------------------------------------
        # 5. AzureResultsWriter fuer die drei Ergebnistabellen
        # -----------------------------------------------------

        writer = AzureResultsWriter(
            run_id=RUN_ID,
            scenario=os.path.basename(str(OSC_FILE)),
            shard_index=SHARD_INDEX,
            account_name=os.environ["AZURE_STORAGE_ACCOUNT"],
            account_key=os.environ["AZURE_STORAGE_KEY"],
            container=RESULTS_CONTAINER,
            prefix=RESULTS_PREFIX,
        )

        # -----------------------------------------------------
        # 6. Matching ueber die komplette Shard
        # -----------------------------------------------------

        processed_scenes = 0
        total_hits = 0

        for res in src:

            processed_scenes += 1

            assert res.feats_by_seg, (
                "Parquet wurde gelesen, aber keine Features gefunden"
            )

            engine.set_features(
                res.feats_by_seg,
                res.seg_meta_by_id,
            )

            meta_any = next(iter(res.seg_meta_by_id.values()), None)
            source_uri = getattr(meta_any, "source_uri", "<unknown>")
            scene_id = Path(source_uri).stem

            batch = engine.process_loaded_features_with_plans(
                plans=plans,
                source_uri=source_uri,
                collect_call_windows=True,
                collect_modifier_stats=False,
            )

            for block_label, hitmap in (
                batch.block_hits or {}
            ).items():

                plan = plans.get(block_label)
                if plan is None:
                    continue
                minF = int(getattr(plan, "duration_min_frames", 1) or 1)

                for (seg_id, _role_key), block_state in (
                    hitmap or {}
                ).items():

                    roles = dict(getattr(block_state, "roles", {}) or {})
                    feats = res.feats_by_seg.get(seg_id)
                    if feats is None:
                        continue

                    wbt0 = getattr(block_state, "windows_by_t0", None)
                    t0, t1 = _first_window(wbt0)

                    if t0 is None:
                        continue

                    T = int(getattr(feats, "T", 91) or 91)
                    maxF = int(getattr(plan, "duration_max_frames", None) or T)
                    nwin = int(count_windows(wbt0))
                    nposs = int(max_possible_windows(T, minF, maxF)) if T else 0

                    writer.add_hit(
                        scene_id=scene_id,
                        segment_id=seg_id,
                        block_label=block_label,
                        roles=roles,
                        t0=t0,
                        t1=t1,
                        n_windows=nwin,
                        n_possible_windows=nposs,
                        source_uri=source_uri,
                        feats=feats,
                    )

                    total_hits += 1

            engine.clear_features()

        # -----------------------------------------------------
        # 7. Ergebnisse nach Azure schreiben
        # -----------------------------------------------------

        written = writer.flush()

        # -----------------------------------------------------
        # 8. Assertions
        # -----------------------------------------------------

        assert processed_scenes > 0, (
            "Keine Testszenen aus Azure Parquet geladen"
        )

        assert total_hits >= EXPECTED_MIN_HITS, (
            f"Erwartet mindestens {EXPECTED_MIN_HITS} Hit(s), "
            f"aber Matcher produzierte {total_hits}"
        )

        assert "match_hits" in written, (
            "match_hits wurde nicht nach Azure geschrieben (keine Hits gefunden?)"
        )

    finally:
        os.chdir(old_cwd)