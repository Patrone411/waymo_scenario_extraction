from pathlib import Path

from run_matching import (
    _build_source,
    _first_window,
)
from osc2_parser.parser import OSCProgram
from scenario_matching.matching.post.plan import build_block_plans
from scenario_matching.matching.engine import MatchEngine
from scenario_matching.harness import HarnessConfig
from scenario_extraction.parquet_source import AzureParquetSource
import os



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


BASE_PREFIX = "parquet/matcher_ci_test-small/00000"
EXPECTED_HITS = 2
def test_known_scenario_produces_expected_hits():

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
        cfg.first_window_only = True

        engine = MatchEngine(
            cfg=cfg,
            scn_constraints=scn,
            calls=calls,
        )

        # -----------------------------------------------------
        # 4. Azure ParquetSource
        # -----------------------------------------------------

        
        src = AzureParquetSource(
            account_name=os.environ["AZURE_STORAGE_ACCOUNT"],
            account_key=os.environ["AZURE_STORAGE_KEY"],
            container="parquets",
            base_prefix=BASE_PREFIX,
            min_lanes = prog.min_lanes
        )
    
        #src = next(iter(source), None)

        # -----------------------------------------------------
        # 5. Matching
        # -----------------------------------------------------

        total_hits = 0
        processed_scenes = 0

        for res in src:

            processed_scenes += 1

            assert res.feats_by_seg, (
                "Parquet wurde gelesen, aber keine Features gefunden"
            )

            engine.set_features(
                res.feats_by_seg,
                res.seg_meta_by_id,
            )

            batch = engine.process_loaded_features_with_plans(
                plans=plans,
                source_uri="test",
                collect_call_windows=True,
                collect_modifier_stats=False,
            )

            for block_label, hitmap in (
                batch.block_hits or {}
            ).items():

                for (_seg_id, _role_key), block_state in (
                    hitmap or {}
                ).items():

                    t0, t1 = _first_window(
                        getattr(
                            block_state,
                            "windows_by_t0",
                            None,
                        )
                    )

                    if t0 is not None:
                        total_hits += 1

            engine.clear_features()

        # -----------------------------------------------------
        # 6. Assertions
        # -----------------------------------------------------

        assert processed_scenes > 0, (
            "Keine Testszene aus Azure Parquet geladen"
        )

        assert total_hits == EXPECTED_HITS, (
            f"Expected {EXPECTED_HITS} hits, "
            f"but matcher produced {total_hits}"
        )

    finally:
        os.chdir(old_cwd)