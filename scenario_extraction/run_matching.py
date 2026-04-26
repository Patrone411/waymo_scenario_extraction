#!/usr/bin/env python3
"""
run_matching.py

Läuft sowohl lokal (via CLI-Args) als auch als Kubernetes Job (via Env-Vars).

Input:  Parquet-Dateien auf S3 oder lokalem Dateisystem
        (erzeugt von worker.py aus GCS TFRecords)

Output (lokal oder im Pod-FS, optional nach S3 hochgeladen):
  - hits_windows.jsonl        (optional; Block-Hit-Windows pro (seg, binding))
  - stats_shard.json          (mergeable Counters + Param-Histogramme)
  - example_windows.jsonl     (optional; 1 Beispiel-Window pro Hit-Binding)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# --- core pipeline ---
from osc2_parser.parser import OSCProgram
from scenario_matching.matching.post.plan import build_block_plans
from scenario_matching.matching.engine import MatchEngine
from scenario_matching.harness import HarnessConfig

# --- stats modules ---
from scenario_matching.analysis_stats.stats_collector import StatsCollector
from scenario_matching.analysis_stats.stats_windows import max_possible_windows, count_windows
from scenario_matching.analysis_stats.example_windows_io import append_example_windows_jsonl

# --- generischer OSC2-Extraktor (ersetzt szenario-spezifische Extraktoren) ---
from generic_window_extractor import make_generic_extractor, extract_params_for_window

# --- szenario-spezifische Extraktoren (Rückwärtskompatibilität) ---
try:
    from scenario_matching.analysis_stats.stats_extractors_Cut_in import (
        PARAM_SPECS as CUT_IN_PARAM_SPECS,
        extract_change_lane_features,
    )
except Exception:
    CUT_IN_PARAM_SPECS = {}
    extract_change_lane_features = None

try:
    from scenario_matching.analysis_stats.stats_extractors_cross import (
        PARAM_SPECS as CROSS_PARAM_SPECS,
        extract_cross_features,
    )
except Exception:
    CROSS_PARAM_SPECS = {}
    extract_cross_features = None

try:
    from scenario_matching.analysis_stats.stats_extractors_ttc import (
        PARAM_SPECS as TTC_PARAM_SPECS,
        extract_ttc_features,
    )
except Exception:
    TTC_PARAM_SPECS = {}
    extract_ttc_features = None

# --- Parquet-Source (ersetzt S3PickleSource) ---
from parquet_source import LocalParquetSource, ParquetSource


# ─────────────────────────────────────────────────────────────────────────────
# Env-Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _env(k, default=None):
    return os.environ.get(k, default)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None else default


def _parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {x!r}")


def _add_bool_opt(ap: argparse.ArgumentParser, name: str, default: bool, help: str) -> None:
    ap.add_argument(name, nargs="?", const=True, default=default, type=_parse_bool, help=help)


# ─────────────────────────────────────────────────────────────────────────────
# Misc Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _intervals_to_jsonable(intervals: Any) -> list:
    out = []
    if not intervals:
        return out
    for it in intervals:
        if it is None:
            continue
        if is_dataclass(it):
            d = asdict(it)
            a = d.get("start", d.get("t0", d.get("lo", None)))
            b = d.get("end",   d.get("t1", d.get("hi", None)))
            if a is not None and b is not None:
                out.append([int(a), int(b)])
            continue
        if isinstance(it, (tuple, list)) and len(it) == 2:
            out.append([int(it[0]), int(it[1])])
            continue
        a = getattr(it, "start", None)
        b = getattr(it, "end",   None)
        if a is not None and b is not None:
            out.append([int(a), int(b)])
    return out


def _windows_by_t0_to_jsonable(wbt0: Any) -> Optional[Dict[str, list]]:
    if not wbt0:
        return None
    out: Dict[str, list] = {}
    for t0, ranges in (wbt0 or {}).items():
        k = str(int(t0))
        out[k] = [[int(lo), int(hi)] for (lo, hi) in ranges]
    return out


def _s3_put_bytes(*, bucket: str, key: str, data: bytes,
                  endpoint_url: Optional[str], verify: Optional[str]) -> None:
    try:
        import boto3
    except ImportError:
        raise RuntimeError("boto3 nicht verfügbar; S3-Upload nicht möglich.")
    s3 = boto3.client("s3", endpoint_url=endpoint_url, verify=verify)
    s3.put_object(Bucket=bucket, Key=key, Body=data)


def _s3_put_file(*, bucket: str, key: str, path: Path,
                 endpoint_url: Optional[str], verify: Optional[str]) -> None:
    _s3_put_bytes(bucket=bucket, key=key, data=path.read_bytes(),
                  endpoint_url=endpoint_url, verify=verify)


def _compute_shard_prefix(base_prefix_root: str, shard_index: int) -> str:
    return f"{base_prefix_root.rstrip('/')}/{shard_index:05d}"


def _debug_dump(args, base_prefix: str, shard_tag: str) -> None:
    dbg = {
        "resolved": {
            "base_prefix": base_prefix,
            "shard_tag":   shard_tag,
        },
        "cli_args": vars(args),
        "env_relevant": {k: _env(k) for k in [
            "OSC_FILE", "USE_SED", "BLOCK_OVERLAP",
            "COLLECT_UNCOND_DENOMS", "COLLECT_CALLS", "COLLECT_CHECKS", "COLLECT_PARAMS",
            "HIT_SAMPLES_PER_BINDING", "BASE_SAMPLES_PER_BINDING",
            "BASELINE_ONLY_FOR_HIT_BINDINGS", "MAX_BINDINGS_PER_SEG",
            "BASE_PREFIX_ROOT", "BASE_PREFIX", "N_PICKLES",
            "JOB_COMPLETION_INDEX", "MAX_SHARD_INDEX", "START_OFFSET",
        ]},
    }
    print("=== MATCHER DEBUG CONFIG ===")
    print(json.dumps(dbg, indent=2, sort_keys=True), flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Extraktor-Auswahl
# ─────────────────────────────────────────────────────────────────────────────

def _pick_extractor(osc_file: str, calls: list, left_is_decreasing: bool):
    """
    Versucht einen szenario-spezifischen Extraktor zu laden (Rückwärtskompatibilität).
    Fällt auf den generischen OSC2-basierten Extraktor zurück wenn keiner passt.

    Returns: (param_specs, extractor_fn)
    """
    name = os.path.basename(osc_file).lower()

    if "cut_in" in name and extract_change_lane_features is not None:
        return (
            CUT_IN_PARAM_SPECS,
            lambda feats, roles, t0, t1: extract_change_lane_features(
                feats, roles, t0, t1, left_is_decreasing=left_is_decreasing
            ),
        )

    if "cross" in name and extract_cross_features is not None:
        return (
            CROSS_PARAM_SPECS,
            lambda feats, roles, t0, t1: extract_cross_features(
                feats, roles, t0, t1, left_is_decreasing=left_is_decreasing
            ),
        )

    if "ttc" in name and extract_ttc_features is not None:
        return (
            TTC_PARAM_SPECS,
            lambda feats, roles, t0, t1: extract_ttc_features(
                feats, roles, t0, t1, left_is_decreasing=left_is_decreasing
            ),
        )

    # generischer Fallback: Parameter direkt aus OSC2-Calls ableiten
    print(
        f"[INFO] kein spezifischer Extraktor für '{name}' "
        f"— nutze generischen OSC2-Extraktor",
        flush=True,
    )
    return make_generic_extractor(calls, left_is_decreasing=left_is_decreasing)


# ─────────────────────────────────────────────────────────────────────────────
# Source-Auswahl
# ─────────────────────────────────────────────────────────────────────────────

def _build_source(base_prefix: str, bucket: str, endpoint_url: str,
                  verify: str, min_lanes: Optional[int], local: bool = False):
    if local or Path(base_prefix).exists():
        scenes_dir = str(Path(base_prefix) / "scenes")
        print(f"[INFO] lokale Quelle: {scenes_dir}", flush=True)
        return LocalParquetSource(scenes_dir=scenes_dir, min_lanes=min_lanes)

    print(f"[INFO] S3-Quelle: s3://{bucket}/{base_prefix}/scenes/", flush=True)
    return ParquetSource(
        bucket=bucket,
        base_prefix=base_prefix,
        endpoint_url=endpoint_url,
        verify=verify,
        min_lanes=min_lanes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Kern-Logik
# ─────────────────────────────────────────────────────────────────────────────

def run_one_prefix(
    *,
    osc_prefix: str,
    osc_file: str,
    fps: int,
    overlap: str,
    use_sed: bool,
    bucket: str,
    base_prefix: str,
    endpoint_url: str,
    verify: str,
    out_dir: Path,
    n_pickles_limit: Optional[int],
    write_hits_jsonl: bool,
    write_example_windows_jsonl: bool,
    n_hit_samples: int,
    n_base_samples: int,
    seed: int,
    max_bindings_per_seg: int,
    left_is_decreasing: bool,
    collect_uncond_denoms: bool,
    collect_calls: bool,
    collect_checks: bool,
    collect_params: bool,
    baseline_only_for_hit_bindings: bool,
    segment_stats_mode: bool,
) -> Tuple[Path, Path, Path]:

    out_dir.mkdir(parents=True, exist_ok=True)
    hits_path    = out_dir / "hits_windows.jsonl"
    stats_path   = out_dir / "stats_shard.json"
    example_path = out_dir / "example_windows.jsonl"

    # ── OSC2 parsen & kompilieren ─────────────────────────────────────────────
    prog = OSCProgram(osc_path=os.path.join(osc_prefix, osc_file)).compile()
    scn  = prog.constraints_by_scenario["top"]

    calls = []
    dur_by_label = getattr(prog, "block_durations", {}) or {}
    for c in prog.calls:
        c2  = dict(c)
        lbl = c2.get("block_label")
        if lbl in dur_by_label and c2.get("duration") is None:
            c2["block_duration"] = dur_by_label[lbl]
        c2.setdefault("block_overlap", overlap)
        calls.append(c2)

    print(calls, flush=True)

    plans = build_block_plans(calls, fps=fps)

    need_block_windows = bool(
        write_hits_jsonl or collect_params or collect_calls or collect_checks
    )
    for p in plans.values():
        p.collect_block_windows = need_block_windows

    # ── Engine ───────────────────────────────────────────────────────────────
    cfg = HarnessConfig(
        fps=fps,
        exact_lanes=prog.min_lanes,
        debug_match=False,
        debug_segments=False,
        debug_checks=False,
    )
    cfg.use_sed           = bool(use_sed)
    cfg.debug_pcs         = True
    cfg.first_window_only = True

    engine = MatchEngine(cfg=cfg, scn_constraints=scn, calls=calls)

    # ── Extraktor (generisch oder szenario-spezifisch) ────────────────────────
    param_specs, extractor = _pick_extractor(
        osc_file=osc_file,
        calls=calls,
        left_is_decreasing=left_is_decreasing,
    )

    # window_extractor: einheitlich über param_specs gebaut
    window_extractor = lambda feats, roles, t0, t1: extract_params_for_window(
        feats, roles, t0, t1,
        param_specs=param_specs,
        left_is_decreasing=bool(left_is_decreasing),
    )

    # ── StatsCollector ────────────────────────────────────────────────────────
    collector = StatsCollector(
        osc=os.path.basename(osc_file),
        fps=fps,
        overlap=overlap,
        use_sed=bool(use_sed),
        scn_constraints=scn,
        calls=calls,
        plans=plans,
        param_specs=param_specs,
        extractor=extractor,
        n_hit_samples=n_hit_samples,
        n_base_samples=n_base_samples,
        seed=seed,
        max_bindings_per_seg=max_bindings_per_seg,
        collect_uncond_denoms=bool(collect_uncond_denoms),
        collect_calls=bool(collect_calls),
        collect_checks=bool(collect_checks),
        collect_params=bool(collect_params),
        baseline_only_for_hit_bindings=bool(baseline_only_for_hit_bindings),
    )

    # ── Datenquelle ───────────────────────────────────────────────────────────
    src = _build_source(
        base_prefix=base_prefix,
        bucket=bucket,
        endpoint_url=endpoint_url,
        verify=verify,
        min_lanes=cfg.min_lanes if not segment_stats_mode else None,
    )

    # ── segment_stats_mode: nur Segment-Metadaten schreiben ──────────────────
    if segment_stats_mode:
        stats_path = out_dir / "segment_summary.jsonl"
        out_dir.mkdir(parents=True, exist_ok=True)
        with stats_path.open("a", encoding="utf-8") as f:
            for res in src:
                for seg_id, feats in res.feats_by_seg.items():
                    meta = res.seg_meta_by_id.get(seg_id)
                    if meta is None:
                        continue
                    row = {
                        "source_uri": meta.source_uri,
                        "seg_id":     seg_id,
                        "num_lanes":  int(getattr(feats, "num_lanes", 0) or 0),
                        "length_m":   float(getattr(feats, "length_m", 0.0) or 0.0),
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return hits_path, stats_path, example_path

    # ── Output-Dateien initialisieren ─────────────────────────────────────────
    if write_hits_jsonl:
        with hits_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({
                "type":        "header",
                "osc":         os.path.basename(osc_file),
                "fps":         fps,
                "overlap":     overlap,
                "use_sed":     bool(use_sed),
                "base_prefix": base_prefix,
            }, ensure_ascii=False) + "\n")

    if write_example_windows_jsonl:
        example_path.parent.mkdir(parents=True, exist_ok=True)
        example_path.write_text("", encoding="utf-8")

    # ── Haupt-Schleife ────────────────────────────────────────────────────────
    processed = 0

    for res in src:
        if n_pickles_limit is not None and processed >= int(n_pickles_limit):
            break

        engine.set_features(res.feats_by_seg, res.seg_meta_by_id)

        meta_any   = next(iter(res.seg_meta_by_id.values()), None)
        source_uri = getattr(meta_any, "source_uri", "<unknown>")

        batch = engine.process_loaded_features_with_plans(
            plans=plans,
            source_uri=source_uri,
            collect_call_windows=bool(collect_calls),
            collect_modifier_stats=bool(collect_checks),
        )

        # debug summary
        bh       = batch.block_hits or {}
        total_bs = sum(len(hm or {}) for hm in bh.values())
        with_ex  = sum(
            1 for hm in bh.values()
            for bs in (hm or {}).values()
            if getattr(bs, "example_window", None)
        )
        print(
            f"[BH] {source_uri} "
            f"total_bs={total_bs} with_example_window={with_ex}",
            flush=True,
        )

        # stats sammeln
        collector.observe_pickle(
            feats_by_seg=res.feats_by_seg,
            batch=batch,
            source_uri=source_uri,
        )

        # example windows schreiben
        if write_example_windows_jsonl:
            n_ex = append_example_windows_jsonl(
                path=out_dir / "example_windows.jsonl",
                osc=os.path.basename(osc_file),
                fps=engine.cfg.fps,
                source_uri=source_uri,
                block_hits=batch.block_hits,
                feats_by_seg=engine.h.feats_by_seg,
                include_counts=True,
                extractor=window_extractor,
            )
            print(f"[EX] {n_ex} example windows für {source_uri}", flush=True)

        # hits jsonl schreiben
        if write_hits_jsonl:
            with hits_path.open("a", encoding="utf-8") as f:
                for block_label, hitmap in (batch.block_hits or {}).items():
                    plan = plans.get(block_label)
                    if plan is None:
                        continue
                    minF = int(getattr(plan, "duration_min_frames", 1) or 1)

                    for (seg_id, _rk), bs in (hitmap or {}).items():
                        roles = dict(getattr(bs, "roles", {}) or {})
                        T     = (
                            int(getattr(bs, "T", 0) or 0)
                            or int(getattr(res.feats_by_seg.get(seg_id), "T", 0) or 0)
                        )
                        maxF      = getattr(plan, "duration_max_frames", None)
                        maxF      = int(maxF) if maxF is not None else int(T)
                        intervals = _intervals_to_jsonable(getattr(bs, "intervals", None))
                        wbt0      = getattr(bs, "windows_by_t0", None)
                        wbt0_json = _windows_by_t0_to_jsonable(wbt0)
                        nwin      = (
                            int(count_windows(wbt0)) if wbt0
                            else int(getattr(bs, "n_windows", 0) or 0)
                        )
                        nposs = (
                            int(max_possible_windows(T, minF, maxF)) if T
                            else int(getattr(bs, "n_possible_windows", 0) or 0)
                        )
                        rec = {
                            "type":               "block_hit",
                            "osc":                os.path.basename(osc_file),
                            "block":              block_label,
                            "segment_id":         seg_id,
                            "roles":              roles,
                            "T":                  int(T),
                            "fps":                int(fps),
                            "intervals":          intervals,
                            "windows_by_t0":      wbt0_json,
                            "n_windows":          int(nwin),
                            "n_possible_windows": int(nposs),
                            "source_uri":         source_uri,
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        engine.clear_features()
        processed += 1
        print(processed, source_uri, flush=True)

    # stats schreiben
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(collector.to_json(), f, indent=2, ensure_ascii=False)

    return hits_path, stats_path, example_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="OSC2 Szenario-Matcher — lokal oder als Kubernetes Job"
    )

    # OSC / Matcher-Config
    ap.add_argument("--osc_prefix",  default=_env_str("OSC_PREFIX",  "osc2_parser/osc/"))
    ap.add_argument("--osc_file",    default=_env_str("OSC_FILE",    "change_lane.osc"))
    ap.add_argument("--fps",         type=int, default=_env_int("FPS", 10))
    ap.add_argument("--overlap",     default=_env_str("BLOCK_OVERLAP", "start"))
    _add_bool_opt(ap, "--use_sed",
                  default=_env_bool("USE_SED", True),
                  help="S/E/D Fenster-Semantik im Matcher verwenden.")
    _add_bool_opt(ap, "--left_is_decreasing",
                  default=_env_bool("LEFT_IS_DECREASING", True),
                  help="Linke Spur = abnehmender Index.")

    # Input (S3 oder lokal)
    ap.add_argument("--input_bucket",
                    default=_env_str("INPUT_BUCKET", "waymo"))
    ap.add_argument("--base_prefix_root",
                    default=_env_str("BASE_PREFIX_ROOT", "results/k8s_run-1331520"))
    ap.add_argument("--base_prefix",
                    default=os.getenv("BASE_PREFIX"))
    ap.add_argument("--endpoint_url",
                    default=_env_str("S3_ENDPOINT_URL", "https://gif.s3.iavgroup.local"))
    ap.add_argument("--verify",
                    default=_env_str("S3_VERIFY_PATH",
                                     _env_str("AWS_CA_BUNDLE", "certs/IAV-CA-Bundle.pem")))

    # Output
    ap.add_argument("--out_dir", default=_env_str("OUT_DIR", "out_local"))
    _add_bool_opt(ap, "--write_hits_jsonl",
                  default=_env_bool("WRITE_HITS_JSONL", False),
                  help="hits_windows.jsonl schreiben.")
    _add_bool_opt(ap, "--write_example_windows_jsonl",
                  default=_env_bool("WRITE_EXAMPLE_WINDOWS_JSONL", True),
                  help="example_windows.jsonl schreiben.")

    # Stats-Sampling
    ap.add_argument("--n_hit_samples",
                    type=int, default=_env_int("HIT_SAMPLES_PER_BINDING", 30))
    ap.add_argument("--n_base_samples",
                    type=int, default=_env_int("BASE_SAMPLES_PER_BINDING", 30))
    ap.add_argument("--n_hit_samples_per_binding",
                    type=int, dest="n_hit_samples", default=None)
    ap.add_argument("--n_base_samples_per_binding",
                    type=int, dest="n_base_samples", default=None)
    ap.add_argument("--seed",
                    type=int, default=_env_int("SEED", 0))
    ap.add_argument("--max_bindings_per_seg",
                    type=int, default=_env_int("MAX_BINDINGS_PER_SEG", 20000))

    # Collector-Toggles
    _add_bool_opt(ap, "--collect_uncond_denoms",
                  default=_env_bool("COLLECT_UNCOND_DENOMS", True),
                  help="Unbedingte Nenner sammeln.")
    _add_bool_opt(ap, "--collect_calls",
                  default=_env_bool("COLLECT_CALLS", False),
                  help="Per-Call Stats sammeln.")
    _add_bool_opt(ap, "--collect_checks",
                  default=_env_bool("COLLECT_CHECKS", False),
                  help="Per-Check S/E/D Stats sammeln.")
    _add_bool_opt(ap, "--collect_params",
                  default=_env_bool("COLLECT_PARAMS", True),
                  help="Parameter-Histogramme sammeln.")
    _add_bool_opt(ap, "--baseline_only_for_hit_bindings",
                  default=_env_bool("BASELINE_ONLY_FOR_HIT_BINDINGS", False),
                  help="Baseline nur für Hit-Bindings sampeln.")
    _add_bool_opt(ap, "--segment_stats_mode",
                  default=_env_bool("SEGMENT_STATS_MODE", False),
                  help="Nur Segment-Metadaten-JSONL schreiben, kein Matching.")
    _add_bool_opt(ap, "--local",
              default=_env_bool("LOCAL_MODE", False),
              help="Lokale Parquet-Dateien statt S3 verwenden.")

    # Lokales Limit
    ap.add_argument("--n_pickles", type=int, default=_env_int("N_PICKLES", -1),
                help="Max. Anzahl Scenes lokal zu verarbeiten. -1 = alle.")

    # Kubernetes-Shard
    ap.add_argument("--job_index",
                    type=int, default=_env_int("JOB_COMPLETION_INDEX", -1))
    ap.add_argument("--start_offset",
                    type=int, default=_env_int("START_OFFSET", 0))
    ap.add_argument("--max_shard_index",
                    type=int, default=_env_int("MAX_SHARD_INDEX", -1))

    # S3-Upload der Outputs
    _add_bool_opt(ap, "--upload",
                  default=_env_bool("UPLOAD_RESULTS", False),
                  help="Outputs nach S3 hochladen.")
    ap.add_argument("--output_bucket", default=_env_str("OUTPUT_BUCKET", ""))
    ap.add_argument("--output_prefix", default=_env_str("OUTPUT_PREFIX", ""))

    args = ap.parse_args()

    # Alias-Auflösung
    if args.n_hit_samples is None:
        args.n_hit_samples  = _env_int("HIT_SAMPLES_PER_BINDING",  30)
    if args.n_base_samples is None:
        args.n_base_samples = _env_int("BASE_SAMPLES_PER_BINDING", 30)

    # base_prefix bestimmen
    if args.base_prefix:
        base_prefix = args.base_prefix
        shard_tag   = "manual"
    elif args.job_index >= 0 and args.max_shard_index >= 0:
        shard_index = int(args.start_offset) + int(args.job_index)
        if shard_index > int(args.max_shard_index):
            print(f"[INFO] shard_index={shard_index} > max_shard_index={args.max_shard_index}; beende.")
            return 0
        base_prefix = _compute_shard_prefix(args.base_prefix_root, shard_index)
        shard_tag   = f"{shard_index:05d}"
    else:
        base_prefix = args.base_prefix_root
        shard_tag   = "local"

    out_dir = Path(args.out_dir) / os.path.basename(args.osc_file) / shard_tag
    _debug_dump(args, base_prefix, shard_tag)

    hits_path, stats_path, example_path = run_one_prefix(
        osc_prefix=args.osc_prefix,
        osc_file=args.osc_file,
        fps=args.fps,
        overlap=args.overlap,
        use_sed=bool(args.use_sed),
        bucket=args.input_bucket,
        base_prefix=base_prefix,
        endpoint_url=args.endpoint_url,
        verify=args.verify,
        out_dir=out_dir,
        n_pickles_limit=(
            None if args.job_index >= 0          # Kubernetes: immer alle
            else None if args.n_pickles < 0      # lokal: -1 = alle
            else int(args.n_pickles)             # lokal: explizites Limit
        ),
        write_hits_jsonl=bool(args.write_hits_jsonl),
        write_example_windows_jsonl=bool(args.write_example_windows_jsonl),
        n_hit_samples=int(args.n_hit_samples),
        n_base_samples=int(args.n_base_samples),
        seed=int(args.seed),
        max_bindings_per_seg=int(args.max_bindings_per_seg),
        left_is_decreasing=bool(args.left_is_decreasing),
        collect_uncond_denoms=bool(args.collect_uncond_denoms),
        collect_calls=bool(args.collect_calls),
        collect_checks=bool(args.collect_checks),
        collect_params=bool(args.collect_params),
        baseline_only_for_hit_bindings=bool(args.baseline_only_for_hit_bindings),
        segment_stats_mode=bool(args.segment_stats_mode),
    )

    print(f"[OK] outputs geschrieben:\n  {hits_path}\n  {stats_path}\n  {example_path}")

    if bool(args.upload):
        if not args.output_bucket or not args.output_prefix:
            print("[WARN] --upload gesetzt aber OUTPUT_BUCKET/OUTPUT_PREFIX fehlen; überspringe.")
            return 0

        osc_name = os.path.basename(args.osc_file)
        key_base = f"{args.output_prefix.rstrip('/')}/{osc_name}/{shard_tag}"

        try:
            if bool(args.write_hits_jsonl) and hits_path.exists():
                _s3_put_file(
                    bucket=args.output_bucket,
                    key=f"{key_base}/hits_windows.jsonl",
                    path=hits_path,
                    endpoint_url=args.endpoint_url,
                    verify=args.verify,
                )
            if stats_path.exists():
                _s3_put_file(
                    bucket=args.output_bucket,
                    key=f"{key_base}/{stats_path.name}",
                    path=stats_path,
                    endpoint_url=args.endpoint_url,
                    verify=args.verify,
                )
            if bool(args.write_example_windows_jsonl) and example_path.exists():
                _s3_put_file(
                    bucket=args.output_bucket,
                    key=f"{key_base}/example_windows.jsonl",
                    path=example_path,
                    endpoint_url=args.endpoint_url,
                    verify=args.verify,
                )
            print(f"[OK] hochgeladen nach s3://{args.output_bucket}/{key_base}/")
        except Exception as e:
            print(f"[ERROR] Upload fehlgeschlagen: {e!r}", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())