#!/usr/bin/env python3
"""
run_matching.py

Läuft sowohl lokal (via CLI-Args) als auch als Kubernetes / AWS Batch Job
(via Env-Vars).

Input:  Parquet-Dateien auf S3 oder lokalem Dateisystem
        (erzeugt von worker.py)

Output: drei Parquet-Tabellen pro Shard, Hive-partitioniert für Athena:
  results/match_hits/scenario=.../run_id=.../shard=XXXXX.parquet
  results/match_actor_frames/...
  results/match_pair_frames/...
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

# --- core pipeline ---
from osc2_parser.parser import OSCProgram
from scenario_matching.matching.post.plan import build_block_plans
from scenario_matching.matching.engine import MatchEngine
from scenario_matching.harness import HarnessConfig
from scenario_matching.analysis_stats.stats_windows import (
    max_possible_windows,
    count_windows,
)

# --- Parquet-Source ---
from parquet_source import LocalParquetSource, ParquetSource


# ─────────────────────────────────────────────────────────────────────────────
# Parquet Schemas
# ─────────────────────────────────────────────────────────────────────────────

HITS_SCHEMA = pa.schema([
    pa.field("run_id",             pa.string()),
    pa.field("scenario",           pa.string()),
    pa.field("shard_index",        pa.int32()),
    pa.field("scene_id",           pa.string()),
    pa.field("segment_id",         pa.string()),
    pa.field("block_label",        pa.string()),
    pa.field("roles_json",         pa.string()),
    pa.field("t0",                 pa.int32()),
    pa.field("t1",                 pa.int32()),
    pa.field("n_windows",          pa.int32()),
    pa.field("n_possible_windows", pa.int32()),
    pa.field("source_uri",         pa.string()),
])

ACTOR_FRAMES_SCHEMA = pa.schema([
    pa.field("run_id",      pa.string()),
    pa.field("scenario",    pa.string()),
    pa.field("scene_id",    pa.string()),
    pa.field("segment_id",  pa.string()),
    pa.field("t0",          pa.int32()),
    pa.field("t1",          pa.int32()),
    pa.field("role",        pa.string()),
    pa.field("actor_id",    pa.string()),
    pa.field("frame",       pa.string()),
    pa.field("t",           pa.int32()),
    pa.field("x",           pa.float64()),
    pa.field("y",           pa.float64()),
    pa.field("yaw",         pa.float64()),
    pa.field("speed",       pa.float64()),
    pa.field("accel",       pa.float64()),
    pa.field("s",           pa.float64()),
    pa.field("t_lat",       pa.float64()),
    pa.field("s_dot",       pa.float64()),
    pa.field("t_dot",       pa.float64()),
    pa.field("yaw_delta",   pa.float64()),
    pa.field("osc_lane_id", pa.float64()),
])

PAIR_FRAMES_SCHEMA = pa.schema([
    pa.field("run_id",       pa.string()),
    pa.field("scenario",     pa.string()),
    pa.field("scene_id",     pa.string()),
    pa.field("segment_id",   pa.string()),
    pa.field("t0",           pa.int32()),
    pa.field("t1",           pa.int32()),
    pa.field("role_a",       pa.string()),
    pa.field("role_b",       pa.string()),
    pa.field("actor_a",      pa.string()),
    pa.field("actor_b",      pa.string()),
    pa.field("frame",        pa.string()),
    pa.field("t",            pa.int32()),
    pa.field("rel_distance", pa.float64()),
    pa.field("ttc",          pa.float64()),
    pa.field("rel_position", pa.string()),
    pa.field("lat_rel",      pa.string()),
])


# ─────────────────────────────────────────────────────────────────────────────
# Feature-Zugriffs-Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_val(feat_dict, actor_id: str, t: int) -> Optional[float]:
    arr = (feat_dict or {}).get(actor_id)
    if arr is None:
        return None
    try:
        v = float(arr[t])
        return None if (v != v or abs(v) == float("inf")) else v
    except (IndexError, TypeError):
        return None


def _safe_pair(feat_dict, a: str, b: str, t: int) -> Optional[float]:
    arr = (feat_dict or {}).get((a, b)) or (feat_dict or {}).get((b, a))
    if arr is None:
        return None
    try:
        v = float(arr[t])
        return None if (v != v or abs(v) == float("inf")) else v
    except (IndexError, TypeError):
        return None


def _safe_pair_str(feat_dict, a: str, b: str, t: int) -> Optional[str]:
    arr = (feat_dict or {}).get((a, b)) or (feat_dict or {}).get((b, a))
    if arr is None:
        return None
    try:
        return str(arr[t])
    except (IndexError, TypeError):
        return None


def _first_window(wbt0) -> Tuple[Optional[int], Optional[int]]:
    if not wbt0:
        return None, None
    for t0, ranges in sorted(wbt0.items()):
        if ranges:
            lo, hi = ranges[0]
            return int(t0), int(hi)
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# ResultsWriter
# ─────────────────────────────────────────────────────────────────────────────

class ResultsWriter:
    """
    Sammelt Hit-Rows waehrend eines Shard-Runs und schreibt am Ende
    drei Parquet-Tabellen nach S3 oder auf lokale Disk.

    S3-Partitionierung (Hive-kompatibel fuer Athena):
      {prefix}/{table}/scenario={scenario}/run_id={run_id}/shard={N:05d}.parquet
    """

    def __init__(
        self,
        run_id: str,
        scenario: str,
        shard_index: int,
        bucket: str,
        prefix: str,
        endpoint_url: Optional[str] = None,
        verify: Optional[str] = None,
        local_dir: Optional[str] = None,
    ):
        self.run_id       = run_id
        self.scenario     = scenario
        self.shard_index  = shard_index
        self.bucket       = bucket
        self.prefix       = prefix.rstrip("/")
        self.endpoint_url = endpoint_url
        self.verify       = verify
        self.local_dir    = local_dir

        self._hits:         list = []
        self._actor_frames: list = []
        self._pair_frames:  list = []

    def add_hit(
        self,
        *,
        scene_id: str,
        segment_id: str,
        block_label: str,
        roles: dict,
        t0: int,
        t1: int,
        n_windows: int,
        n_possible_windows: int,
        source_uri: str,
        feats,
    ) -> None:
        base = dict(
            run_id=self.run_id,
            scenario=self.scenario,
            scene_id=scene_id,
            segment_id=segment_id,
            t0=int(t0),
            t1=int(t1),
        )

        self._hits.append({
            **base,
            "shard_index":        self.shard_index,
            "block_label":        block_label,
            "roles_json":         json.dumps(roles),
            "n_windows":          int(n_windows),
            "n_possible_windows": int(n_possible_windows),
            "source_uri":         source_uri,
        })

        role_list = list(roles.items())

        for frame_label, t in [("start", t0), ("end", t1)]:

            for role, actor_id in role_list:
                self._actor_frames.append({
                    **base,
                    "role":        role,
                    "actor_id":    actor_id,
                    "frame":       frame_label,
                    "t":           int(t),
                    "x":           _safe_val(feats.x,         actor_id, t),
                    "y":           _safe_val(feats.y,         actor_id, t),
                    "yaw":         _safe_val(feats.yaw,       actor_id, t),
                    "speed":       _safe_val(feats.speed,     actor_id, t),
                    "accel":       _safe_val(feats.accel,     actor_id, t),
                    "s":           _safe_val(feats.s,         actor_id, t),
                    "t_lat":       _safe_val(feats.t,         actor_id, t),
                    "s_dot":       _safe_val(feats.s_dot,     actor_id, t),
                    "t_dot":       _safe_val(feats.t_dot,     actor_id, t),
                    "yaw_delta":   _safe_val(feats.yaw_delta, actor_id, t),
                    "osc_lane_id": _safe_val(feats.lane_idx,  actor_id, t),
                })

            for i in range(len(role_list)):
                for j in range(len(role_list)):
                    if i == j:
                        continue
                    role_a, actor_a = role_list[i]
                    role_b, actor_b = role_list[j]
                    self._pair_frames.append({
                        **base,
                        "role_a":       role_a,
                        "role_b":       role_b,
                        "actor_a":      actor_a,
                        "actor_b":      actor_b,
                        "frame":        frame_label,
                        "t":            int(t),
                        "rel_distance": _safe_pair(
                            feats.rel_distance, actor_a, actor_b, t),
                        "ttc":          _safe_pair(
                            feats.ttc, actor_a, actor_b, t),
                        "rel_position": _safe_pair_str(
                            feats.rel_position, actor_a, actor_b, t),
                        "lat_rel":      _safe_pair_str(
                            feats.lat_rel, actor_a, actor_b, t),
                    })

    def flush(self) -> dict:
        written = {}
        for table_name, rows, schema in [
            ("match_hits",         self._hits,         HITS_SCHEMA),
            ("match_actor_frames", self._actor_frames, ACTOR_FRAMES_SCHEMA),
            ("match_pair_frames",  self._pair_frames,  PAIR_FRAMES_SCHEMA),
        ]:
            if not rows:
                print(f"[results] {table_name}: keine Rows", flush=True)
                continue

            table = pa.Table.from_pylist(rows, schema=schema)

            if self.local_dir:
                out = Path(self.local_dir) / table_name
                out.mkdir(parents=True, exist_ok=True)
                path = out / f"shard_{self.shard_index:05d}.parquet"
                pq.write_table(table, path, compression="snappy")
                written[table_name] = str(path)
            else:
                key = (
                    f"{self.prefix}/{table_name}"
                    f"/scenario={self.scenario}"
                    f"/run_id={self.run_id}"
                    f"/shard={self.shard_index:05d}.parquet"
                )
                buf = io.BytesIO()
                pq.write_table(table, buf, compression="snappy")
                buf.seek(0)
                self._s3().upload_fileobj(buf, self.bucket, key)
                written[table_name] = f"s3://{self.bucket}/{key}"

            print(f"[results] {len(rows):>6} rows -> {written[table_name]}", flush=True)

        return written

    def _s3(self):
        import boto3
        kwargs: dict = {}
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        if self.verify:
            kwargs["verify"] = self.verify
        return boto3.client("s3", **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Env / Arg Helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def _add_bool_opt(ap, name, default, help):
    ap.add_argument(
        name, nargs="?", const=True,
        default=default, type=_parse_bool, help=help,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Path / Source Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_shard_prefix(base_prefix_root: str, shard_index: int) -> str:
    return f"{base_prefix_root.rstrip('/')}/{shard_index:05d}"


def _resolve_prefix(base_prefix: str) -> Path:
    _repo_root = Path(__file__).resolve().parent.parent
    p = Path(base_prefix)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    candidate = _repo_root / p
    if candidate.exists():
        return candidate.resolve()
    return p.resolve()


def _build_source(
    base_prefix: str,
    bucket: str,
    endpoint_url: str,
    verify: str,
    min_lanes: Optional[int],
    local: bool,
):
    if local:
        resolved   = _resolve_prefix(base_prefix)
        scenes_dir = str(resolved / "scenes")
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


def _debug_dump(args, base_prefix: str, shard_tag: str) -> None:
    print("=== MATCHER DEBUG CONFIG ===")
    print(json.dumps({
        "resolved": {"base_prefix": base_prefix, "shard_tag": shard_tag},
        "cli_args": vars(args),
    }, indent=2, sort_keys=True), flush=True)


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
    run_id: str,
    shard_index: int,
    bucket: str,
    base_prefix: str,
    endpoint_url: str,
    verify: str,
    out_dir: Path,
    results_bucket: str,
    results_prefix: str,
    n_scenes_limit: Optional[int],
    left_is_decreasing: bool,
    local: bool,
    segment_stats_mode: bool,
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    # OSC2 parsen und kompilieren
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
    for p in plans.values():
        p.collect_block_windows = True

    # Engine
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

    # ResultsWriter
    writer = ResultsWriter(
        run_id=run_id,
        scenario=os.path.basename(osc_file),
        shard_index=shard_index,
        bucket=results_bucket,
        prefix=results_prefix,
        endpoint_url=endpoint_url,
        verify=verify,
        local_dir=str(out_dir) if local else None,
    )

    # Datenquelle
    src = _build_source(
        base_prefix=base_prefix,
        bucket=bucket,
        endpoint_url=endpoint_url,
        verify=verify,
        min_lanes=cfg.min_lanes if not segment_stats_mode else None,
        local=local,
    )

    # segment_stats_mode: nur Metadaten schreiben
    if segment_stats_mode:
        seg_path = out_dir / "segment_summary.jsonl"
        with seg_path.open("a", encoding="utf-8") as f:
            for res in src:
                for seg_id, feats in res.feats_by_seg.items():
                    meta = res.seg_meta_by_id.get(seg_id)
                    if meta is None:
                        continue
                    f.write(json.dumps({
                        "source_uri": meta.source_uri,
                        "seg_id":     seg_id,
                        "num_lanes":  int(getattr(feats, "num_lanes", 0) or 0),
                        "length_m":   float(getattr(feats, "length_m", 0.0) or 0.0),
                    }, ensure_ascii=False) + "\n")
        return

    # Haupt-Schleife
    processed  = 0
    total_hits = 0

    for res in src:
        if n_scenes_limit is not None and processed >= n_scenes_limit:
            break

        engine.set_features(res.feats_by_seg, res.seg_meta_by_id)

        meta_any   = next(iter(res.seg_meta_by_id.values()), None)
        source_uri = getattr(meta_any, "source_uri", "<unknown>")
        scene_id   = Path(source_uri).stem

        batch = engine.process_loaded_features_with_plans(
            plans=plans,
            source_uri=source_uri,
            collect_call_windows=True,
            collect_modifier_stats=False,
        )

        for block_label, hitmap in (batch.block_hits or {}).items():
            plan = plans.get(block_label)
            if plan is None:
                continue
            minF = int(getattr(plan, "duration_min_frames", 1) or 1)

            for (seg_id, _rk), bs in (hitmap or {}).items():
                roles = dict(getattr(bs, "roles", {}) or {})
                feats = res.feats_by_seg.get(seg_id)
                if feats is None:
                    continue

                wbt0   = getattr(bs, "windows_by_t0", None)
                t0, t1 = _first_window(wbt0)
                if t0 is None:
                    continue

                T     = int(getattr(feats, "T", 91) or 91)
                maxF  = int(getattr(plan, "duration_max_frames", None) or T)
                nwin  = int(count_windows(wbt0))
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
        processed += 1
        print(f"[{processed}] {source_uri} -- {total_hits} hits gesamt", flush=True)

    written = writer.flush()
    print(f"\n[OK] {processed} scenes, {total_hits} hits", flush=True)
    for table, path in written.items():
        print(f"  {table} -> {path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="OSC2 Szenario-Matcher -- schreibt Ergebnisse als Parquet fuer Athena"
    )

    ap.add_argument("--osc_prefix",
                    default=_env_str("OSC_PREFIX", "osc2_parser/osc/"))
    ap.add_argument("--osc_file",
                    default=_env_str("OSC_FILE", "change_lane.osc"))
    ap.add_argument("--fps",
                    type=int, default=_env_int("FPS", 10))
    ap.add_argument("--overlap",
                    default=_env_str("BLOCK_OVERLAP", "start"))
    _add_bool_opt(ap, "--use_sed",
                  default=_env_bool("USE_SED", True),
                  help="S/E/D Fenster-Semantik verwenden.")
    _add_bool_opt(ap, "--left_is_decreasing",
                  default=_env_bool("LEFT_IS_DECREASING", True),
                  help="Linke Spur = abnehmender Index.")

    ap.add_argument("--run_id",
                    default=_env_str("RUN_ID", "run-001"),
                    help="Eindeutiger Run-Bezeichner.")

    ap.add_argument("--input_bucket",
                    default=_env_str("INPUT_BUCKET", "waymo"))
    ap.add_argument("--base_prefix_root",
                    default=_env_str("BASE_PREFIX_ROOT", "parquet"))
    ap.add_argument("--base_prefix",
                    default=os.getenv("BASE_PREFIX"))
    ap.add_argument("--endpoint_url",
                    default=_env_str("S3_ENDPOINT_URL", "https://gif.s3.iavgroup.local"))
    ap.add_argument("--verify",
                    default=_env_str("S3_VERIFY_PATH",
                                     _env_str("AWS_CA_BUNDLE", "certs/IAV-CA-Bundle.pem")))
    _add_bool_opt(ap, "--local",
                  default=_env_bool("LOCAL_MODE", False),
                  help="Lokale Parquet-Dateien statt S3.")

    ap.add_argument("--out_dir",
                    default=_env_str("OUT_DIR", "out_local"))
    ap.add_argument("--results_bucket",
                    default=_env_str("RESULTS_BUCKET", ""))
    ap.add_argument("--results_prefix",
                    default=_env_str("RESULTS_PREFIX", "results"))

    ap.add_argument("--n_scenes",
                    type=int, default=_env_int("N_SCENES", -1),
                    help="Max. Scenes. -1 = alle.")

    ap.add_argument("--job_index",
                    type=int, default=_env_int("JOB_COMPLETION_INDEX", -1))
    ap.add_argument("--start_offset",
                    type=int, default=_env_int("START_OFFSET", 0))
    ap.add_argument("--max_shard_index",
                    type=int, default=_env_int("MAX_SHARD_INDEX", -1))

    _add_bool_opt(ap, "--segment_stats_mode",
                  default=_env_bool("SEGMENT_STATS_MODE", False),
                  help="Nur Segment-Metadaten schreiben.")

    args = ap.parse_args()

    # base_prefix und shard bestimmen
    if args.base_prefix:
        base_prefix = args.base_prefix
        shard_tag   = "manual"
        shard_index = 0
    elif args.job_index >= 0 and args.max_shard_index >= 0:
        shard_index = int(args.start_offset) + int(args.job_index)
        if shard_index > int(args.max_shard_index):
            print(f"[INFO] shard_index={shard_index} > max; beende.")
            return 0
        base_prefix = _compute_shard_prefix(args.base_prefix_root, shard_index)
        shard_tag   = f"{shard_index:05d}"
    else:
        base_prefix = args.base_prefix_root
        shard_tag   = "local"
        shard_index = 0

    n_scenes_limit = None if args.n_scenes < 0 else int(args.n_scenes)
    out_dir = Path(args.out_dir) / os.path.basename(args.osc_file) / shard_tag

    _debug_dump(args, base_prefix, shard_tag)

    run_one_prefix(
        osc_prefix=args.osc_prefix,
        osc_file=args.osc_file,
        fps=args.fps,
        overlap=args.overlap,
        use_sed=bool(args.use_sed),
        run_id=args.run_id,
        shard_index=shard_index,
        bucket=args.input_bucket,
        base_prefix=base_prefix,
        endpoint_url=args.endpoint_url,
        verify=args.verify,
        out_dir=out_dir,
        results_bucket=args.results_bucket or args.input_bucket,
        results_prefix=args.results_prefix,
        n_scenes_limit=n_scenes_limit,
        left_is_decreasing=bool(args.left_is_decreasing),
        local=bool(args.local),
        segment_stats_mode=bool(args.segment_stats_mode),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())