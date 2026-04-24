import os
import io
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import boto3
import tensorflow as tf

from feature_extraction.pipeline import process_scenario
from feature_extraction.tools.scenario import Scenario, features_description
from shapely.geometry import mapping

# ─────────────────────────────────────────────────────────────────────────────
# Config from environment
# ─────────────────────────────────────────────────────────────────────────────

"""GCS_BUCKET    = os.environ["GCS_BUCKET"]
GCS_PREFIX    = os.environ.get("GCS_PREFIX", "")
S3_BUCKET     = os.environ["S3_BUCKET"]
S3_PREFIX     = os.environ.get("S3_PREFIX", "output")
SHARD_INDEX   = int(os.environ["SHARD_INDEX"])
TOTAL_SHARDS  = int(os.environ.get("TOTAL_SHARDS", 1000))"""


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_serializable(obj):
    """Recursively convert numpy / shapely / set types to plain Python."""
    if isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
        return None
    if isinstance(obj, np.floating):
        if obj != obj or obj == np.inf or obj == -np.inf:
            return None
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return make_serializable(obj.item())
        return [make_serializable(x) for x in obj]
    if isinstance(obj, set):
        return list(obj)
    if hasattr(obj, '__geo_interface__'):
        return mapping(obj)
    if isinstance(obj, dict):
        return {
            int(k)   if isinstance(k, np.integer)
            else float(k) if isinstance(k, np.floating)
            else str(k)   if not isinstance(k, (str, int, float, bool, type(None)))
            else k
            : make_serializable(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [make_serializable(i) for i in obj]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Sparse inter-actor encoding
# ─────────────────────────────────────────────────────────────────────────────

def _encode_sparse_series(values: list) -> dict:
    """
    Convert a length-91 list (with None gaps) into a compact representation.

    Returns a dict with:
      intervals: [[t_start, t_end], ...]   inclusive timestep ranges
      data:      [v0, v1, ...]             values for those ranges concatenated

    Example:
      [None, None, 1.2, 1.3, None, 0.9]
      → intervals: [[2, 3], [5, 5]]
        data:      [1.2, 1.3, 0.9]
    """
    intervals = []
    data      = []
    i = 0
    n = len(values)
    while i < n:
        if values[i] is None:
            i += 1
            continue
        # start of a valid run
        j = i
        while j < n and values[j] is not None:
            j += 1
        intervals.append([i, j - 1])
        data.extend(values[i:j])
        i = j
    return {"intervals": intervals, "data": data}


def _encode_sparse_string_series(values: list) -> dict:
    """
    Same as above but for string (position) series.
    Treats None and the string 'unknown' as invalid.
    """
    INVALID = {None, "unknown", ""}
    intervals = []
    data      = []
    i = 0
    n = len(values)
    while i < n:
        if values[i] in INVALID:
            i += 1
            continue
        j = i
        while j < n and values[j] not in INVALID:
            j += 1
        intervals.append([i, j - 1])
        data.extend(values[i:j])
        i = j
    return {"intervals": intervals, "data": data}


def encode_inter_actor_pair(pair_data: dict) -> dict | None:
    """
    Encode one actor-pair dict into a compact sparse form.
    Returns None if the pair has no valid data at all (skip entirely).

    Input:  {"position": [...91 strs...], "ttc": [...91 floats/None...], "eucl_distance": [...]}
    Output: {"ttc": sparse, "eucl_distance": sparse, "position": sparse}
            or None
    """
    ttc      = _encode_sparse_series(pair_data.get("ttc")           or [])
    dist     = _encode_sparse_series(pair_data.get("eucl_distance") or [])
    position = _encode_sparse_string_series(pair_data.get("position") or [])

    # drop completely empty pairs
    if not ttc["data"] and not dist["data"] and not position["data"]:
        return None

    return {"ttc": ttc, "eucl_distance": dist, "position": position}


# ─────────────────────────────────────────────────────────────────────────────
# Scene → flat Parquet rows
# ─────────────────────────────────────────────────────────────────────────────

def scene_to_parquet_rows(result: dict) -> list[dict]:
    """
    Convert one process_scenario() result dict into a list of rows,
    one row per road segment, ready to write as a Parquet file.

    Heavy nested data (geometry, actor arrays, sparse inter-actor) is
    serialised to JSON strings so the Parquet schema stays flat and stable
    across scenes with varying numbers of actors/segments.
    """
    scene_id = result["scene_id"]
    rows = []

    for seg_id, seg_proc in result["processed_road_segments"].items():

        # ── geometry (JSON strings) ──────────────────────────────────────────
        ref_line    = seg_proc.get("reference_line")
        tgt_poly    = seg_proc.get("target_polygon")
        left_poly   = seg_proc.get("left_polygon")
        right_poly  = seg_proc.get("right_polygon")
        centerlines = seg_proc.get("centerline_by_chain")

        # ── env elements ─────────────────────────────────────────────────────
        seg_env     = (result["segment_env_elements"] or {}).get(seg_id, {})

        # ── road segment metadata ─────────────────────────────────────────────
        road_seg    = (result["road_segments"] or {}).get(seg_id, {})
        num_lanes   = int(road_seg.get("num_lanes", 0) or 0)

        # ── actor data for this segment ───────────────────────────────────────
        gad         = result["general_actor_data"]
        seg_actor_ids: list = (gad.get("per_segment_ids") or {}).get(seg_id, [])

        # global actor time series (only actors present in this segment)
        actor_ts: dict = {}
        for actor_id in seg_actor_ids:
            raw = (gad.get("actor_activities") or {}).get(actor_id)
            if raw is None:
                continue
            valid = raw.get("valid") or [None, None]
            actor_ts[actor_id] = {
                "x":           raw.get("x"),
                "y":           raw.get("y"),
                "yaw":         raw.get("yaw"),
                "long_v":      raw.get("long_v"),
                "lane_id":     raw.get("lane_id"),
                "valid_start": valid[0] if len(valid) > 0 else None,
                "valid_end":   valid[1] if len(valid) > 1 else None,
            }

        # frenet / segment-level actor data
        seg_actor_ts: dict = {}
        for actor_id, data in ((result["segment_actor_data"] or {}).get(seg_id) or {}).items():
            valid = data.get("valid") or [None, None]
            seg_actor_ts[actor_id] = {
                "s":           data.get("s"),
                "t":           data.get("t"),
                "yaw_delta":   data.get("yaw_delta"),
                "s_dot":       data.get("s_dot"),
                "t_dot":       data.get("t_dot"),
                "osc_lane_id": data.get("osc_lane_id"),
                "valid_start": valid[0] if len(valid) > 0 else None,
                "valid_end":   valid[1] if len(valid) > 1 else None,
            }

        # sparse inter-actor pairs (only actors in this segment, only non-empty pairs)
        inter_actor: dict = {}
        inter_raw = result.get("inter_actor_activities") or {}
        for actor_a in seg_actor_ids:
            pairs_for_a = (inter_raw or {}).get(actor_a) or {}
            for actor_b in seg_actor_ids:
                if actor_a == actor_b:
                    continue
                pair_data = pairs_for_a.get(actor_b)
                if pair_data is None:
                    continue
                encoded = encode_inter_actor_pair(pair_data)
                if encoded is not None:
                    inter_actor[f"{actor_a}|{actor_b}"] = encoded

        # ── assemble actors blob ──────────────────────────────────────────────
        actors_payload = {
            "actor_ids":    seg_actor_ids,
            "actor_ts":     actor_ts,
            "seg_actor_ts": seg_actor_ts,
            "inter_actor":  inter_actor,   # sparse, keyed "actorA|actorB"
        }

        rows.append({
            "scene_id":              scene_id,
            "segment_id":            seg_id,
            "num_lanes":             num_lanes,
            "num_segments":          int(road_seg.get("num_segments", 0) or 0),
            "target_chain_id":       int(seg_proc.get("target_chain_id", 0) or 0),
            "valid":                 bool(seg_proc.get("valid", True)),
            "reference_line_source": str(seg_proc.get("reference_line_source") or ""),
            # geometry
            "reference_line_json":   json.dumps(ref_line,    allow_nan=False),
            "target_polygon_json":   json.dumps(tgt_poly,    allow_nan=False),
            "left_polygon_json":     json.dumps(left_poly,   allow_nan=False),
            "right_polygon_json":    json.dumps(right_poly,  allow_nan=False),
            "centerlines_json":      json.dumps(centerlines, allow_nan=False),
            # env
            "tl_results_json":       json.dumps(seg_env.get("tl_results", []), allow_nan=False),
            "cw_results_json":       json.dumps(seg_env.get("cw_results", []), allow_nan=False),
            # all actor data in one blob
            "actors_json":           json.dumps(actors_payload, allow_nan=False),
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Parquet schema (flat, stable across all scenes)
# ─────────────────────────────────────────────────────────────────────────────

SCENE_SCHEMA = pa.schema([
    pa.field("scene_id",              pa.string()),
    pa.field("segment_id",            pa.string()),
    pa.field("num_lanes",             pa.int32()),
    pa.field("num_segments",          pa.int32()),
    pa.field("target_chain_id",       pa.int32()),
    pa.field("valid",                 pa.bool_()),
    pa.field("reference_line_source", pa.string()),
    pa.field("reference_line_json",   pa.string()),
    pa.field("target_polygon_json",   pa.string()),
    pa.field("left_polygon_json",     pa.string()),
    pa.field("right_polygon_json",    pa.string()),
    pa.field("centerlines_json",      pa.string()),
    pa.field("tl_results_json",       pa.string()),
    pa.field("cw_results_json",       pa.string()),
    pa.field("actors_json",           pa.string()),
])


# ─────────────────────────────────────────────────────────────────────────────
# TFRecord reading
# ─────────────────────────────────────────────────────────────────────────────

def gcs_path() -> str:
    name = f"training_tfexample.tfrecord-{SHARD_INDEX:05d}-of-{TOTAL_SHARDS:05d}"
    return f"gs://{GCS_BUCKET}/{GCS_PREFIX}/{name}" if GCS_PREFIX else f"gs://{GCS_BUCKET}/{name}"


def parse_example(serialized):
    example = tf.io.parse_single_example(serialized, features_description)
    return {k: v.numpy() for k, v in example.items()}


def stream_tfrecord(path: str):
    for raw in tf.data.TFRecordDataset(path):
        yield parse_example(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ─────────────────────────────────────────────────────────────────────────────

def process_shard():
    path = gcs_path()
    print(f"[shard {SHARD_INDEX}] reading {path}", flush=True)

    s3 = boto3.client("s3")
    n_scenes = 0
    n_errors = 0

    for example in stream_tfrecord(path):
        try:
            scenario = Scenario(example)
            scenario.setup()
            result = process_scenario(scenario)
            if result is None:
                continue

            result = make_serializable(result)
            rows   = scene_to_parquet_rows(result)
            if not rows:
                continue

            # one Parquet file per scene
            scene_id = result["scene_id"]
            table    = pa.Table.from_pylist(rows, schema=SCENE_SCHEMA)
            buf      = io.BytesIO()
            pq.write_table(
                table, buf,
                compression="snappy",
                # keep row groups small — each file has only ~6 rows (one per segment)
                row_group_size=len(rows),
            )
            buf.seek(0)

            key = f"{S3_PREFIX}/scenes/{scene_id}.parquet"
            s3.upload_fileobj(buf, S3_BUCKET, key)
            n_scenes += 1

            if n_scenes % 10 == 0:
                print(f"[shard {SHARD_INDEX}] {n_scenes} scenes written", flush=True)

        except Exception as e:
            import traceback
            n_errors += 1
            print(f"[shard {SHARD_INDEX}] ERROR: {e}", flush=True)
            traceback.print_exc()
            continue

    print(f"[shard {SHARD_INDEX}] done — {n_scenes} scenes, {n_errors} errors", flush=True)


if __name__ == "__main__":
    process_shard()