#!/usr/bin/env python3
"""
worker.py

Reads one TFRecord shard (from GCS or local), runs process_scenario()
on every example, and writes one Parquet file per scene to S3 or local disk.

Environment variables
---------------------
LOCAL_MODE      "1" to bypass GCS/S3 and use local filesystem (default: "0")
LOCAL_INPUT     local directory containing TFRecord files   (default: "data")
LOCAL_OUTPUT    local directory for Parquet output          (default: "test_output")

GCS_BUCKET      GCS bucket name                (required when LOCAL_MODE=0)
GCS_PREFIX      prefix inside the bucket       (optional, default: "")
SHARD_INDEX     which shard this pod processes (required)
TOTAL_SHARDS    total number of shards         (default: 1000)

S3_BUCKET       S3 bucket for output           (required when LOCAL_MODE=0)
S3_PREFIX       S3 prefix for output           (default: "output")
S3_ENDPOINT_URL custom S3 endpoint URL         (optional)
AWS_CA_BUNDLE   path to CA bundle for TLS      (optional)
"""

from __future__ import annotations

import io
import json
import os
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import mapping
import tensorflow as tf

from feature_extraction.pipeline import process_scenario
from feature_extraction.tools.scenario import Scenario, features_description


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_MODE   = os.environ.get("LOCAL_MODE",   "0") == "1"
LOCAL_INPUT  = os.environ.get("LOCAL_INPUT",  "data")
LOCAL_OUTPUT = os.environ.get("LOCAL_OUTPUT", "test_output")

GCS_BUCKET   = os.environ.get("GCS_BUCKET",  "")
GCS_PREFIX   = os.environ.get("GCS_PREFIX",  "")
SHARD_INDEX  = int(os.environ.get("SHARD_INDEX",  "0"))
TOTAL_SHARDS = int(os.environ.get("TOTAL_SHARDS", "1000"))

S3_BUCKET       = os.environ.get("S3_BUCKET",       "")
S3_PREFIX       = os.environ.get("S3_PREFIX",       "output")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", None)
AWS_CA_BUNDLE   = os.environ.get("AWS_CA_BUNDLE",   None)


# ─────────────────────────────────────────────────────────────────────────────
# Parquet schema  (flat, stable across all scenes)
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
# Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_serializable(obj):
    """
    Recursively convert numpy / shapely / set types to plain Python so that
    json.dumps() and allow_nan=False both work cleanly.
    """
    # plain Python float: NaN / Inf → None
    if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
        return None
    # numpy floating scalar
    if isinstance(obj, np.floating):
        if obj != obj or obj == np.inf or obj == -np.inf:
            return None
        return float(obj)
    # numpy integer scalar
    if isinstance(obj, np.integer):
        return int(obj)
    # numpy array — recurse element-by-element so NaN/Inf pass through above
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return make_serializable(obj.item())
        return [make_serializable(x) for x in obj]
    # Python set
    if isinstance(obj, set):
        return list(obj)
    # Shapely geometry (Point, Polygon, LineString, …)
    if hasattr(obj, "__geo_interface__"):
        return mapping(obj)
    # dict — also convert non-string keys
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
    Convert a length-91 list (with None gaps) into a compact sparse form.

    Returns:
        {
            "intervals": [[t_start, t_end], ...],   # inclusive ranges
            "data":      [v0, v1, ...]               # values concatenated
        }

    A completely empty list (all None) returns {"intervals": [], "data": []}.
    """
    intervals: list = []
    data:      list = []
    i = 0
    n = len(values)
    while i < n:
        if values[i] is None:
            i += 1
            continue
        j = i
        while j < n and values[j] is not None:
            j += 1
        intervals.append([i, j - 1])
        data.extend(values[i:j])
        i = j
    return {"intervals": intervals, "data": data}


def _encode_sparse_string_series(values: list) -> dict:
    """
    Same as _encode_sparse_series but for string series (e.g. position labels).
    Treats None, "", and "unknown" as invalid / absent.
    """
    INVALID = {None, "unknown", ""}
    intervals: list = []
    data:      list = []
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


def encode_inter_actor_pair(pair_data: dict) -> Optional[dict]:
    """
    Encode one actor-pair dict into compact sparse form.
    Returns None if the pair carries no valid data at all (→ drop entirely).

    Input shape:
        {
            "position":      [...91 str/None...],
            "ttc":           [...91 float/None...],
            "eucl_distance": [...91 float/None...]
        }
    """
    ttc      = _encode_sparse_series(pair_data.get("ttc")           or [])
    dist     = _encode_sparse_series(pair_data.get("eucl_distance") or [])
    position = _encode_sparse_string_series(pair_data.get("position") or [])

    if not ttc["data"] and not dist["data"] and not position["data"]:
        return None

    return {"ttc": ttc, "eucl_distance": dist, "position": position}


# ─────────────────────────────────────────────────────────────────────────────
# Scene → flat Parquet rows
# ─────────────────────────────────────────────────────────────────────────────

def scene_to_parquet_rows(result: dict) -> list:
    """
    Convert one process_scenario() result dict into a list of row dicts,
    one row per road segment, ready for pa.Table.from_pylist().

    Heavy nested data (geometry, actor arrays, sparse inter-actor) is
    serialised to JSON strings so the Parquet schema remains flat and stable
    across scenes with varying numbers of actors / segments.
    """
    scene_id = result["scene_id"]
    rows: list = []

    for seg_id, seg_proc in (result.get("processed_road_segments") or {}).items():

        # ── geometry (serialised to JSON strings) ────────────────────────────
        ref_line    = seg_proc.get("reference_line")
        tgt_poly    = seg_proc.get("target_polygon")
        left_poly   = seg_proc.get("left_polygon")
        right_poly  = seg_proc.get("right_polygon")
        centerlines = seg_proc.get("centerline_by_chain")

        # ── env elements ─────────────────────────────────────────────────────
        seg_env = (result.get("segment_env_elements") or {}).get(seg_id) or {}

        # ── road segment metadata ─────────────────────────────────────────────
        road_seg  = (result.get("road_segments") or {}).get(seg_id) or {}
        num_lanes = int(road_seg.get("num_lanes", 0) or 0)

        # ── actor membership for this segment ─────────────────────────────────
        gad           = result.get("general_actor_data") or {}
        seg_actor_ids = list(
            ((gad.get("per_segment_ids") or {}).get(seg_id)) or []
        )

        # global Cartesian time series (only actors present in this segment)
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

        # Frenet / segment-relative time series
        seg_actor_ts: dict = {}
        for actor_id, data in (
            (result.get("segment_actor_data") or {}).get(seg_id) or {}
        ).items():
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

        # sparse inter-actor pairs (only actors in this segment, only non-empty)
        inter_actor: dict = {}
        inter_raw = result.get("inter_actor_activities") or {}
        for actor_a in seg_actor_ids:
            pairs_for_a = inter_raw.get(actor_a) or {}
            for actor_b in seg_actor_ids:
                if actor_a == actor_b:
                    continue
                pair_data = pairs_for_a.get(actor_b)
                if pair_data is None:
                    continue
                encoded = encode_inter_actor_pair(pair_data)
                if encoded is not None:
                    inter_actor[f"{actor_a}|{actor_b}"] = encoded

        # ── actors blob ───────────────────────────────────────────────────────
        actors_payload = {
            "actor_ids":    seg_actor_ids,
            "actor_ts":     actor_ts,
            "seg_actor_ts": seg_actor_ts,
            "inter_actor":  inter_actor,
        }

        rows.append({
            "scene_id":              scene_id,
            "segment_id":            seg_id,
            "num_lanes":             num_lanes,
            "num_segments":          int(road_seg.get("num_segments", 0) or 0),
            "target_chain_id":       int(seg_proc.get("target_chain_id", 0) or 0),
            "valid":                 bool(seg_proc.get("valid", True)),
            "reference_line_source": str(seg_proc.get("reference_line_source") or ""),
            "reference_line_json":   json.dumps(ref_line,    allow_nan=False),
            "target_polygon_json":   json.dumps(tgt_poly,    allow_nan=False),
            "left_polygon_json":     json.dumps(left_poly,   allow_nan=False),
            "right_polygon_json":    json.dumps(right_poly,  allow_nan=False),
            "centerlines_json":      json.dumps(centerlines, allow_nan=False),
            "tl_results_json":       json.dumps(seg_env.get("tl_results", []), allow_nan=False),
            "cw_results_json":       json.dumps(seg_env.get("cw_results", []), allow_nan=False),
            "actors_json":           json.dumps(actors_payload, allow_nan=False),
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# TFRecord I/O
# ─────────────────────────────────────────────────────────────────────────────

def gcs_path() -> str:
    """Return the path to this shard's TFRecord file (local or GCS URI)."""
    name = (
        f"training_tfexample.tfrecord"
        f"-{SHARD_INDEX:05d}"
        f"-of-{TOTAL_SHARDS:05d}"
    )
    if LOCAL_MODE:
        return os.path.join(LOCAL_INPUT, name)
    return (
        f"gs://{GCS_BUCKET}/{GCS_PREFIX}/{name}"
        if GCS_PREFIX
        else f"gs://{GCS_BUCKET}/{name}"
    )


def parse_example(serialized) -> dict:
    example = tf.io.parse_single_example(serialized, features_description)
    return {k: v.numpy() for k, v in example.items()}


def stream_tfrecord(path: str):
    """Yield parsed feature dicts from a TFRecord file."""
    for raw in tf.data.TFRecordDataset(path):
        yield parse_example(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Output: local or S3
# ─────────────────────────────────────────────────────────────────────────────

def _s3_client():
    import boto3
    kwargs: dict = {}
    if S3_ENDPOINT_URL:
        kwargs["endpoint_url"] = S3_ENDPOINT_URL
    if AWS_CA_BUNDLE:
        kwargs["verify"] = AWS_CA_BUNDLE
    return boto3.client("s3", **kwargs)


def write_scene(scene_id: str, table: pa.Table, n_rows: int) -> None:
    """Write one scene Parquet file to local disk or S3."""
    if LOCAL_MODE:
        out_dir = Path(LOCAL_OUTPUT) / f"{SHARD_INDEX:05d}" / "scenes"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{scene_id}.parquet"
        pq.write_table(
            table,
            out_path,
            compression="snappy",
            row_group_size=n_rows,
        )
        print(f"[local] shard={SHARD_INDEX} scene={scene_id} → {out_path}", flush=True)
    else:
        buf = io.BytesIO()
        pq.write_table(
            table,
            buf,
            compression="snappy",
            row_group_size=n_rows,
        )
        buf.seek(0)
        key = f"{S3_PREFIX}/{SHARD_INDEX:05d}/scenes/{scene_id}.parquet"
        _s3_client().upload_fileobj(buf, S3_BUCKET, key)
        print(f"[s3] shard={SHARD_INDEX} scene={scene_id} → s3://{S3_BUCKET}/{key}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ─────────────────────────────────────────────────────────────────────────────

def process_shard() -> None:
    path = gcs_path()
    print(f"[shard {SHARD_INDEX}] lese {path}", flush=True)

    n_scenes  = 0
    n_skipped = 0
    n_errors  = 0

    for example in stream_tfrecord(path):
        try:
            scenario = Scenario(example)
            scenario.setup()
            result = process_scenario(scenario)

            if result is None:
                n_skipped += 1
                continue

            # make everything JSON-safe before building rows
            result = make_serializable(result)

            rows = scene_to_parquet_rows(result)
            if not rows:
                n_skipped += 1
                continue

            scene_id = result["scene_id"]
            table    = pa.Table.from_pylist(rows, schema=SCENE_SCHEMA)
            write_scene(scene_id, table, n_rows=len(rows))
            n_scenes += 1

            if n_scenes % 10 == 0:
                print(
                    f"[shard {SHARD_INDEX}] {n_scenes} scenes geschrieben "
                    f"({n_skipped} übersprungen, {n_errors} fehler)",
                    flush=True,
                )

        except Exception:
            n_errors += 1
            print(f"[shard {SHARD_INDEX}] ERROR bei example:", flush=True)
            traceback.print_exc()
            continue

    print(
        f"[shard {SHARD_INDEX}] fertig — "
        f"{n_scenes} scenes, {n_skipped} übersprungen, {n_errors} fehler",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Sanity-checks
    if not LOCAL_MODE:
        if not GCS_BUCKET:
            raise ValueError("GCS_BUCKET muss gesetzt sein wenn LOCAL_MODE=0")
        if not S3_BUCKET:
            raise ValueError("S3_BUCKET muss gesetzt sein wenn LOCAL_MODE=0")

    print(
        f"[startup] LOCAL_MODE={LOCAL_MODE} "
        f"SHARD_INDEX={SHARD_INDEX} "
        f"TOTAL_SHARDS={TOTAL_SHARDS}",
        flush=True,
    )

    process_shard()