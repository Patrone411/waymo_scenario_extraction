#!/usr/bin/env python3
"""
worker.py

Reads one TFRecord shard (from s3 or local), runs process_scenario()
on every example, and writes one Parquet file per scene to S3 or local disk.

Environment variables
---------------------
LOCAL_MODE      "1" to bypass Azure and use local filesystem (default: "0")
LOCAL_INPUT     local directory containing TFRecord files   (default: "data")
LOCAL_OUTPUT    local directory for Parquet output          (default: "test_output")
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

from worker_utils import (
    make_serializable,
    encode_inter_actor_pair,
)

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_MODE   = os.environ.get("LOCAL_MODE",   "0") == "1"
LOCAL_INPUT  = os.environ.get("LOCAL_INPUT",  "data")
LOCAL_OUTPUT = os.environ.get("LOCAL_OUTPUT", "test_output")

SHARD_INDEX  = int(os.environ.get("SHARD_INDEX",  "0"))
TOTAL_SHARDS = int(os.environ.get("TOTAL_SHARDS", "1000"))



# Azure Config
AZURE_STORAGE_ACCOUNT  = os.environ.get("AZURE_STORAGE_ACCOUNT",  "")
AZURE_STORAGE_KEY      = os.environ.get("AZURE_STORAGE_KEY",      "")
AZURE_INPUT_CONTAINER  = os.environ.get("AZURE_INPUT_CONTAINER",  "tfrecords")
AZURE_OUTPUT_CONTAINER = os.environ.get("AZURE_OUTPUT_CONTAINER", "parquets")
AZURE_PREFIX           = os.environ.get("AZURE_PREFIX",           "parquet/run-001")

def get_blob_credential():
    """
    Authentication:
    1. AZURE_STORAGE_KEY gesetzt -> Storage Account Key
       (z.B. bestehende CI-Pipeline)
    2. Kein Storage Key -> Azure Workload Identity
       (z.B. AKS)
    """
    if AZURE_STORAGE_KEY:
        print(
            "[azure] authentication: Storage Account Key",
            flush=True,
        )
        return AZURE_STORAGE_KEY

    print(
        "[azure] authentication: Azure Workload Identity",
        flush=True,
    )

    return DefaultAzureCredential(
        exclude_interactive_browser_credential=True,
    )

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

def _tfrecord_name() -> str:
    return (
        f"training_tfexample.tfrecord"
        f"-{SHARD_INDEX:05d}"
        f"-of-{1000:05d}"
    )


def get_input_path() -> str:
    name = _tfrecord_name()

    if LOCAL_MODE:
        return os.path.join(LOCAL_INPUT, name)

    if not AZURE_STORAGE_ACCOUNT:
        raise ValueError(
            "AZURE_STORAGE_ACCOUNT muss gesetzt sein wenn LOCAL_MODE=0"
        )


    local_tmp = f"/tmp/shard_{SHARD_INDEX:05d}.tfrecord"

    """https://waymostorage.blob.core.windows.net/tfrecords/training_tfexample.tfrecord-00000-of-01000"""
    print(
        f"[shard {SHARD_INDEX}] lade Azure Blob "
        f"{AZURE_INPUT_CONTAINER}/{name}",
        flush=True,
    )

    print(AZURE_STORAGE_ACCOUNT)
    
    blob_service = BlobServiceClient(
        account_url=(
            f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net"
        ),
        credential=get_blob_credential(),
    )

    blob_client = blob_service.get_blob_client(
        container=AZURE_INPUT_CONTAINER,
        blob=name,
    )

    with open(local_tmp, "wb") as f:
        blob_client.download_blob().readinto(f)

    print(
        f"[shard {SHARD_INDEX}] download fertig → {local_tmp}",
        flush=True,
    )

    return local_tmp


def parse_example(serialized) -> dict:
    example = tf.io.parse_single_example(serialized, features_description)
    return {k: v.numpy() for k, v in example.items()}


def stream_tfrecord(path: str):
    """Yield parsed feature dicts from a TFRecord file."""
    for raw in tf.data.TFRecordDataset(path):
        yield parse_example(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Output: local or Azure Blob Storage
# ─────────────────────────────────────────────────────────────────────────────

def write_scene(scene_id: str, table: pa.Table, n_rows: int) -> None:
    if LOCAL_MODE:
        out_dir = (
            Path(LOCAL_OUTPUT)
            / f"{SHARD_INDEX:05d}"
            / "scenes"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{scene_id}.parquet"

        pq.write_table(
            table,
            out_path,
            compression="snappy",
            row_group_size=n_rows,
        )

        print(
            f"[local] {scene_id} → {out_path}",
            flush=True,
        )
        return

    if not AZURE_STORAGE_ACCOUNT:
        raise ValueError(
            "AZURE_STORAGE_ACCOUNT muss gesetzt sein wenn LOCAL_MODE=0"
        )


    buf = io.BytesIO()

    pq.write_table(
        table,
        buf,
        compression="snappy",
        row_group_size=n_rows,
    )

    buf.seek(0)

    blob_name = (
        f"{AZURE_PREFIX}/"
        f"{SHARD_INDEX:05d}/"
        f"scenes/"
        f"{scene_id}.parquet"
    )

    blob_service = BlobServiceClient(
        account_url=(
            f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net"
        ),
        credential=get_blob_credential(),
    )

    blob_service.get_blob_client(
        container=AZURE_OUTPUT_CONTAINER,
        blob=blob_name,
    ).upload_blob(
        buf,
        overwrite=True,
    )

    print(
        f"[azure] shard={SHARD_INDEX} "
        f"scene={scene_id} → "
        f"{AZURE_OUTPUT_CONTAINER}/{blob_name}",
        flush=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ─────────────────────────────────────────────────────────────────────────────

def process_shard() -> None:
    path = get_input_path()   # ← statt gcs_path()
    print(f"[shard {SHARD_INDEX}] verarbeite {path}", flush=True)

    n_scenes  = 0
    n_skipped = 0
    n_errors  = 0
    skipped_list = []

    for example in stream_tfrecord(path):
        try:
            scenario = Scenario(example)
            scenario.setup()
            result = process_scenario(scenario)

            if result is None:
                parsed = scenario.example
                scene_id = parsed['scenario/id'].item().decode("utf-8")
                skipped_list.append(scene_id)
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
        print("skipped scenes: ", skipped_list)


    print(
        f"[shard {SHARD_INDEX}] fertig — "
        f"{n_scenes} scenes, {n_skipped} übersprungen, {n_errors} fehler",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not LOCAL_MODE:
        if not AZURE_STORAGE_ACCOUNT:
            raise ValueError(
                "AZURE_STORAGE_ACCOUNT muss gesetzt sein "
                "wenn LOCAL_MODE=0"
            )

    print(
        f"[startup] LOCAL_MODE={LOCAL_MODE} "
        f"SHARD_INDEX={SHARD_INDEX} "
        f"TOTAL_SHARDS={TOTAL_SHARDS}",
        flush=True,
    )

    process_shard()