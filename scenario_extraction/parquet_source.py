# parquet_source.py
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterator, List, Optional

import boto3
import pandas as pd
import pyarrow.parquet as pq

from scenario_matching.feature_providers.features import TagFeatures, SkipSegment

def _safe_json(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return json.loads(val) if isinstance(val, str) else val


class SegmentMeta:
    def __init__(self, source_uri: str, seg_id: str, num_lanes: int, length_m: float):
        self.source_uri = source_uri
        self.seg_id     = seg_id
        self.num_lanes  = num_lanes
        self.length_m   = length_m


class SceneResult:
    def __init__(self, feats_by_seg: dict, seg_meta_by_id: dict):
        self.feats_by_seg   = feats_by_seg
        self.seg_meta_by_id = seg_meta_by_id


def _row_to_tag_json(df: pd.DataFrame) -> dict:
    """
    Reconstruct the original JSON dict shape that TagFeatures.from_tag_json()
    expects from all rows of a single scene Parquet file.

    Original shape:
    {
      "scene_id": "...",
      "road_segments":        { seg_id: { num_lanes, num_segments, ... } },
      "general_actor_data":   { "actor_activities": { actor_id: { x, y, yaw, long_v, lane_id, valid } } },
      "segment_actor_data":   { seg_id: { actor_id: { s, t, yaw_delta, s_dot, t_dot, osc_lane_id } } },
      "inter_actor_activities": { actor_a: { actor_b: { position, ttc, eucl_distance } } },
      "segment_env_elements": { seg_id: { tl_results, cw_results } },
      "processed_road_segments": { seg_id: { reference_line, target_polygon, ... } },
    }
    """
    scene_id = str(df["scene_id"].iloc[0])

    road_segments             = {}
    general_actor_activities  = {}   # actor_id -> arrays
    segment_actor_data        = {}   # seg_id -> actor_id -> arrays
    inter_actor_activities    = {}   # actor_a -> actor_b -> pair
    segment_env_elements      = {}
    processed_road_segments   = {}

    for _, row in df.iterrows():
        seg_id    = str(row["segment_id"])
        actors    = _safe_json(row["actors_json"]) or {}

        # ── road_segments ────────────────────────────────────────────────
        road_segments[seg_id] = {
            "num_lanes":    int(row.get("num_lanes")    or 0),
            "num_segments": int(row.get("num_segments") or 0),
        }

        # ── general_actor_data.actor_activities ──────────────────────────
        # merge across segments (same actor may appear in multiple segments
        # but carries the same global time series — first write wins)
        for actor_id, ts in (actors.get("actor_ts") or {}).items():
            if actor_id not in general_actor_activities:
                general_actor_activities[actor_id] = {
                    "x":       ts.get("x"),
                    "y":       ts.get("y"),
                    "yaw":     ts.get("yaw"),
                    "long_v":  ts.get("long_v"),
                    "lane_id": ts.get("lane_id"),
                    "valid":   [ts.get("valid_start"), ts.get("valid_end")],
                }

        # ── segment_actor_data ───────────────────────────────────────────
        segment_actor_data[seg_id] = {}
        for actor_id, seg_ts in (actors.get("seg_actor_ts") or {}).items():
            segment_actor_data[seg_id][actor_id] = {
                "s":           seg_ts.get("s"),
                "t":           seg_ts.get("t"),
                "yaw_delta":   seg_ts.get("yaw_delta"),
                "s_dot":       seg_ts.get("s_dot"),
                "t_dot":       seg_ts.get("t_dot"),
                "osc_lane_id": seg_ts.get("osc_lane_id"),
                "valid":       [seg_ts.get("valid_start"), seg_ts.get("valid_end")],
            }

        # ── inter_actor_activities (decode sparse → dense) ───────────────
        for key, pair in (actors.get("inter_actor") or {}).items():
            actor_a, actor_b = key.split("|", 1)
            if actor_a not in inter_actor_activities:
                inter_actor_activities[actor_a] = {}
            inter_actor_activities[actor_a][actor_b] = {
                "position":      _decode_sparse(pair.get("position") or {}, 91),
                "ttc":           _decode_sparse(pair.get("ttc")       or {}, 91),
                "eucl_distance": _decode_sparse(pair.get("eucl_distance") or {}, 91),
            }

        # ── segment_env_elements ─────────────────────────────────────────
        segment_env_elements[seg_id] = {
            "tl_results": _safe_json(row.get("tl_results_json")) or [],
            "cw_results": _safe_json(row.get("cw_results_json")) or [],
        }

        # ── processed_road_segments ──────────────────────────────────────
        processed_road_segments[seg_id] = {
            "target_chain_id":       int(row.get("target_chain_id") or 0),
            "valid":                 bool(row.get("valid", True)),
            "reference_line_source": str(row.get("reference_line_source") or ""),
            "reference_line":        _safe_json(row.get("reference_line_json")),
            "target_polygon":        _safe_json(row.get("target_polygon_json")),
            "left_polygon":          _safe_json(row.get("left_polygon_json")),
            "right_polygon":         _safe_json(row.get("right_polygon_json")),
            "centerline_by_chain":   _safe_json(row.get("centerlines_json")),
        }

    return {
        "scene_id":                scene_id,
        "road_segments":           road_segments,
        "general_actor_data":      {"actor_activities": general_actor_activities},
        "segment_actor_data":      segment_actor_data,
        "inter_actor_activities":  inter_actor_activities,
        "segment_env_elements":    segment_env_elements,
        "processed_road_segments": processed_road_segments,
    }


def _decode_sparse(sparse: dict, T: int = 91) -> list:
    """Reconstruct a full-length list from the sparse encoding worker.py wrote."""
    out = [None] * T
    intervals = sparse.get("intervals") or []
    data      = sparse.get("data")      or []
    pos = 0
    for (t0, t1) in intervals:
        for t in range(t0, t1 + 1):
            if pos < len(data):
                out[t] = data[pos]
                pos += 1
    return out


class ParquetSource:
    """
    Drop-in replacement for S3PickleSource.
    Reads scene Parquet files, reconstructs the original JSON dict shape,
    and builds TagFeatures via TagFeatures.from_tag_json() — exactly as
    the old pickle pipeline did.
    """

    def __init__(
        self,
        bucket: str,
        base_prefix: str,
        endpoint_url: Optional[str] = None,
        verify: Optional[str]       = None,
        min_lanes: Optional[int]    = None,
    ):
        self.bucket      = bucket
        self.base_prefix = base_prefix.rstrip("/")
        self.endpoint_url = endpoint_url
        self.verify       = verify
        self.min_lanes    = min_lanes

    def _s3_client(self):
        kwargs = {}
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        if self.verify and self.verify not in ("true", "1", "yes"):
            kwargs["verify"] = self.verify
        return boto3.client("s3", **kwargs)

    def _list_scene_keys(self, s3) -> List[str]:
        prefix = f"{self.base_prefix}/scenes/"
        paginator = s3.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                if obj["Key"].endswith(".parquet"):
                    keys.append(obj["Key"])
        return sorted(keys)

    def __iter__(self) -> Iterator[SceneResult]:
        s3   = self._s3_client()
        keys = self._list_scene_keys(s3)
        for key in keys:
            try:
                buf = io.BytesIO()
                s3.download_fileobj(self.bucket, key, buf)
                buf.seek(0)
                df = pq.read_table(buf).to_pandas()
            except Exception as e:
                print(f"[WARN] could not read {key}: {e}", flush=True)
                continue

            yield from _iter_scene(df, source_uri=f"s3://{self.bucket}/{key}",
                                   min_lanes=self.min_lanes)


class LocalParquetSource:
    """Same interface as ParquetSource but reads from a local directory."""

    def __init__(self, scenes_dir: str, min_lanes: Optional[int] = None):
        self.scenes_dir = Path(scenes_dir)
        self.min_lanes  = min_lanes

    def __iter__(self) -> Iterator[SceneResult]:
        for path in sorted(self.scenes_dir.glob("*.parquet")):
            try:
                df = pq.read_table(path).to_pandas()
            except Exception as e:
                print(f"[WARN] could not read {path}: {e}", flush=True)
                continue
            yield from _iter_scene(df, source_uri=str(path),
                                   min_lanes=self.min_lanes)


def _iter_scene(
    df: pd.DataFrame,
    source_uri: str,
    min_lanes: Optional[int],
) -> Iterator[SceneResult]:
    """
    Shared logic: reconstruct tag_json dict → TagFeatures per segment,
    then yield one SceneResult.
    """
    tag_json = _row_to_tag_json(df)

    feats_by_seg: dict = {}
    meta_by_id:   dict = {}

    for seg_id, seg_meta in (tag_json.get("road_segments") or {}).items():
        num_lanes = int(seg_meta.get("num_lanes") or 0)
        if min_lanes is not None and num_lanes < min_lanes:
            continue
        try:
            feats = TagFeatures.from_tag_json(tag_json, seg_id)
        except SkipSegment:
            continue
        except Exception as e:
            print(f"[WARN] TagFeatures.from_tag_json failed for {seg_id}: {e}", flush=True)
            continue

        meta = SegmentMeta(
            source_uri=source_uri,
            seg_id=seg_id,
            num_lanes=num_lanes,
            length_m=feats.length_m,
        )
        feats_by_seg[seg_id] = feats
        meta_by_id[seg_id]   = meta

    if feats_by_seg:
        yield SceneResult(feats_by_seg=feats_by_seg, seg_meta_by_id=meta_by_id)