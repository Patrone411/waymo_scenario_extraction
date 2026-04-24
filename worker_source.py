# worker_source.py  (or inline in run_matching.py, replacing S3PickleSource)

import io
import json
from pathlib import Path
from typing import Iterator, List, Optional

import boto3
import pyarrow.parquet as pq
import pandas as pd


def _safe_json(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return json.loads(val) if isinstance(val, str) else val


def _decode_sparse(sparse: dict, length: int = 91) -> list:
    """Reconstruct a full-length list from the sparse encoding worker.py wrote."""
    out = [None] * length
    intervals = sparse.get("intervals") or []
    data      = sparse.get("data")      or []
    pos = 0
    for (t0, t1) in intervals:
        for t in range(t0, t1 + 1):
            if pos < len(data):
                out[t] = data[pos]
                pos += 1
    return out


class SegmentFeatures:
    """
    Reconstructed from one Parquet row (one segment in one scene).
    Attribute names here must match what MatchEngine reads off feats objects.
    If the engine uses different names, rename the self.X assignments below.
    """
    def __init__(self, row: dict):
        actors = _safe_json(row["actors_json"]) or {}

        self.seg_id    = row["segment_id"]
        self.num_lanes = int(row["num_lanes"] or 0)
        self.num_segments = int(row.get("num_segments") or 0)
        self.valid     = bool(row.get("valid", True))
        self.T         = 91   # fixed timestep count

        # actor membership for this segment
        self.actor_ids: list = actors.get("actor_ids") or []

        # global Cartesian time series  {actor_id: {x, y, yaw, long_v, lane_id, valid_start, valid_end}}
        self.actor_ts: dict = actors.get("actor_ts") or {}

        # Frenet / segment-relative time series  {actor_id: {s, t, yaw_delta, s_dot, t_dot, osc_lane_id}}
        self.seg_actor_ts: dict = actors.get("seg_actor_ts") or {}

        # sparse inter-actor  {"actorA|actorB": {ttc: sparse, eucl_distance: sparse, position: sparse}}
        # keep as-is; let the engine call get_inter_actor() below, or decode lazily
        self._inter_actor_sparse: dict = actors.get("inter_actor") or {}

        # geometry
        self.reference_line = _safe_json(row.get("reference_line_json"))
        self.target_polygon = _safe_json(row.get("target_polygon_json"))
        self.left_polygon   = _safe_json(row.get("left_polygon_json"))
        self.right_polygon  = _safe_json(row.get("right_polygon_json"))
        self.centerlines    = _safe_json(row.get("centerlines_json"))
        self.reference_line_source = row.get("reference_line_source") or ""

        # env
        self.tl_results = _safe_json(row.get("tl_results_json")) or []
        self.cw_results = _safe_json(row.get("cw_results_json")) or []

        # derived scalar used in segment_stats_mode
        coords = (self.reference_line or {}).get("coordinates") or []
        self.length_m = float(len(coords))

    def get_inter_actor(self, actor_a: str, actor_b: str, field: str) -> list:
        """
        Decode one sparse series on demand.
        field: 'ttc' | 'eucl_distance' | 'position'
        Returns a full-length list[91] with None gaps restored.
        """
        key    = f"{actor_a}|{actor_b}"
        sparse = (self._inter_actor_sparse.get(key) or {}).get(field)
        if sparse is None:
            return [None] * self.T
        return _decode_sparse(sparse, self.T)

    def get_inter_actor_dense(self) -> dict:
        """
        Fully decode all inter-actor pairs back to the original nested dict shape:
        { actor_a: { actor_b: { "ttc": [...91...], "eucl_distance": [...], "position": [...] } } }
        Only call this if the engine needs the full matrix at once.
        """
        out: dict = {}
        for key, pair in self._inter_actor_sparse.items():
            actor_a, actor_b = key.split("|", 1)
            out.setdefault(actor_a, {})[actor_b] = {
                "ttc":           _decode_sparse(pair.get("ttc")           or {}, self.T),
                "eucl_distance": _decode_sparse(pair.get("eucl_distance") or {}, self.T),
                "position":      _decode_sparse(pair.get("position")      or {}, self.T),
            }
        return out


class SegmentMeta:
    def __init__(self, source_uri: str, seg_id: str, num_lanes: int, length_m: float):
        self.source_uri = source_uri
        self.seg_id     = seg_id
        self.num_lanes  = num_lanes
        self.length_m   = length_m


class SceneResult:
    def __init__(self, feats_by_seg: dict, seg_meta_by_id: dict):
        self.feats_by_seg    = feats_by_seg
        self.seg_meta_by_id  = seg_meta_by_id


class ParquetSource:
    """
    Drop-in replacement for S3PickleSource.
    Yields one SceneResult per scene Parquet file under
    s3://<bucket>/<prefix>/scenes/*.parquet
    """

    def __init__(
        self,
        bucket: str,
        base_prefix: str,
        endpoint_url: Optional[str] = None,
        verify: Optional[str]       = None,
        min_lanes: Optional[int]    = None,
    ):
        self.bucket       = bucket
        self.base_prefix  = base_prefix.rstrip("/")
        self.endpoint_url = endpoint_url
        self.verify       = verify
        self.min_lanes    = min_lanes

    def _s3_client(self):
        kwargs = {}
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        if self.verify and self.verify not in ("true", "1", "yes"):
            import botocore.config
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

            feats_by_seg: dict = {}
            meta_by_id:   dict = {}

            for _, row in df.iterrows():
                num_lanes = int(row.get("num_lanes") or 0)
                if self.min_lanes is not None and num_lanes < self.min_lanes:
                    continue

                feats  = SegmentFeatures(row.to_dict())
                seg_id = feats.seg_id
                meta   = SegmentMeta(
                    source_uri=f"s3://{self.bucket}/{key}",
                    seg_id=seg_id,
                    num_lanes=num_lanes,
                    length_m=feats.length_m,
                )
                feats_by_seg[seg_id] = feats
                meta_by_id[seg_id]   = meta

            if feats_by_seg:
                yield SceneResult(feats_by_seg=feats_by_seg, seg_meta_by_id=meta_by_id)