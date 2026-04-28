# results_writer.py
import pyarrow as pa
import pyarrow.parquet as pq
import json
from pathlib import Path
from typing import Optional
import io
import boto3

HITS_SCHEMA = pa.schema([
    pa.field("run_id",       pa.string()),
    pa.field("scenario",     pa.string()),
    pa.field("shard_index",  pa.int32()),
    pa.field("scene_id",     pa.string()),
    pa.field("segment_id",   pa.string()),
    pa.field("block_label",  pa.string()),
    pa.field("roles_json",   pa.string()),  # {"ego_vehicle": "vehicle_3", "npc": "vehicle_4"}
    pa.field("t0",           pa.int32()),
    pa.field("t1",           pa.int32()),
    pa.field("n_windows",    pa.int32()),
    pa.field("n_possible_windows", pa.int32()),
    pa.field("source_uri",   pa.string()),
])

# alle skalaren Feature-Werte für einen Akteur zu t0 oder t1
ACTOR_FRAMES_SCHEMA = pa.schema([
    pa.field("run_id",      pa.string()),
    pa.field("scenario",    pa.string()),
    pa.field("scene_id",    pa.string()),
    pa.field("segment_id",  pa.string()),
    pa.field("t0",          pa.int32()),
    pa.field("t1",          pa.int32()),
    pa.field("role",        pa.string()),   # "ego_vehicle" | "npc"
    pa.field("actor_id",    pa.string()),   # "vehicle_3"
    pa.field("frame",       pa.string()),   # "start" | "end"
    pa.field("t",           pa.int32()),    # konkreter Frame-Index
    # Cartesian
    pa.field("x",           pa.float64()),
    pa.field("y",           pa.float64()),
    pa.field("yaw",         pa.float64()),
    pa.field("speed",       pa.float64()),  # m/s
    pa.field("accel",       pa.float64()),  # m/s²
    # Frenet
    pa.field("s",           pa.float64()),
    pa.field("t_lat",       pa.float64()),  # laterale Position (t heißt hier t_lat)
    pa.field("s_dot",       pa.float64()),
    pa.field("t_dot",       pa.float64()),
    pa.field("yaw_delta",   pa.float64()),
    pa.field("osc_lane_id", pa.float64()),
])

# Interaktions-Features zwischen zwei Akteuren zu t0 oder t1
PAIR_FRAMES_SCHEMA = pa.schema([
    pa.field("run_id",        pa.string()),
    pa.field("scenario",      pa.string()),
    pa.field("scene_id",      pa.string()),
    pa.field("segment_id",    pa.string()),
    pa.field("t0",            pa.int32()),
    pa.field("t1",            pa.int32()),
    pa.field("role_a",        pa.string()),
    pa.field("role_b",        pa.string()),
    pa.field("actor_a",       pa.string()),
    pa.field("actor_b",       pa.string()),
    pa.field("frame",         pa.string()),   # "start" | "end"
    pa.field("t",             pa.int32()),
    pa.field("rel_distance",  pa.float64()),  # m
    pa.field("ttc",           pa.float64()),  # s
    pa.field("rel_position",  pa.string()),   # "front" | "back" | "unknown"
    pa.field("lat_rel",       pa.string()),   # "left" | "right" | "same"
])

class ResultsWriter:

    def __init__(self, run_id, scenario, shard_index,
                 bucket, prefix, endpoint_url=None, verify=None,
                 local_dir=None):
        self.run_id       = run_id
        self.scenario     = scenario
        self.shard_index  = shard_index
        self.bucket       = bucket
        self.prefix       = prefix
        self.endpoint_url = endpoint_url
        self.verify       = verify
        self.local_dir    = local_dir

        self._hits:         list = []
        self._actor_frames: list = []
        self._pair_frames:  list = []

    def add_hit(self, *, scene_id, segment_id, block_label,
                roles, t0, t1, n_windows, n_possible_windows,
                source_uri, feats):
        """
        Speichert einen Hit + alle Feature-Werte der beteiligten
        Akteure zu t0 und t1. feats ist das TagFeatures-Objekt.
        """
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

        # Feature-Werte für jeden beteiligten Akteur zu t0 und t1
        for frame_label, t in [("start", t0), ("end", t1)]:
            for role, actor_id in roles.items():
                self._actor_frames.append({
                    **base,
                    "role":       role,
                    "actor_id":   actor_id,
                    "frame":      frame_label,
                    "t":          int(t),
                    "x":          _safe_val(feats.x,         actor_id, t),
                    "y":          _safe_val(feats.y,         actor_id, t),
                    "yaw":        _safe_val(feats.yaw,       actor_id, t),
                    "speed":      _safe_val(feats.speed,     actor_id, t),
                    "accel":      _safe_val(feats.accel,     actor_id, t),
                    "s":          _safe_val(feats.s,         actor_id, t),
                    "t_lat":      _safe_val(feats.t,         actor_id, t),
                    "s_dot":      _safe_val(feats.s_dot,     actor_id, t),
                    "t_dot":      _safe_val(feats.t_dot,     actor_id, t),
                    "yaw_delta":  _safe_val(feats.yaw_delta, actor_id, t),
                    "osc_lane_id":_safe_val(feats.lane_idx,  actor_id, t),
                })

            # Interaktions-Features für alle Rollenpaare
            role_list = list(roles.items())
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
                        "rel_distance": _safe_pair(feats.rel_distance, actor_a, actor_b, t),
                        "ttc":          _safe_pair(feats.ttc,          actor_a, actor_b, t),
                        "rel_position": _safe_pair_str(feats.rel_position, actor_a, actor_b, t),
                        "lat_rel":      _safe_pair_str(feats.lat_rel,      actor_a, actor_b, t),
                    })

    def flush(self) -> dict: 
        written = {}
        for table_name, rows, schema in [
            ("match_hits",         self._hits,         HITS_SCHEMA),
            ("match_actor_frames", self._actor_frames, ACTOR_FRAMES_SCHEMA),
            ("match_pair_frames",  self._pair_frames,  PAIR_FRAMES_SCHEMA),
        ]:
            if not rows:
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

            print(f"[results] {len(rows)} rows -> {written[table_name]}", flush=True)

        return written

    def _s3(self):
        kwargs = {}
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        if self.verify:
            kwargs["verify"] = self.verify
        return boto3.client("s3", **kwargs)


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def _safe_val(feat_dict, actor_id, t) -> Optional[float]:
    arr = (feat_dict or {}).get(actor_id)
    if arr is None:
        return None
    try:
        v = float(arr[t])
        return None if (v != v or v == float("inf") or v == float("-inf")) else v
    except (IndexError, TypeError):
        return None


def _safe_pair(feat_dict, a, b, t) -> Optional[float]:
    arr = (feat_dict or {}).get((a, b)) or (feat_dict or {}).get((b, a))
    if arr is None:
        return None
    try:
        v = float(arr[t])
        return None if (v != v or v == float("inf") or v == float("-inf")) else v
    except (IndexError, TypeError):
        return None


def _safe_pair_str(feat_dict, a, b, t) -> Optional[str]:
    arr = (feat_dict or {}).get((a, b)) or (feat_dict or {}).get((b, a))
    if arr is None:
        return None
    try:
        return str(arr[t])
    except (IndexError, TypeError):
        return None