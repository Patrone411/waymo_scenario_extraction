from __future__ import annotations

from typing import Optional

import pyarrow as pa


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
    arr = (feat_dict or {}).get((a, b))

    if arr is None:
        arr = (feat_dict or {}).get((b, a))

    if arr is None:
        return None

    try:
        v = float(arr[t])
        return None if (v != v or abs(v) == float("inf")) else v
    except (IndexError, TypeError):
        return None


def _safe_pair_str(feat_dict, a: str, b: str, t: int) -> Optional[str]:
    arr = (feat_dict or {}).get((a, b))

    if arr is None:
        arr = (feat_dict or {}).get((b, a))

    if arr is None:
        return None

    try:
        return str(arr[t])
    except (IndexError, TypeError):
        return Nonez

def _first_window(wbt0) -> Tuple[Optional[int], Optional[int]]:
    if not wbt0:
        return None, None
    for t0, ranges in sorted(wbt0.items()):
        if ranges:
            lo, hi = ranges[0]
            return int(t0), int(hi)
    return None, None