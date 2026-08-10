from __future__ import annotations

from typing import Optional

import numpy as np
from shapely.geometry import mapping


def make_serializable(obj):
    """
    Recursively convert numpy / shapely / set types to plain Python so that
    json.dumps() and allow_nan=False both work cleanly.
    """
    if isinstance(obj, float) and (
        obj != obj or obj == float("inf") or obj == float("-inf")
    ):
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

    if hasattr(obj, "__geo_interface__"):
        return mapping(obj)

    if isinstance(obj, dict):
        return {
            (
                int(k)
                if isinstance(k, np.integer)
                else float(k)
                if isinstance(k, np.floating)
                else str(k)
                if not isinstance(k, (str, int, float, bool, type(None)))
                else k
            ): make_serializable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [make_serializable(i) for i in obj]

    return obj


def _encode_sparse_series(values: list) -> dict:
    """Convert a list with None gaps into compact sparse form."""
    intervals = []
    data = []

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

    return {
        "intervals": intervals,
        "data": data,
    }


def _encode_sparse_string_series(values: list) -> dict:
    """Sparse encoding for string series."""
    invalid = {None, "unknown", ""}

    intervals = []
    data = []

    i = 0
    n = len(values)

    while i < n:
        if values[i] in invalid:
            i += 1
            continue

        j = i
        while j < n and values[j] not in invalid:
            j += 1

        intervals.append([i, j - 1])
        data.extend(values[i:j])
        i = j

    return {
        "intervals": intervals,
        "data": data,
    }


def encode_inter_actor_pair(pair_data: dict) -> Optional[dict]:
    """
    Encode one actor-pair dict into compact sparse form.

    Returns None if the pair contains no valid data.
    """
    ttc = _encode_sparse_series(pair_data.get("ttc") or [])
    dist = _encode_sparse_series(pair_data.get("eucl_distance") or [])
    position = _encode_sparse_string_series(
        pair_data.get("position") or []
    )

    if not ttc["data"] and not dist["data"] and not position["data"]:
        return None

    return {
        "ttc": ttc,
        "eucl_distance": dist,
        "position": position,
    }
