# generic_window_extractor.py
"""
Automatically derives parameter extraction specs from OSC2 lowered calls,
then extracts the actual values from a matched window's TagFeatures.

Replaces hand-written stats_extractors_change_lane.py / stats_extractors_cross.py etc.
One extractor works for any OSC2 scenario without writing scenario-specific code.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Unit conversion
# ─────────────────────────────────────────────────────────────────────────────

_UNIT_TO_SI = {
    # speed
    "kilometer_per_hour": 1.0 / 3.6,
    "kph":                1.0 / 3.6,
    "mph":                0.44704,
    "meter_per_second":   1.0,
    "m/s":                1.0,
    # distance
    "meter":              1.0,
    "m":                  1.0,
    "kilometer":          1000.0,
    "km":                 1000.0,
    # time
    "second":             1.0,
    "s":                  1.0,
    # angle
    "degree":             math.pi / 180.0,
    "rad":                1.0,
}

def _to_si(value: float, unit: Optional[str]) -> float:
    if unit is None:
        return value
    factor = _UNIT_TO_SI.get(unit.lower().strip(), 1.0)
    return value * factor


# ─────────────────────────────────────────────────────────────────────────────
# Derive PARAM_SPECS from OSC2 calls
# ─────────────────────────────────────────────────────────────────────────────

def build_param_specs_from_calls(calls: list) -> Dict[str, dict]:
    """
    Walk the lowered OSC2 calls and produce a PARAM_SPECS dict in the same
    format the existing StatsCollector expects:

        {
            "param_name": {
                "range": [lo, hi],   # in SI units
                "n_bins": 20,
                "unit": "m/s",       # display unit
            },
            ...
        }

    Handles: speed, position (distance), lane, ttc, accel.
    """
    specs: Dict[str, dict] = {}

    for call in calls:
        actor    = call.get("actor", "unknown")
        for mod in (call.get("modifiers") or []):
            name = mod.get("name", "")
            args = mod.get("args") or {}
            at   = args.get("at", "")          # "start" | "end" | ""

            suffix = f"_{at}" if at else ""

            # ── speed ────────────────────────────────────────────────────────
            if name == "speed":
                speed_arg = args.get("speed") or {}
                rng  = speed_arg.get("range")
                unit = speed_arg.get("unit", "meter_per_second")
                if rng and len(rng) == 2:
                    key = f"{actor}_speed{suffix}"
                    specs[key] = {
                        "range":  [_to_si(rng[0], unit), _to_si(rng[1], unit)],
                        "n_bins": 20,
                        "unit":   "m/s",
                        # extraction metadata
                        "_actor":   actor,
                        "_field":   "speed",
                        "_at":      at or "window",
                        "_mod":     name,
                    }

            # ── position (distance) ──────────────────────────────────────────
            elif name == "position":
                dist_arg = args.get("distance") or {}
                rng      = dist_arg.get("range")
                unit     = dist_arg.get("unit", "meter")
                ref_actor = args.get("ahead_of") or args.get("behind") or ""
                if rng and len(rng) == 2:
                    key = f"{actor}_distance_to_{ref_actor}{suffix}"
                    specs[key] = {
                        "range":  [_to_si(rng[0], unit), _to_si(rng[1], unit)],
                        "n_bins": 20,
                        "unit":   "m",
                        "_actor":     actor,
                        "_ref_actor": ref_actor,
                        "_field":     "rel_distance",
                        "_at":        at or "window",
                        "_mod":       name,
                    }

            # ── lane ─────────────────────────────────────────────────────────
            elif name == "lane":
                lane_val = args.get("lane")
                if lane_val is not None and not args.get("same_as"):
                    key = f"{actor}_lane{suffix}"
                    specs[key] = {
                        "range":  [float(lane_val) - 0.5, float(lane_val) + 0.5],
                        "n_bins": max(1, abs(int(lane_val)) * 2 + 1),
                        "unit":   "osc_lane_id",
                        "_actor": actor,
                        "_field": "lane_idx",
                        "_at":    at or "window",
                        "_mod":   name,
                    }

            # ── ttc (if explicitly named as modifier) ────────────────────────
            elif name == "ttc":
                rng  = (args.get("ttc") or {}).get("range")
                unit = (args.get("ttc") or {}).get("unit", "second")
                ref_actor = args.get("to") or ""
                if rng and len(rng) == 2:
                    key = f"{actor}_ttc_to_{ref_actor}{suffix}"
                    specs[key] = {
                        "range":  [_to_si(rng[0], unit), _to_si(rng[1], unit)],
                        "n_bins": 20,
                        "unit":   "s",
                        "_actor":     actor,
                        "_ref_actor": ref_actor,
                        "_field":     "ttc",
                        "_at":        at or "window",
                        "_mod":       name,
                    }

    return specs


# ─────────────────────────────────────────────────────────────────────────────
# Generic window extractor
# ─────────────────────────────────────────────────────────────────────────────

def _get_frame(at: str, t0: int, t1: int, T: int) -> int:
    """Resolve 'start' | 'end' | 'window' to a concrete frame index."""
    if at == "start":
        return t0
    if at == "end":
        return t1
    # "window" → use midpoint
    return (t0 + t1) // 2


def _scalar_at(arr, frame: int):
    """
    Safely extract a scalar from a numpy array or plain list at `frame`.
    Returns None if out of bounds or not finite.
    """
    if arr is None:
        return None
    try:
        v = arr[frame]
        if isinstance(v, (int, float)) and not math.isfinite(float(v)):
            return None
        return float(v)
    except (IndexError, TypeError):
        return None


def extract_params_for_window(
    feats,          # TagFeatures instance
    roles: dict,    # {"ego_vehicle": "vehicle_3", "npc": "vehicle_4"}
    t0: int,
    t1: int,
    param_specs: Optional[Dict[str, dict]] = None,
    left_is_decreasing: bool = True,
) -> Dict[str, Any]:
    """
    Generic parameter extractor.  Works for any OSC2 scenario as long as
    param_specs was built by build_param_specs_from_calls().

    Falls back gracefully: if a field is missing for a role, the key is
    omitted from the result (StatsCollector skips None values).

    Parameters
    ----------
    feats           TagFeatures for this segment
    roles           role-name → actor-id mapping from the binding
    t0, t1          matched window start/end frame (inclusive)
    param_specs     output of build_param_specs_from_calls()
    left_is_decreasing  OSC2 lane sign convention

    Returns
    -------
    dict of { param_name: float_value }
    """
    if not param_specs:
        return {}

    T      = getattr(feats, "T", 91)
    result: Dict[str, Any] = {}

    for param_name, spec in param_specs.items():
        actor_role = spec["_actor"]
        field      = spec["_field"]
        at         = spec["_at"]
        ref_role   = spec.get("_ref_actor", "")

        # resolve role → actual actor id
        actor_id = roles.get(actor_role)
        if actor_id is None:
            continue

        frame = _get_frame(at, t0, t1, T)

        # ── scalar per-actor fields ──────────────────────────────────────────
        if field in ("speed", "lane_idx", "accel", "s", "t", "s_dot", "t_dot",
                     "yaw", "yaw_delta"):
            arr = getattr(feats, field, {}).get(actor_id)
            val = _scalar_at(arr, frame)
            if val is not None:
                result[param_name] = val

        # ── pairwise fields: rel_distance, ttc, rel_position ─────────────────
        elif field in ("rel_distance", "ttc", "rel_position"):
            ref_id = roles.get(ref_role)
            if ref_id is None:
                continue
            pair_dict = getattr(feats, field, {})
            # try both key orderings
            arr = pair_dict.get((actor_id, ref_id)) or pair_dict.get((ref_id, actor_id))
            if arr is None:
                continue
            if field == "rel_position":
                # string array — just record as-is
                try:
                    result[param_name] = str(arr[frame])
                except (IndexError, TypeError):
                    pass
            else:
                val = _scalar_at(arr, frame)
                if val is not None:
                    result[param_name] = val

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build everything from calls in one call
# ─────────────────────────────────────────────────────────────────────────────

def make_generic_extractor(calls: list, left_is_decreasing: bool = True):
    """
    Returns (param_specs, extractor_fn) ready to pass into StatsCollector.

    Usage in run_matching.py:
        from generic_window_extractor import make_generic_extractor
        param_specs, extractor = make_generic_extractor(calls)
    """
    param_specs = build_param_specs_from_calls(calls)

    def extractor(feats, roles, t0, t1):
        return extract_params_for_window(
            feats, roles, t0, t1,
            param_specs=param_specs,
            left_is_decreasing=left_is_decreasing,
        )

    return param_specs, extractor