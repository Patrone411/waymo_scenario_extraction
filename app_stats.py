import numpy as np
import pandas as pd


# =========================================================
# Shared TTC
# =========================================================

def ttc(delta_s, delta_v):
    if delta_v > 0:
        val = delta_s / delta_v
        return val if np.isfinite(val) else np.nan
    return np.nan


# =========================================================
# CPNA
# =========================================================

def cpna_stats(actor_frames, pair_frames):

    af = actor_frames[actor_frames["scenario"] == "cpna.osc"].copy()
    pf = pair_frames[pair_frames["scenario"] == "cpna.osc"].copy()

    group_cols = ["scene_id", "segment_id", "t0", "t1"]

    min_dist_values = []
    ttc_values = []

    for _, group in pf.groupby(group_cols):

        if group.empty:
            continue

        idx_min = group["rel_distance"].idxmin()
        min_row = group.loc[idx_min]

        min_dist_values.append(min_row["rel_distance"])
        t_min = min_row["t"]

        ego = af[
            (af["scene_id"] == min_row["scene_id"]) &
            (af["segment_id"] == min_row["segment_id"]) &
            (af["t"] == t_min) &
            (af["role"] == "ego_vehicle")
        ]

        npc = af[
            (af["scene_id"] == min_row["scene_id"]) &
            (af["segment_id"] == min_row["segment_id"]) &
            (af["t"] == t_min) &
            (af["role"] == "npc")
        ]

        if ego.empty or npc.empty:
            continue

        delta_s = npc.iloc[0]["s"] - ego.iloc[0]["s"]
        delta_v = ego.iloc[0]["s_dot"] - npc.iloc[0]["s_dot"]

        val = ttc(delta_s, delta_v)
        if np.isfinite(val):
            ttc_values.append(val)

    return np.array(min_dist_values), np.array(ttc_values)


# =========================================================
# CCRB
# =========================================================

def ccrb_stats(actor_frames):
    af = actor_frames[actor_frames["scenario"] == "ccrb.osc"].copy()
    group_cols = ["scene_id", "segment_id", "t0", "t1"]

    ttc_t1_values = []
    ego_avg_speeds = []
    npc_avg_speeds = []

    for _, group in af.groupby(group_cols):

        ego = group[group["role"] == "ego_vehicle"]
        npc = group[group["role"] == "npc"]

        if ego.empty or npc.empty:
            continue

        # -------------------------
        # TTC at t1
        # -------------------------
        t1 = group["t1"].iloc[0]

        ego_t1 = ego[ego["t"] == t1]
        npc_t1 = npc[npc["t"] == t1]

        if not ego_t1.empty and not npc_t1.empty:
            e = ego_t1.iloc[0]
            n = npc_t1.iloc[0]

            delta_s = n["s"] - e["s"]
            delta_v = e["s_dot"] - n["s_dot"]

            if delta_v > 0:
                ttc = delta_s / delta_v
                if np.isfinite(ttc):
                    ttc_t1_values.append(ttc)

        # -------------------------
        # Average speeds
        # -------------------------
        ego_avg_speeds.append(ego["speed"].mean() * 3.6)
        npc_avg_speeds.append(npc["speed"].mean() * 3.6)

    return (
        np.array(ttc_t1_values),
        np.array(ego_avg_speeds),
        np.array(npc_avg_speeds),
    )


# =========================================================
# CUT-IN
# =========================================================

def cut_in_stats(actor_frames):
    af = actor_frames[actor_frames["scenario"] == "cut_in.osc"].copy()

    group_cols = ["scene_id", "segment_id", "t0", "t1"]

    min_ttc_values = []

    for _, group in af.groupby(group_cols):

        ego = group[group["role"] == "ego_vehicle"].copy()
        npc = group[group["role"] == "npc"].copy()

        if ego.empty or npc.empty:
            continue

        ego = ego.sort_values("t")
        npc = npc.sort_values("t")

        npc_sorted = npc.sort_values("t")

        lane_change_mask = (
            (npc_sorted["osc_lane_id"] == -1)
            & (npc_sorted["osc_lane_id"].shift(1) == -2)
        )

        if not lane_change_mask.any():
            continue

        t_change = npc_sorted[lane_change_mask]["t"].iloc[0]

        ego_post = ego[(ego["t"] >= t_change) & (ego["t"] <= group["t1"].iloc[0])]
        npc_post = npc[(npc["t"] >= t_change) & (npc["t"] <= group["t1"].iloc[0])]

        if ego_post.empty or npc_post.empty:
            continue

        merged = pd.merge(
            ego_post,
            npc_post,
            on="t",
            suffixes=("_ego", "_npc"),
        )

        if merged.empty:
            continue

        delta_s = merged["s_npc"] - merged["s_ego"]
        delta_v = merged["s_dot_ego"] - merged["s_dot_npc"]

        valid = (delta_v > 0) & (delta_s > 0)

        ttc_series = np.where(valid, delta_s / delta_v, np.nan)

        min_ttc = np.nanmin(ttc_series)

        if np.isfinite(min_ttc):
            min_ttc_values.append(min_ttc)

    return np.array(min_ttc_values)