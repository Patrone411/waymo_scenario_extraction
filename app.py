"""
app.py — Waymo Scenario Explorer

Streamlit demo for the Waymo Scenario Extraction portfolio project.
Shows matched OSC2 scenarios interactively with trajectory plots,
road geometry, and interaction time series.

Run:
    streamlit run app.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ── Path setup: scenario_plot.py is in the same directory or repo root
_HERE = Path(__file__).resolve().parent
for _candidate in [_HERE, _HERE.parent, _HERE / "scenario_extraction"]:
    if (_candidate / "scenario_plot.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from scenario_plot import plot_hit, plot_hits_grid, plot_hit_plotly, plot_hit_animated
from app_stats import cpna_stats, ccrb_stats, cut_in_stats
from app_plots import plot_cpna, plot_ccrb, plot_cut_in, plot_start_stats
# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

AWS_REGION      = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
RESULTS_BUCKET  = os.environ.get("RESULTS_BUCKET",  "womd-features")
FEATURES_PREFIX = os.environ.get("FEATURES_PREFIX", "parquet/run-001")
RESULTS_PREFIX  = os.environ.get("RESULTS_PREFIX",  "results")

SCENARIO_DESCRIPTIONS = {
    "cut_in.osc": "A cut-in scenario describes a traffic situation in which a Global Vehicle Target (GVT) "
    "changes from an adjacent lane into the current lane of the Vehicle Under Test (VUT) "
    "and merges in front of it. A key characteristic of this scenario is that the lane "
    "change occurs when the longitudinal distance between the two vehicles is relatively "
    "small, resulting in a potentially safety-critical interaction. The scenario is "
    "primarily characterized by the following parameters: the initial lane assignment of "
    "the GVT and the VUT, the speeds of both vehicles at the start of the scenario, the "
    "longitudinal distance at the moment the GVT merges into the VUT's lane, and the "
    "duration of the interaction window.",

    "cpna.osc":       "The Car-to-Pedestrian "
    "Nearside Adult (CPNA) scenario describes a traffic "
    "situation in which an adult pedestrian, acting as a Vulnerable Road User (VRU), "
    "enters the roadway from the roadside while the Vehicle Under Test (VUT) is "
    "approaching on the same road. A key characteristic of this scenario is the "
    "pedestrian's lateral crossing movement combined with the vehicle's approach, "
    "resulting in a potentially conflict-prone interaction between motorized traffic "
    "and an unprotected road user. The scenario is primarily described by the following "
    "parameters: the longitudinal and lateral distance between the VRU and the VUT at "
    "the start of the scenario, the vehicle's speed, and the pedestrian's lateral "
    "crossing movement across the lane.",

    "ccrb.osc":        "The Car-to-Car Rear Braking (CCRb) scenario describes a longitudinal vehicle-to-vehicle "
    "interaction in which the Global Vehicle Target (GVT) is positioned ahead of the Vehicle "
    "Under Test (VUT) in the same lane and performs a significant deceleration maneuver "
    "(braking). This scenario is characterized by the following features: the GVT and the "
    "VUT are assigned to the same lane, the initial speeds of both vehicles, the "
    "longitudinal distance at the start of the scenario, and a negative acceleration value "
    "of the GVT.",
}

# PNG und OSC2-Dateien liegen im selben Verzeichnis wie app.py.
# Namenskonvention: change_lane.osc → change_lane.png / change_lane.osc
_ASSETS_DIR = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Waymo Scenario Explorer",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Minimal custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* tighter header */
    .block-container { padding-top: 5rem; }

    /* metric cards */
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }

    /* hit details JSON block */
    .hit-details {
        background: #1e1e2e;
        color: #cdd6f4;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.82rem;
        line-height: 1.6;
    }

    /* scenario badge */
    .scenario-badge {
        display: inline-block;
        background: #0d6efd;
        color: white;
        border-radius: 4px;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading — cached so S3 is only queried once per session
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading match results from S3 …")
def load_hits() -> pd.DataFrame:
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs

    s3 = pafs.S3FileSystem(region=AWS_REGION)
    return ds.dataset(
        f"{RESULTS_BUCKET}/{RESULTS_PREFIX}/match_hits",
        filesystem=s3,
        format="parquet",
        partitioning="hive",
    ).to_table().to_pandas()


@st.cache_data(show_spinner="Loading actor frames …")
def load_actor_frames() -> pd.DataFrame:
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs

    s3 = pafs.S3FileSystem(region=AWS_REGION)
    return ds.dataset(
        f"{RESULTS_BUCKET}/{RESULTS_PREFIX}/match_actor_frames",
        filesystem=s3,
        format="parquet",
        partitioning="hive",
    ).to_table().to_pandas()

@st.cache_data
def load_pair_frames() -> pd.DataFrame:
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs

    s3 = pafs.S3FileSystem(region=AWS_REGION)

    return ds.dataset(
        f"{RESULTS_BUCKET}/{RESULTS_PREFIX}/match_pair_frames",
        filesystem=s3,
        format="parquet",
        partitioning="hive",
    ).to_table().to_pandas()




# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🚗 Scenario Explorer")
    st.caption("Waymo Open Dataset · OSC2 Matching Pipeline")
    st.divider()

    st.subheader("Filter")

    try:
        hits = load_hits()
        data_ok = True
    except Exception as e:
        st.error(f"S3 connection failed:\n{e}")
        st.info(
            "Make sure AWS credentials are set:\n"
            "`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`"
        )
        st.stop()


    preferred_order = ["cut_in.osc", "ccrb.osc", "cpna.osc"]

    scenarios = [
        s for s in preferred_order
        if s in hits["scenario"].unique()
    ]

    scenario = st.selectbox(
        "Scenario",
        scenarios,
        format_func=lambda s: s.replace(".osc", ""),
    )

    desc = SCENARIO_DESCRIPTIONS.get(scenario, "")
    if desc:
        st.caption(f"_{desc}_")

    st.divider()

    shards_available = sorted(hits["shard_index"].unique().tolist())
    shard_filter = st.multiselect(
        "Shards",
        options=shards_available,
        default=[],
        placeholder="All shards",
        help="Leave empty to include all shards.",
    )

    st.divider()

    show_polygons = st.toggle("Road polygons", value=True)
    show_ref_line = st.toggle("Reference line", value=True)
    #show_context  = st.toggle("Full trajectory (fine)", value=True)
    #show_interaction = st.toggle("Interaction inset", value=True)
    show_interaction = True
    show_animation = st.toggle("Enable animation", value=True)
    show_trail = st.toggle("trajectory trails in animation", value=True)

    st.divider()
    st.caption(
        "**Stack:** Python · PyArrow · AWS Batch · S3\n\n"
        "**Data:** [Waymo Open Dataset](https://waymo.com/open/)"
    )

    st.divider()
    

# ─────────────────────────────────────────────────────────────────────────────
# Filter hits
# ─────────────────────────────────────────────────────────────────────────────

mask = hits["scenario"] == scenario
if shard_filter:
    mask &= hits["shard_index"].isin(shard_filter)

filtered = hits[mask].reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header metrics
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("📖 User Guide", expanded=True):
    st.markdown("""
## Waymo Scenario Explorer — User Guide

This app lets you interactively explore traffic scenarios extracted from the
[Waymo Open Motion Dataset](https://waymo.com/open/) using a formal
OpenSCENARIO 2.0 matching pipeline.

---

### What is a "hit"?

A **hit** is a time window in a Waymo driving scene where real actor trajectories
match the constraints of an OSC2 scenario definition — for example, a vehicle
performing a lane change in front of another vehicle (Cut-In), a pedestrian
crossing the road (CPNA), or a leading vehicle braking hard (CCRb).

Each hit records:
- which scene and road segment it was found in
- the matched actor roles (e.g. `ego_vehicle`, `npc`)
- the start frame **t₀** and end frame **t₁** of the matched window (at 10 Hz)

---

### Sidebar controls

| Control | Effect |
|---------|--------|
| **Scenario** | Switch between Cut-In, CCRb, and CPNA scenarios |
| **Shards** | Filter by dataset shard (0–49). Leave empty for all shards |
| **Road polygons** | Show/hide filled lane polygons in the trajectory plot |
| **Reference line** | Show/hide the OSC2 reference line of the road segment |
| **Enable animation** | Toggle between static interactive plot and frame-by-frame animation |
| **Trajectory trails** | Show context trajectory (full 9.1 s) or only the matched window |

---

### 🎯 Single Hit tab

1. Use the **Select hit** dropdown to pick a matched scenario instance
2. The left panel shows the **trajectory plot** with road geometry and actor paths
3. The right panel shows **hit metadata**: scene ID, segment, timing, roles, and S3 source

**Plot controls (Plotly toolbar, top right of plot):**

| Button | Action |
|--------|--------|
| 🔍 | Box zoom — drag to zoom into a region |
| ✋ | Pan — drag to move around |
| 🏠 | Reset view |
| 📷 | Save as PNG |
| Hover | Move mouse over trajectory to see speed, position, TTC values |

**Animation mode** (when enabled in sidebar):

- Press **▶ Play** to start the frame-by-frame playback at 10 Hz
- Press **⏸ Pause** to stop at any frame
- Drag the **frame slider** to jump to a specific timestep
- Each actor marker shows its current position; the trail shows recent history

---

### ⊞ Grid View tab

Shows multiple hits side by side for quick visual comparison.

- Use the **Columns** slider to set the grid width (2–4 columns)
- Use the **Number of hits** slider to control how many hits are displayed
- All hits shown respect the current sidebar filters (scenario, shard)

---

### 📊 Statistics tab

Shows aggregated statistics for all hits matching the current scenario filter:

| Chart | Description |
|-------|-------------|
| **Hits per shard** | Distribution of matched scenarios across dataset shards |
| **Scenario-specific plots** | TTC, distance, and speed distributions for Cut-In / CCRb / CPNA |
| **Speed at t₀** | Speed distribution of each role at scenario start |
| **Scenario duration** | Distribution of matched window lengths in seconds |
| **Raw data** | Expandable table with all filtered hit rows |

---

### 📋 Scenario definitions

Each scenario can be inspected above the tabs:

- **🖼️ Example Trajectory** — reference plot showing a typical instance of the scenario
- **📄 OpenSCENARIO 2.0 Definition** — the formal `.osc` source file that was used for matching

---

### Data source

All data is loaded from **Amazon S3** on first access and cached for the session.
Feature Parquet files (scene geometry + actor trajectories) are read on demand
when a hit is selected — only the relevant scene file is fetched.

| S3 path | Content |
|---------|---------|
| `s3://womd-features/parquet/run-001/` | Feature Parquets (one file per scene) |
| `s3://womd-features/results/match_hits/` | Matched scenario hits |
| `s3://womd-features/results/match_actor_frames/` | Actor features at t₀ and t₁ |
| `s3://womd-features/results/match_pair_frames/` | Pairwise interaction features at t₀ and t₁ |

---

""")

st.markdown(
    f'<div class="scenario-badge">{scenario}</div>',
    unsafe_allow_html=True,
)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Hits found",      len(filtered))
col_m2.metric("Scenes",          filtered["scene_id"].nunique())
col_m3.metric("Avg duration (s)",
              f"{((filtered['t1'] - filtered['t0']) / 10.0).mean():.1f}"
              if not filtered.empty else "—")

st.divider()

if filtered.empty:
    st.warning("No hits for the current filters. Adjust the settings.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Scenario reference: example trajectory PNG + OSC2 source
# ─────────────────────────────────────────────────────────────────────────────

_scenario_stem = scenario.replace(".osc", "")
_png_path  = _ASSETS_DIR / f"{_scenario_stem}.png"
_osc_path  = _ASSETS_DIR / scenario

_has_png = _png_path.exists()
_has_osc = _osc_path.exists()


with st.expander("🖼️  Example Trajectory", expanded=False):
    if _has_png:
        st.image(str(_png_path),  width="stretch")
    else:
        st.caption(f"_{_scenario_stem}.png not found in {_ASSETS_DIR}_")

with st.expander("📄  OpenSCENARIO 2.0 Definition", expanded=False):
    if _has_osc:
        _osc_source = _osc_path.read_text(encoding="utf-8")
        st.code(_osc_source, language="python")   # OSC2 ~ Python syntax highlighting
    else:
        st.caption(f"_{scenario} not found in {_ASSETS_DIR}_")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs: Single Hit | Grid | Statistics
# ─────────────────────────────────────────────────────────────────────────────

tab_single, tab_grid, tab_stats = st.tabs([
    "🎯  Single Hit",
    "⊞  Grid View",
    "📊  Statistics",
])

# ── Tab 1: Single Hit ─────────────────────────────────────────────────────────

with tab_single:

    def _hit_label(i, row) -> str:
        roles = json.loads(row["roles_json"])
        roles_str = "  ·  ".join(f"{k}: {v}" for k, v in roles.items())
        return (
            f"#{i:04d}  |  {row['scene_id'][:14]}  |  {row['segment_id']}  |  "
            f"t₀={row['t0']} → t₁={row['t1']}  |  "
            f"n_win={row['n_windows']}  |  {roles_str}"
        )

    hit_labels = [_hit_label(i, row) for i, row in filtered.iterrows()]

    selected_idx = st.selectbox(
        "Select hit",
        range(len(hit_labels)),
        format_func=lambda i: hit_labels[i],
    )

    hit = filtered.iloc[selected_idx]
    roles = json.loads(hit["roles_json"])

    # two-column layout: plot left, details right
    col_plot, col_detail = st.columns([3, 1])

    with col_plot:
        with st.spinner("Loading scenario data …"):
            try:
                if show_animation:
                    fig = plot_hit_animated(
                        hit,
                        scenes_dir=f"s3://{RESULTS_BUCKET}/{FEATURES_PREFIX}",
                        show_polygons=show_polygons,
                        show_reference_line=show_ref_line,
                        trail_frames=show_trail,
                    )
                else:
                    fig = plot_hit_plotly(
                        hit,
                        scenes_dir=f"s3://{RESULTS_BUCKET}/{FEATURES_PREFIX}",
                        show_polygons=show_polygons,
                        show_reference_line=show_ref_line,
                        show_trajectories=show_trail,
                        show_markers=True,
                    )
                st.plotly_chart(fig,  width="stretch")
            except Exception as e:
                st.error(f"Error loading plot:\n{e}")

    with col_detail:
        st.subheader("Hit Details")

        duration_s = round((int(hit["t1"]) - int(hit["t0"])) / 10.0, 1)

        st.markdown(f"""
| Field | Value |
|-------|-------|
| **Scene** | `{hit['scene_id']}` |
| **Segment** | `{hit['segment_id']}` |
| **Shard** | `{hit['shard_index']}` |
| **t₀** | {hit['t0']} ({hit['t0']/10:.1f} s) |
| **t₁** | {hit['t1']} ({hit['t1']/10:.1f} s) |
| **Duration** | {duration_s} s |
| **Windows** | {hit['n_windows']} |
| **Block** | `{hit['block_label']}` |
""")

        st.subheader("Roles")
        for role, actor_id in roles.items():
            color = {"ego_vehicle": "🔵", "npc": "🔴"}.get(role, "⚪")
            st.markdown(f"{color} **{role}**: `{actor_id}`")

        st.subheader("Source")
        st.code(hit["source_uri"], language=None)

        if st.button("⬇ Save plot"):
            save_path = f"hit_{hit['scene_id'][:8]}_{hit['segment_id']}.png"
            try:
                fig2 = plot_hit(
                    hit,
                    scenes_dir=f"s3://{RESULTS_BUCKET}/{FEATURES_PREFIX}",
                    show_polygons=show_polygons,
                    show_reference_line=show_ref_line,
                    show_interaction=show_interaction,
                    figsize=(14, 9),
                    save_path=save_path,
                )
                plt.close(fig2)
                st.success(f"Saved: {save_path}")
            except Exception as e:
                st.error(str(e))


# ── Tab 2: Grid View ──────────────────────────────────────────────────────────

with tab_grid:

    col_g1, col_g2 = st.columns(2)
    n_cols      = col_g1.slider("Columns", 2, 4, 3)
    n_hits_grid = col_g2.slider("Number of hits", 4, 16, 9)

    sample = filtered.head(n_hits_grid)

    with st.spinner(f"Loading {len(sample)} hits …"):
        try:
            fig_grid = plot_hits_grid(
                sample,
                scenes_dir=f"s3://{RESULTS_BUCKET}/{FEATURES_PREFIX}",
                n_cols=n_cols,
                figsize_per_cell=(5, 5),
                show_road=show_polygons,
            )
            st.pyplot(fig_grid,  width="stretch")
            plt.close(fig_grid)
        except Exception as e:
            st.error(f"Grid error: {e}")


# ── Tab 3: Statistics ─────────────────────────────────────────────────────────

with tab_stats:

    import matplotlib.pyplot as plt
    import numpy as np

    st.subheader(f"Statistics — {scenario}")

    # ── Hits per shard ────────────────────────────────────────────────────────
    st.subheader("Hits per shard")
    hits_per_shard = (
        filtered.groupby("shard_index")["scene_id"]
        .count()
        .rename("n_hits")
        .reset_index()
    )
    fig_shard, ax_shard = plt.subplots(figsize=(12, 3))
    ax_shard.bar(
        hits_per_shard["shard_index"],
        hits_per_shard["n_hits"],
        color="#2196F3", edgecolor="white", linewidth=0.3,
    )
    ax_shard.set_xlabel("Shard index")
    ax_shard.set_ylabel("Hits")
    ax_shard.set_title(f"Hit distribution across shards — {scenario}")
    ax_shard.grid(True, alpha=0.3, axis="y")
    fig_shard.tight_layout()
    st.pyplot(fig_shard,  width="stretch")
    plt.close(fig_shard)

    # ── Speed distribution ────────────────────────────────────────────────────
    try:
        
        actor_frames = load_actor_frames()
        
        if scenario == "cpna.osc":
            pair_frames = load_pair_frames()
            min_dist, ttc = cpna_stats(actor_frames, pair_frames)
            st.pyplot(plot_cpna(min_dist, ttc))
            
        if scenario == "ccrb.osc":
            ttc, ego_spd, npc_spd = ccrb_stats(actor_frames)
            if len(ttc) > 0:
                fig = plot_ccrb(ttc, ego_spd, npc_spd)
                st.pyplot(fig)

        if scenario == "cut_in.osc":
            min_ttc_values = cut_in_stats(actor_frames)
            if len(min_ttc_values) == 0:
                st.warning("No valid TTC values found.")
            else:
                fig = plot_cut_in(min_ttc_values, actor_frames)
                st.pyplot(fig)
                plt.close(fig)



        af_filtered  = actor_frames[
            (actor_frames["scenario"] == scenario) &
            (actor_frames["frame"]    == "start")
        ]

        if not af_filtered.empty:

            fig = plot_start_stats(af_filtered)

            st.pyplot(fig, width="stretch")
            plt.close(fig)

        else:
            st.info("No actor frame data available for this scenario.")

    except Exception as e:
        st.warning(f"Could not load actor frames: {e}")


    # ── Raw data ──────────────────────────────────────────────────────────────
    with st.expander("Raw data (filtered hits)"):
        st.dataframe(
            filtered[[
                "scene_id", "segment_id", "shard_index",
                "t0", "t1", "n_windows", "roles_json", "source_uri"
            ]],
            width="content",
        )
