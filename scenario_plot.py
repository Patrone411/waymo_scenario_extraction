"""
scenario_plot.py

Plots a matched scenario hit — road geometry, actor trajectories,
and pairwise interaction data — from Feature Parquets and match result rows.

Usage
-----
    from scenario_plot import plot_hit

    # from match_hits DataFrame
    hit = hits_df.iloc[0]

    # local feature parquets
    plot_hit(hit, scenes_dir="test_output/00000/scenes")

    # S3 feature parquets
    plot_hit(hit, scenes_dir="s3://womd-features/parquet/run-001/00000/scenes")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pyarrow.parquet as pq
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Polygon,
    shape,
)

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

_ROLE_COLORS = {
    "ego_vehicle": "#2196F3",   # blue
    "npc":         "#F44336",   # red
    "pedestrian":  "#FF9800",   # orange
    "cyclist":     "#9C27B0",   # purple
}
_DEFAULT_ROLE_COLOR = "#607D8B"  # blue-grey for unknown roles

_ROAD_COLORS = {
    "target": "#90CAF9",   # light blue
    "left":   "#A5D6A7",   # light green
    "right":  "#FFCC80",   # light orange
}
_REFERENCE_LINE_COLOR = "#1A237E"   # dark navy


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (extracted from segment_polygon_handling.py style)
# ─────────────────────────────────────────────────────────────────────────────

def _scenes_dir_from_hit(hit: dict, scenes_dir: str) -> str:
    """
    If scenes_dir contains a shard placeholder or the hit has a source_uri,
    derive the correct scenes dir from source_uri.
    
    source_uri looks like:
      s3://womd-features/parquet/run-001/00003/scenes/4680e1fb10c57daa.parquet
    → scenes_dir = s3://womd-features/parquet/run-001/00003/scenes
    """
    source_uri = hit.get("source_uri", "")
    if source_uri and source_uri != "<unknown>":
        # strip the filename to get the directory
        if "/" in source_uri:
            return source_uri.rsplit("/", 1)[0]
    return scenes_dir


def _plot_line(
    ax: Axes,
    line: LineString,
    *,
    color: str,
    lw: float = 1.5,
    ls: str = "-",
    label: Optional[str] = None,
    alpha: float = 1.0,
) -> None:
    if line is None or line.is_empty:
        return
    x, y = line.xy
    ax.plot(x, y, linestyle=ls, linewidth=lw, color=color,
            label=label, alpha=alpha)


def _plot_line_with_arrows(
    ax: Axes,
    line: LineString,
    *,
    color: str,
    lw: float = 1.5,
    ls: str = "-",
    n_arrows: int = 8,
    alpha: float = 1.0,
    label: Optional[str] = None,
) -> None:
    if line is None or line.is_empty or line.length <= 0:
        return
    x, y = line.xy
    ax.plot(x, y, linestyle=ls, linewidth=lw, color=color,
            alpha=alpha, label=label)
    n = max(1, n_arrows)
    for i in range(1, n + 1):
        t1 = i / (n + 1)
        t0 = max(0.0, t1 - 0.02)
        p0 = line.interpolate(t0, normalized=True)
        p1 = line.interpolate(t1, normalized=True)
        ax.annotate(
            "",
            xy=(p1.x, p1.y),
            xytext=(p0.x, p0.y),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=max(1.0, lw - 0.2),
                alpha=alpha,
            ),
        )


def _plot_poly(
    ax: Axes,
    poly: Polygon,
    *,
    color: str,
    face_alpha: float = 0.25,
    edge_lw: float = 1.0,
    label: Optional[str] = None,
) -> None:
    if poly is None or poly.is_empty:
        return
    x, y = poly.exterior.xy
    ax.fill(x, y, alpha=face_alpha, color=color, label=label)
    ax.plot(x, y, linewidth=edge_lw, color=color)
    for ring in poly.interiors:
        xi, yi = ring.xy
        ax.plot(xi, yi, linewidth=edge_lw, color=color)


def _plot_poly_any(
    ax: Axes,
    geom,
    *,
    color: str,
    face_alpha: float = 0.25,
    edge_lw: float = 1.0,
    label: Optional[str] = None,
) -> None:
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        _plot_poly(ax, geom, color=color, face_alpha=face_alpha,
                   edge_lw=edge_lw, label=label)
    elif isinstance(geom, MultiPolygon):
        for i, g in enumerate(geom.geoms):
            _plot_poly(ax, g, color=color, face_alpha=face_alpha,
                       edge_lw=edge_lw, label=(label if i == 0 else None))
    elif isinstance(geom, GeometryCollection):
        for i, g in enumerate(geom.geoms):
            _plot_poly_any(ax, g, color=color, face_alpha=face_alpha,
                           edge_lw=edge_lw, label=(label if i == 0 else None))


# ─────────────────────────────────────────────────────────────────────────────
# Sparse decode helper
# ─────────────────────────────────────────────────────────────────────────────

def _decode_sparse(sparse: dict, T: int = 91) -> list:
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


def _safe_series(values: list, t0: int, t1: int) -> np.ndarray:
    """Return a float array for t0..t1, with None replaced by NaN."""
    window = values[t0 : t1 + 1]
    return np.array(
        [float(v) if v is not None else np.nan for v in window],
        dtype=float,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_scene_row(scene_id: str, segment_id: str, scenes_dir: str) -> dict:
    path = f"{scenes_dir.rstrip('/')}/{scene_id}.parquet"

    if path.startswith("s3://"):
        import pyarrow.fs as pafs
        without_scheme = path[5:]
        bucket, _, key = without_scheme.partition("/")
        fs = pafs.S3FileSystem(region=AWS_REGION)
        table = pq.read_table(f"{bucket}/{key}", filesystem=fs)
    else:
        table = pq.read_table(path)

    df   = table.to_pandas()
    rows = df[df["segment_id"] == segment_id]
    if rows.empty:
        raise ValueError(
            f"segment_id '{segment_id}' not found in scene '{scene_id}'"
        )
    return rows.iloc[0].to_dict()


def _parse_geom(json_str: Optional[str]):
    if not json_str:
        return None
    try:
        d = json.loads(json_str)
        if d is None:
            return None
        return shape(d)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main public API
# ─────────────────────────────────────────────────────────────────────────────

def plot_hit(
    hit,                          # pandas Series — one row from match_hits
    scenes_dir: str,              # local or S3 path to scenes/ directory
    *,
    # display options
    show_road: bool = True,
    show_polygons: bool = True,
    show_reference_line: bool = True,
    show_centerlines: bool = False,
    show_trajectories: bool = True,
    show_interaction: bool = True,
    show_markers: bool = True,    # start/end markers on trajectories
    n_arrows: int = 6,
    figsize: Tuple[int, int] = (14, 9),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> Figure:
    """
    Plot a matched scenario hit.

    Parameters
    ----------
    hit         : one row from the match_hits Parquet table (pandas Series or dict)
    scenes_dir  : path to the scenes/ directory containing feature Parquets
    show_road   : plot road polygons (target / left / right)
    show_polygons : fill road polygons (requires show_road=True)
    show_reference_line : highlight the reference line
    show_centerlines    : plot centerlines per chain (dotted)
    show_trajectories   : plot actor x/y trajectories over [t0, t1]
    show_interaction    : plot a small inset with distance/TTC over time
    show_markers        : mark t0 and t1 positions on trajectories
    n_arrows    : number of direction arrows on trajectories
    figsize     : figure size in inches
    title       : custom title (auto-generated if None)
    save_path   : if set, save figure to this path instead of showing
    ax          : if provided, draw into this existing Axes (disables inset)

    Returns
    -------
    matplotlib Figure
    """
    # ── unpack hit row ────────────────────────────────────────────────────────
    if hasattr(hit, "to_dict"):
        hit = hit.to_dict()

    scene_id   = hit["scene_id"]
    segment_id = hit["segment_id"]
    t0         = int(hit["t0"])
    t1         = int(hit["t1"])
    roles      = json.loads(hit["roles_json"])
    scenario   = hit.get("scenario", "")
    n_windows  = hit.get("n_windows", "?")

    
    # ── load feature data ─────────────────────────────────────────────────────
    resolved_scenes_dir = _scenes_dir_from_hit(hit, scenes_dir)
    row    = _load_scene_row(scene_id, segment_id, resolved_scenes_dir)
    actors = json.loads(row["actors_json"])

    # ── parse road geometry ───────────────────────────────────────────────────
    target_polygon = _parse_geom(row.get("target_polygon_json"))
    left_polygon   = _parse_geom(row.get("left_polygon_json"))
    right_polygon  = _parse_geom(row.get("right_polygon_json"))
    reference_line = _parse_geom(row.get("reference_line_json"))
    centerlines_raw = json.loads(row.get("centerlines_json") or "{}")

    centerlines: Dict[str, LineString] = {}
    for cid, geojson in (centerlines_raw or {}).items():
        if geojson:
            try:
                centerlines[str(cid)] = shape(geojson)
            except Exception:
                pass

    # ── set up figure ─────────────────────────────────────────────────────────
    if ax is not None:
        fig = ax.figure
        main_ax = ax
        inset_ax = None
    elif show_interaction and len(roles) >= 2:
        fig = plt.figure(figsize=figsize)
        gs  = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.35)
        main_ax   = fig.add_subplot(gs[:, 0])
        dist_ax   = fig.add_subplot(gs[0, 1])
        ttc_ax    = fig.add_subplot(gs[1, 1])
        speed_ax  = fig.add_subplot(gs[2, 1])
        inset_ax  = (dist_ax, ttc_ax, speed_ax)
    else:
        fig, main_ax = plt.subplots(figsize=figsize)
        inset_ax = None

    # ── road geometry ─────────────────────────────────────────────────────────
    if show_road:
        if show_polygons:
            _plot_poly_any(
                main_ax, target_polygon,
                color=_ROAD_COLORS["target"], face_alpha=0.35, edge_lw=1.5,
                label="target lane",
            )
            _plot_poly_any(
                main_ax, left_polygon,
                color=_ROAD_COLORS["left"], face_alpha=0.25, edge_lw=1.0,
                label="left lane",
            )
            _plot_poly_any(
                main_ax, right_polygon,
                color=_ROAD_COLORS["right"], face_alpha=0.25, edge_lw=1.0,
                label="right lane",
            )
        else:
            # outlines only
            for poly, color, label in [
                (target_polygon, _ROAD_COLORS["target"], "target lane"),
                (left_polygon,   _ROAD_COLORS["left"],   "left lane"),
                (right_polygon,  _ROAD_COLORS["right"],  "right lane"),
            ]:
                if poly and not poly.is_empty:
                    x, y = poly.exterior.xy
                    main_ax.plot(x, y, color=color, linewidth=1.5,
                                 label=label, alpha=0.8)

    if show_reference_line and reference_line:
        _plot_line_with_arrows(
            main_ax, reference_line,
            color=_REFERENCE_LINE_COLOR, lw=2.5, ls="-",
            n_arrows=max(3, n_arrows // 2),
            label="reference line",
        )
        mid = reference_line.interpolate(0.5, normalized=True)
        ref_source = row.get("reference_line_source", "")
        if ref_source:
            main_ax.text(
                mid.x, mid.y, f"REF",
                fontsize=7, color=_REFERENCE_LINE_COLOR,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
            )

    if show_centerlines:
        cl_colors = ["#4CAF50", "#FF5722", "#9E9E9E"]
        for i, (cid, cl) in enumerate(centerlines.items()):
            col = cl_colors[i % len(cl_colors)]
            _plot_line_with_arrows(
                main_ax, cl,
                color=col, lw=1.2, ls=":",
                n_arrows=n_arrows,
                alpha=0.7,
                label=f"centerline {cid}",
            )

    # ── actor trajectories ────────────────────────────────────────────────────
    t_frames = list(range(t0, t1 + 1))

    actor_data: Dict[str, dict] = {}  # role → {x, y, speed, s, ...}

    for role, actor_id in roles.items():
        ts     = actors["actor_ts"].get(actor_id, {})
        seg_ts = actors["seg_actor_ts"].get(actor_id, {})

        x_full = ts.get("x") or []
        y_full = ts.get("y") or []

        x_win = _safe_series(x_full, t0, t1)
        y_win = _safe_series(y_full, t0, t1)

        speed_full = ts.get("long_v") or []
        speed_win  = _safe_series(speed_full, t0, t1)

        s_full = seg_ts.get("s") or []
        s_win  = _safe_series(s_full, t0, t1)

        actor_data[role] = {
            "actor_id": actor_id,
            "x": x_win, "y": y_win,
            "speed": speed_win, "s": s_win,
        }

        if not show_trajectories:
            continue

        color = _ROLE_COLORS.get(role, _DEFAULT_ROLE_COLOR)

        # full context (outside window) — faint
        x_full_arr = _safe_series(x_full, 0, len(x_full) - 1)
        y_full_arr = _safe_series(y_full, 0, len(y_full) - 1)
        main_ax.plot(
            x_full_arr, y_full_arr,
            color=color, linewidth=0.8, alpha=0.2, linestyle="--",
        )

        # window trajectory with arrows
        _plot_line_with_arrows(
            main_ax,
            LineString(
                [(x, y) for x, y in zip(x_win, y_win)
                 if not (np.isnan(x) or np.isnan(y))]
            ) if len(x_win) >= 2 else LineString(),
            color=color, lw=2.5, ls="-",
            n_arrows=n_arrows,
            label=f"{role} ({actor_id})",
        )

        if show_markers:
            # t0 marker — filled circle
            if not np.isnan(x_win[0]) and not np.isnan(y_win[0]):
                main_ax.plot(
                    x_win[0], y_win[0],
                    "o", color=color, markersize=10, zorder=5,
                )
                main_ax.text(
                    x_win[0], y_win[0], f" t₀",
                    fontsize=8, color=color,
                    va="bottom",
                )
            # t1 marker — filled square
            if not np.isnan(x_win[-1]) and not np.isnan(y_win[-1]):
                main_ax.plot(
                    x_win[-1], y_win[-1],
                    "s", color=color, markersize=10, zorder=5,
                )
                main_ax.text(
                    x_win[-1], y_win[-1], f" t₁",
                    fontsize=8, color=color,
                    va="bottom",
                )

    # ── interaction inset plots ───────────────────────────────────────────────
    if inset_ax is not None and len(roles) >= 2:
        dist_ax, ttc_ax, speed_ax = inset_ax
        role_list = list(roles.items())
        role_a, actor_a = role_list[0]
        role_b, actor_b = role_list[1]

        # get inter-actor sparse data
        key_ab = f"{actor_a}|{actor_b}"
        key_ba = f"{actor_b}|{actor_a}"
        pair   = (actors["inter_actor"].get(key_ab)
                  or actors["inter_actor"].get(key_ba))

        t_axis = [t / 10.0 for t in range(t0, t1 + 1)]  # seconds at 10 Hz

        if pair:
            dist_full = _decode_sparse(pair.get("eucl_distance") or {})
            ttc_full  = _decode_sparse(pair.get("ttc")           or {})

            dist_win  = _safe_series(dist_full, t0, t1)
            ttc_win   = _safe_series(ttc_full,  t0, t1)

            # distance
            dist_ax.plot(t_axis, dist_win, color="#1565C0", linewidth=2)
            dist_ax.set_ylabel("Distance (m)", fontsize=8)
            dist_ax.set_title("Euclidean Distance", fontsize=9)
            dist_ax.grid(True, alpha=0.3)
            dist_ax.set_xlabel("time (s)", fontsize=8)

            # TTC — clip to 0..20 s for readability
            ttc_clipped = np.where(
                np.isfinite(ttc_win) & (ttc_win < 20), ttc_win, np.nan
            )
            ttc_ax.plot(t_axis, ttc_clipped, color="#B71C1C", linewidth=2)
            ttc_ax.axhline(3.0, color="#EF9A9A", linewidth=1,
                           linestyle="--", label="3 s threshold")
            ttc_ax.set_ylabel("TTC (s)", fontsize=8)
            ttc_ax.set_title("Time-to-Collision", fontsize=9)
            ttc_ax.legend(fontsize=7)
            ttc_ax.grid(True, alpha=0.3)
            ttc_ax.set_xlabel("time (s)", fontsize=8)
        else:
            for a in (dist_ax, ttc_ax):
                a.text(0.5, 0.5, "no pair data",
                       ha="center", va="center", transform=a.transAxes,
                       fontsize=9, color="grey")

        # speed per role
        for role, actor_id in roles.items():
            color = _ROLE_COLORS.get(role, _DEFAULT_ROLE_COLOR)
            spd   = actor_data.get(role, {}).get("speed")
            if spd is not None and len(spd) > 0:
                speed_ax.plot(
                    t_axis, spd * 3.6,  # m/s → km/h
                    color=color, linewidth=2,
                    label=f"{role} ({actor_id})",
                )
        speed_ax.set_ylabel("Speed (km/h)", fontsize=8)
        speed_ax.set_title("Speed", fontsize=9)
        speed_ax.legend(fontsize=7)
        speed_ax.grid(True, alpha=0.3)
        speed_ax.set_xlabel("time (s)", fontsize=8)

    # ── main axis formatting ──────────────────────────────────────────────────
    main_ax.set_aspect("equal", adjustable="datalim")
    main_ax.grid(True, linewidth=0.4, alpha=0.5)
    main_ax.margins(0.08)
    main_ax.legend(loc="best", fontsize=8)

    duration_s = (t1 - t0) / 10.0
    auto_title = (
        f"{scenario}  |  scene {scene_id}  |  {segment_id}\n"
        f"t₀={t0} → t₁={t1}  ({duration_s:.1f} s, {n_windows} windows)  |  "
        + "  ".join(f"{role}: {aid}" for role, aid in roles.items())
    )
    main_ax.set_title(title or auto_title, fontsize=9)
    main_ax.set_xlabel("x (m)")
    main_ax.set_ylabel("y (m)")

    if ax is None:   # nur wenn wir die Figure selbst erstellt haben
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"saved → {save_path}")
        else:
            plt.show()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: plot multiple hits in a grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_hits_grid(
    hits,               # iterable of hit rows (pandas DataFrame or list of dicts)
    scenes_dir: str,
    *,
    n_cols: int = 3,
    figsize_per_cell: Tuple[int, int] = (5, 5),
    show_road: bool = True,
    show_interaction: bool = False,  # disabled in grid mode for clarity
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot multiple hits in a grid layout — useful for quick visual inspection.

    Parameters
    ----------
    hits             : iterable of hit rows
    scenes_dir       : path to scenes/ directory
    n_cols           : number of columns in the grid
    figsize_per_cell : (width, height) in inches per cell
    save_path        : if set, save figure instead of showing
    """
    hit_list = list(hits) if not hasattr(hits, "iterrows") else [
        row for _, row in hits.iterrows()
    ]
    n = len(hit_list)
    if n == 0:
        print("No hits to plot.")
        return plt.figure()

    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
    )
    axes_flat = np.array(axes).ravel()

    for i, hit in enumerate(hit_list):
        try:
            plot_hit(
                hit,
                scenes_dir,
                show_road=show_road,
                show_interaction=False,
                show_centerlines=False,
                n_arrows=4,
                figsize=(5, 5),
                ax=axes_flat[i],
            )
        except Exception as e:
            import traceback
            axes_flat[i].text(
                0.5, 0.5, f"error:\n{e}",
                ha="center", va="center",
                transform=axes_flat[i].transAxes,
                fontsize=8, color="red",
            )
            traceback.print_exc()

    # hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved → {save_path}")
    else:
        plt.show()   # ← nur einmal hier, nicht in jedem plot_hit Aufruf

    return fig

def plot_hit_plotly(
    hit,
    scenes_dir: str,
    *,
    show_polygons: bool = True,
    show_reference_line: bool = True,
    show_trajectories: bool = True,
    show_markers: bool = True,
):
    """
    Interaktive Plotly-Version von plot_hit.
    Zoom, Pan, Hover direkt im Browser.
    """
    import plotly.graph_objects as go

    if hasattr(hit, "to_dict"):
        hit = hit.to_dict()

    scene_id   = hit["scene_id"]
    segment_id = hit["segment_id"]
    t0         = int(hit["t0"])
    t1         = int(hit["t1"])
    roles      = json.loads(hit["roles_json"])
    scenario   = hit.get("scenario", "")

    resolved_dir = _scenes_dir_from_hit(hit, scenes_dir)
    row    = _load_scene_row(scene_id, segment_id, resolved_dir)
    actors = json.loads(row["actors_json"])

    fig = go.Figure()

    # ── Fahrbahnpolygone ──────────────────────────────────────────────────────
    if show_polygons:
        _poly_configs = [
            ("target_polygon_json", "rgba(33,150,243,0.25)", "#2196F3", "target lane"),
            ("left_polygon_json",   "rgba(76,175,80,0.20)",  "#4CAF50", "left lane"),
            ("right_polygon_json",  "rgba(255,152,0,0.20)",  "#FF9800", "right lane"),
        ]
        for json_key, fill_color, line_color, name in _poly_configs:
            geom = _parse_geom(row.get(json_key))
            if geom is None or geom.is_empty:
                continue
            polys = (
                list(geom.geoms)
                if isinstance(geom, (MultiPolygon, GeometryCollection))
                else [geom]
            )
            for i, poly in enumerate(polys):
                if not isinstance(poly, Polygon) or poly.is_empty:
                    continue
                x, y = poly.exterior.xy
                fig.add_trace(go.Scatter(
                    x=list(x), y=list(y),
                    fill="toself",
                    fillcolor=fill_color,
                    line=dict(color=line_color, width=1.5),
                    name=name if i == 0 else None,
                    legendgroup=name,
                    showlegend=(i == 0),
                    hoverinfo="skip",
                    mode="lines",
                ))

    # ── Referenzlinie ─────────────────────────────────────────────────────────
    if show_reference_line:
        ref = _parse_geom(row.get("reference_line_json"))
        if ref and not ref.is_empty:
            x, y = ref.xy
            fig.add_trace(go.Scatter(
                x=list(x), y=list(y),
                mode="lines",
                line=dict(color="#1A237E", width=2.5, dash="solid"),
                name="reference line",
                hoverinfo="skip",
            ))

    # ── Akteur-Trajektorien ───────────────────────────────────────────────────
    _plotly_role_colors = {
        "ego_vehicle": "#2196F3",
        "npc":         "#F44336",
        "pedestrian":  "#FF9800",
        "cyclist":     "#9C27B0",
    }

    t_seconds = [t / 10.0 for t in range(t0, t1 + 1)]

    for role, actor_id in roles.items():
        ts     = actors["actor_ts"].get(actor_id, {})
        seg_ts = actors["seg_actor_ts"].get(actor_id, {})
        color  = _plotly_role_colors.get(role, "#607D8B")

        x_full   = ts.get("x")      or []
        y_full   = ts.get("y")      or []
        spd_full = ts.get("long_v") or []
        s_full   = (seg_ts.get("s") or [])

        x_win   = _safe_series(x_full,   t0, t1)
        y_win   = _safe_series(y_full,   t0, t1)
        spd_win = _safe_series(spd_full, t0, t1)
        s_win   = _safe_series(s_full,   t0, t1)

        # hover text
        hover = [
            f"<b>{role}</b> ({actor_id})<br>"
            f"t={t0+i} ({t_seconds[i]:.1f}s)<br>"
            f"x={x_win[i]:.1f}  y={y_win[i]:.1f}<br>"
            f"speed={spd_win[i]*3.6:.1f} km/h<br>"
            f"s={s_win[i]:.1f} m"
            for i in range(len(x_win))
        ]

        # vollständiger Kontext (fein gestrichelt)
        if show_trajectories:
            x_ctx = _safe_series(x_full, 0, len(x_full) - 1)
            y_ctx = _safe_series(y_full, 0, len(y_full) - 1)
            fig.add_trace(go.Scatter(
                x=list(x_ctx), y=list(y_ctx),
                mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.25,
                name=f"{role} (context)",
                legendgroup=role,
                showlegend=False,
                hoverinfo="skip",
            ))

        # Matching-Fenster
        fig.add_trace(go.Scatter(
            x=list(x_win), y=list(y_win),
            mode="lines+markers" if show_markers else "lines",
            line=dict(color=color, width=3),
            marker=dict(size=5, color=color),
            name=f"{role} ({actor_id})",
            legendgroup=role,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))
        
        # t0 / t1 Marker
        if show_markers:
            for t_mark, label, symbol in [
                (0, f"t₀={t0}", "circle"),
                (-1, f"t₁={t1}", "square"),
            ]:
                if not (np.isnan(x_win[t_mark]) or np.isnan(y_win[t_mark])):
                    fig.add_trace(go.Scatter(
                        x=[x_win[t_mark]], y=[y_win[t_mark]],
                        mode="markers+text",
                        marker=dict(size=12, color=color, symbol=symbol,
                                    line=dict(color="white", width=2)),
                        text=[label],
                        textposition="top center",
                        textfont=dict(size=10, color=color),
                        name=label,
                        legendgroup=role,
                        showlegend=False,
                        hoverinfo="skip",
                    ))

    # ── Layout ────────────────────────────────────────────────────────────────
    duration_s = (t1 - t0) / 10.0
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{scenario}</b>  |  {scene_id}  |  {segment_id}<br>"
                f"<sup>t₀={t0} → t₁={t1}  ({duration_s:.1f} s)  |  "
                + "  ·  ".join(f"{k}: {v}" for k, v in roles.items())
                + "</sup>"
            ),
            font=dict(size=12, color="#1a1a2e"),  # ← dunkel
        ),
        xaxis=dict(title="x (m)", scaleanchor="y", scaleratio=1,
                   showgrid=True, gridcolor="#eeeeee"),
        yaxis=dict(title="y (m)", showgrid=True, gridcolor="#eeeeee"),
        legend=dict(
            orientation="v", x=1.02, y=1,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#cccccc", borderwidth=1,
            font=dict(size=11, color="#1a1a2e"),  # ← dunkel
        ),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=180, t=80, b=60),
        height=600,
        font=dict(color="#1a1a2e", size=11),  # ← globale Schriftfarbe
    
    )
    
    fig.update_xaxes(
        tickfont_color="#333333",
        title_font_color="#333333",
        gridcolor="#e8e8e8",
    )
    fig.update_yaxes(
        tickfont_color="#333333",
        title_font_color="#333333",
        gridcolor="#e8e8e8",
    )
    fig.update_annotations(font_color="#333333")
    
    return fig


def plot_hit_animated(
    hit,
    scenes_dir: str,
    *,
    show_polygons: bool = True,
    show_reference_line: bool = True,
    fps: int = 10,
    trail_frames: bool = True,
    height: int = 620,
) -> "go.Figure":
    """
    Animierter Plotly-Plot — zeigt Akteure Frame-by-Frame bei 10 Hz.
 
    Parameter
    ---------
    hit           : eine Zeile aus match_hits (pandas Series oder dict)
    scenes_dir    : lokaler oder S3-Pfad zum scenes/-Verzeichnis
    show_polygons : Fahrbahnpolygone anzeigen
    show_reference_line : Referenzlinie anzeigen
    fps           : Aufnahme-Framerate (default 10 Hz)
    trail_frames  : sollen trajektorien trails angezeigt werden
    height        : Höhe der Figure in Pixeln
 
    Verwendung in Streamlit:
        fig = plot_hit_animated(hit, scenes_dir="s3://...")
        st.plotly_chart(fig, use_container_width=True)
    """
    import json
    import numpy as np
    import plotly.graph_objects as go
    from shapely.geometry import MultiPolygon, GeometryCollection, Polygon
 
    # ── Daten laden ───────────────────────────────────────────────────────────
    if hasattr(hit, "to_dict"):
        hit = hit.to_dict()
 
    t0         = int(hit["t0"])
    t1         = int(hit["t1"])
    roles      = json.loads(hit["roles_json"])
    scene_id   = hit["scene_id"]
    segment_id = hit["segment_id"]
    scenario   = hit.get("scenario", "")
    n_windows  = hit.get("n_windows", "?")
 
    resolved_dir = _scenes_dir_from_hit(hit, scenes_dir)
    row_data     = _load_scene_row(scene_id, segment_id, resolved_dir)
    actors       = json.loads(row_data["actors_json"])
 
    role_list = list(roles.items())
    n_frames  = t1 - t0 + 1
 
    # ── Statische Basis-Traces (Straße + Kontext-Trajektorien) ───────────────
    base_traces = []
 
    # Fahrbahnpolygone
    if show_polygons:
        for json_key, lane_role, (fill, line) in [
            ("target_polygon_json", "target", ("rgba(33,150,243,0.20)", "#2196F3")),
            ("left_polygon_json",   "left",   ("rgba(76,175,80,0.18)",  "#4CAF50")),
            ("right_polygon_json",  "right",  ("rgba(255,152,0,0.18)",  "#FF9800")),
        ]:
            geom = _parse_geom(row_data.get(json_key))
            if geom is None or geom.is_empty:
                continue
            polys = (
                [g for g in geom.geoms if isinstance(g, Polygon)]
                if isinstance(geom, (MultiPolygon, GeometryCollection))
                else [geom] if isinstance(geom, Polygon) else []
            )
            for i, poly in enumerate(polys):
                if poly.is_empty:
                    continue
                x, y = poly.exterior.xy
                base_traces.append(go.Scatter(
                    x=list(x) + [None],
                    y=list(y) + [None],
                    fill="toself",
                    fillcolor=fill,
                    line=dict(color=line, width=1.2),
                    mode="lines",
                    name=f"{lane_role} lane",
                    legendgroup=lane_role,
                    showlegend=(i == 0),
                    hoverinfo="skip",
                ))
 
    # Referenzlinie
    if show_reference_line:
        ref = _parse_geom(row_data.get("reference_line_json"))
        if ref and not ref.is_empty:
            x, y = ref.xy
            base_traces.append(go.Scatter(
                x=list(x), y=list(y),
                mode="lines",
                line=dict(color="#1A237E", width=2.5),
                name="reference line",
                legendgroup="refline",
                showlegend=True,
                hoverinfo="skip",
            ))
 
    # Vollständige Kontext-Trajektorie (statisch, fein gestrichelt)
    _role_colors = {
        "ego_vehicle": "#2196F3",
        "npc":         "#F44336",
        "pedestrian":  "#FF9800",
        "cyclist":     "#9C27B0",
    }
    _default_color = "#607D8B"
 
    all_series = {}
    for role, actor_id in role_list:
        ts     = actors["actor_ts"].get(actor_id, {})
        seg_ts = actors["seg_actor_ts"].get(actor_id, {})
        color  = _role_colors.get(role, _default_color)
 
        x_full   = ts.get("x")      or []
        y_full   = ts.get("y")      or []
        spd_full = ts.get("long_v") or []
        s_full   = seg_ts.get("s")  or []
 
        x_win   = _safe_series(x_full,   t0, t1)
        y_win   = _safe_series(y_full,   t0, t1)
        spd_win = _safe_series(spd_full, t0, t1)
        s_win   = _safe_series(s_full,   t0, t1)
 
        x_ctx = _safe_series(x_full, 0, max(0, len(x_full) - 1))
        y_ctx = _safe_series(y_full, 0, max(0, len(y_full) - 1))
 
        all_series[role] = {
            "actor_id": actor_id,
            "color":    color,
            "x":        x_win,
            "y":        y_win,
            "spd":      spd_win,
            "s":        s_win,
            "x_ctx":    x_ctx,
            "y_ctx":    y_ctx,
        }
 
        # Kontext-Trajektorie als statischer Trace
        base_traces.append(go.Scatter(
            x=list(x_ctx),
            y=list(y_ctx),
            mode="lines",
            line=dict(color=color, width=1, dash="dot"),
            opacity=0.18,
            name=f"{role} (Kontext)",
            legendgroup=role,
            showlegend=False,
            hoverinfo="skip",
        ))
 
    n_base = len(base_traces)
 
    # ── Animierte Traces: Trails, Marker──────────────────────────────
    # Diese werden als leere Platzhalter angelegt und in jedem Frame befüllt
    # alle Trails
    if trail_frames:
        for role, actor_id in role_list:
            color = all_series[role]["color"]

            base_traces.append(go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color=color, width=3),
                name=f"{role} ({actor_id})",
                legendgroup=role,
                legendgrouptitle_text=role,
                showlegend=True,
            ))

    # alle Marker
    for role, actor_id in role_list:
        color = all_series[role]["color"]

        base_traces.append(go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(
                size=16,
                color=color,
                line=dict(color="white", width=2),
            ),
            name=role,
            legendgroup=role,
            showlegend=False,
        ))


    # ── Animations-Frames ─────────────────────────────────────────────────────
    frame_duration_ms = int(1000 / fps)
    frames = []

    for fi in range(n_frames):
        t_abs = t0 + fi
        t_sec = t_abs / fps

        frame_data = []

        # --------------------------------------------------
        # 1. Trails
        # --------------------------------------------------
        if trail_frames:
            for role, actor_id in role_list:
                s = all_series[role]
                color = s["color"]

                trail_start = max(0, fi - n_frames + 1)

                trail_x = [
                    float(v) if not np.isnan(v) else None
                    for v in s["x"][trail_start:fi + 1]
                ]

                trail_y = [
                    float(v) if not np.isnan(v) else None
                    for v in s["y"][trail_start:fi + 1]
                ]

                frame_data.append(go.Scatter(
                    x=trail_x,
                    y=trail_y,
                    mode="lines",
                    line=dict(color=color, width=3),
                ))

        # --------------------------------------------------
        # 2. Marker
        # --------------------------------------------------
        for role, actor_id in role_list:
            s = all_series[role]
            color = s["color"]

            cx = float(s["x"][fi]) if not np.isnan(s["x"][fi]) else None
            cy = float(s["y"][fi]) if not np.isnan(s["y"][fi]) else None

            spd_kmh = (
                float(s["spd"][fi]) * 3.6
                if not np.isnan(s["spd"][fi])
                else 0.0
            )

            sv = (
                float(s["s"][fi])
                if not np.isnan(s["s"][fi])
                else 0.0
            )

            hover_txt = (
                f"<b>{role}</b> ({actor_id})<br>"
                f"t={t_abs} ({t_sec:.1f} s)<br>"
                f"x={cx:.1f} m  y={cy:.1f} m<br>"
                f"speed={spd_kmh:.1f} km/h<br>"
                f"s={sv:.1f} m"
                if cx is not None
                else f"<b>{role}</b><br>keine Daten"
            )

            frame_data.append(go.Scatter(
                x=[cx] if cx is not None else [],
                y=[cy] if cy is not None else [],
                mode="markers",
                marker=dict(
                    size=16,
                    color=color,
                    line=dict(color="white", width=2),
                ),
                hovertext=[hover_txt],
                hovertemplate="%{hovertext}<extra></extra>",
            ))


        frames.append(go.Frame(
            data=frame_data,
            name=str(fi),
            traces=list(
                range(
                    n_base,
                    n_base + len(role_list) * 3
                )
            ),
        ))
 
    # ── Slider ────────────────────────────────────────────────────────────────
    sliders = [dict(
        active=0,
        steps=[
            dict(
                method="animate",
                args=[[str(fi)], dict(
                    mode="immediate",
                    frame=dict(duration=frame_duration_ms, redraw=True),
                    transition=dict(duration=0),
                )],
                label=f"t={t0 + fi}  ({(t0 + fi) / fps:.1f}s)",
            )
            for fi in range(n_frames)
        ],
        x=0.0, y=-0.04,
        len=1.0,
        currentvalue=dict(
            prefix="Frame: ",
            font=dict(size=11, color="#333333"),
            xanchor="center",
            visible=True,
        ),
        transition=dict(duration=0),
        tickcolor="#333333",
        font=dict(color="#333333"),
        pad=dict(t=30),
    )]
 
    # ── Play / Pause Buttons ──────────────────────────────────────────────────
    updatemenus = [dict(
        type="buttons",
        showactive=False,
        x=0.0, y=-0.12,
        xanchor="left",
        bgcolor="#f0f0f0",
        bordercolor="#cccccc",
        font=dict(color="#1a1a2e", size=12),
        buttons=[
            dict(
                label="▶  Play",
                method="animate",
                args=[None, dict(
                    frame=dict(duration=frame_duration_ms, redraw=True),
                    fromcurrent=True,
                    transition=dict(duration=0),
                )],
            ),
            dict(
                label="⏸  Pause",
                method="animate",
                args=[[None], dict(
                    frame=dict(duration=0, redraw=False),
                    mode="immediate",
                    transition=dict(duration=0),
                )],
            ),
        ],
    )]
 
    # ── Figure zusammenbauen ──────────────────────────────────────────────────
    duration_s = (t1 - t0) / 10.0
    title_str  = (
        f"<b>{scenario}</b>  ·  Animation  ·  {scene_id}  ·  {segment_id}<br>"
        f"<sup>t0={t0} → t1={t1}  ({duration_s:.1f} s, {fps} Hz)  ·  "
        f"n_windows={n_windows}  ·  "
        + "  ·  ".join(f"{k}: {v}" for k, v in roles.items())
        + "</sup>"
    )
 
    fig = go.Figure(
        data=base_traces,
        frames=frames,
        layout=go.Layout(
            title=dict(
                text=title_str,
                font=dict(size=12, color="#1a1a2e"),
            ),
            xaxis=dict(
                title=dict(text="x (m)", font=dict(color="#333333", size=11)),
                showgrid=True,
                gridcolor="#e8e8e8",
                zeroline=False,
                scaleanchor="y",
                scaleratio=1,
                tickfont=dict(color="#333333", size=10),
            ),
            yaxis=dict(
                title=dict(text="y (m)", font=dict(color="#333333", size=11)),
                showgrid=True,
                gridcolor="#e8e8e8",
                zeroline=False,
                tickfont=dict(color="#333333", size=10),
            ),
            legend=dict(
                x=1.02, y=1,
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#cccccc",
                borderwidth=1,
                tracegroupgap=8,
                font=dict(size=11, color="#1a1a2e"),
            ),
            hovermode="closest",
            hoverlabel=dict(
                bgcolor="#1e1e2e",
                font_color="white",
                font_size=12,
                bordercolor="#444466",
            ),
            plot_bgcolor="white",
            paper_bgcolor="#fafafa",
            margin=dict(l=60, r=220, t=90, b=130),
            height=height,
            font=dict(color="#1a1a2e", size=11),
            sliders=sliders,
            updatemenus=updatemenus,
        ),
    )
 
    return fig