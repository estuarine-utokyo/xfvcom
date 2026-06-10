"""Panel-grid map of an unstructured (FVCOM) node field.

``plot_field_panels`` renders a list of node-length fields as a 2xN panel grid
with per-row colorbars, source-node markers and a per-panel info-text box. It is
the consolidated, reusable form of the dye source-attribution maps (previously
``TB-FVCOM/dye/analysis/plot_dye.py``); see
``docs/directions/20260610_panel_grid_map.md``.

Land rendering (``land=``):

- ``"mesh"`` (default): ordered mesh-boundary polygons -- white water, lightgray
  islands, black coastline. Shows exactly where the model has water.
- ``"osm"``: ``xcoast`` OSM real-land overlay (lazy import; presentation use).
- ``"none"``: no land.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.tri import Triangulation

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["plot_field_panels", "log_contour_levels", "log_colorbar_ticks"]


# ---------------------------------------------------------------------------
# mesh-boundary helpers (ported from the legacy dye renderer)
# ---------------------------------------------------------------------------
def _edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _extract_boundary_edges(nv0: np.ndarray) -> list[tuple[int, int]]:
    """Boundary edges (used once by exactly one triangle). ``nv0`` is (3, nele),
    0-based."""
    counts: Counter[tuple[int, int]] = Counter()
    for i in range(nv0.shape[1]):
        t = nv0[:, i]
        for j in range(3):
            counts[_edge(int(t[j]), int(t[(j + 1) % 3]))] += 1
    return [e for e, c in counts.items() if c == 1]


def _order_boundary_nodes(edges: list[tuple[int, int]]) -> list[list[int]]:
    """Order boundary edges into continuous closed paths (node-index lists)."""
    if not edges:
        return []
    adj: dict[int, list[int]] = {}
    for n1, n2 in edges:
        adj.setdefault(n1, []).append(n2)
        adj.setdefault(n2, []).append(n1)
    paths: list[list[int]] = []
    visited: set[tuple[int, int]] = set()
    for start in edges:
        if start in visited:
            continue
        path = [start[0], start[1]]
        visited.add(start)
        visited.add((start[1], start[0]))
        prev, cur = start
        while True:
            nxt = None
            for n in adj[cur]:
                if (_edge(cur, n) not in visited or n == path[0]) and n != prev:
                    nxt = n
                    break
            if nxt is None or nxt == path[0]:
                break
            path.append(nxt)
            visited.add(_edge(cur, nxt))
            visited.add(_edge(nxt, cur))
            prev, cur = cur, nxt
        if len(path) > 2:
            paths.append(path)
    return paths


def log_contour_levels(vmin: float, vmax: float) -> np.ndarray:
    """Contour levels evenly spaced within each decade (0.1,0.2,...,1,2,...)."""
    if vmin <= 0:
        vmin = 0.1
    lo, hi = int(np.floor(np.log10(vmin))), int(np.ceil(np.log10(vmax)))
    levels: list[float] = []
    for order in range(lo, hi + 1):
        for mult in range(1, 10):
            v = (10.0**order) * mult
            if vmin <= v <= vmax:
                levels.append(v)
            elif v > vmax:
                break
    if levels and levels[-1] < vmax:
        levels.append(vmax)
    if levels and levels[0] > vmin:
        levels.insert(0, vmin)
    return np.array(levels)


def _fmt_tick(t: float) -> str:
    return str(int(t)) if t >= 1 else str(t)


def log_colorbar_ticks(
    vmin: float, vmax: float
) -> tuple[list[float], list[str], list[float]]:
    """Decade ticks (labeled) + intermediate ticks (unlabeled minor) for a log
    colorbar.

    Returns ``(major_ticks, major_labels, minor_ticks)``. Only the decade ticks
    (``..., 0.1, 1, 10, 100, ...``) plus the ``vmin``/``vmax`` endpoints are
    labeled; the intermediate band ticks (``0.2, 0.5, 2, 5, ...``) are returned
    separately so callers can draw them as unlabeled minor ticks -- this keeps
    the band structure visible without colliding tick text (e.g. ``1`` & ``2``,
    ``100`` & ``200``).
    """
    candidates = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    in_range = [t for t in candidates if vmin <= t <= vmax]
    majors = [t for t in in_range if float(np.log10(t)).is_integer()]
    for end in (vmin, vmax):
        if end in in_range and end not in majors:
            majors.append(end)
    majors = sorted(set(majors))
    labels = [_fmt_tick(t) for t in majors]
    minors = [t for t in in_range if t not in majors]
    return majors, labels, minors


def _linear_levels(vmin: float, vmax: float, n: int = 20) -> Any:
    if vmin == 0 and vmax == 20:
        return list(range(0, 21, 1))
    if vmin == 0 and vmax == 50:
        return list(range(0, 52, 2))
    if vmin == 0 and vmax == 100:
        return list(range(0, 105, 5))
    return np.linspace(vmin, vmax, n)


def _values(field: Any) -> np.ndarray:
    """Accept a numpy array or an xarray.DataArray."""
    return np.asarray(getattr(field, "values", field), dtype=float)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def plot_field_panels(
    fields: Sequence[Any],
    lon: np.ndarray,
    lat: np.ndarray,
    nv: np.ndarray,
    *,
    labels: Sequence[str] | None = None,
    marker_nodes: Sequence[Sequence[int]] | None = None,
    layout: list[int] | None = None,
    vmin: float = 0.1,
    vmax: float = 500.0,
    log_scale: bool = True,
    cmap: str = "turbo",
    cbar_label: str = "",
    land: str = "mesh",
    osm_region: str | tuple[float, float, float, float] | None = None,
    land_facecolor: str = "lightgray",
    water_facecolor: str = "white",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    fontsize_label: float = 16,
    fontsize_tick: float = 14,
    info_fontsize: float = 14,
    marker_size: float = 40,
    dpi: int = 300,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Render node fields as a 2xN panel grid (one field per panel).

    Parameters
    ----------
    fields
        Node-length arrays / DataArrays, one per panel.
    lon, lat, nv
        Grid: node coordinates and connectivity ``nv`` (3, nele), 1- or 0-based.
    labels
        Per-panel info-text (bottom-right box); ``"\\n"``-separated lines.
    marker_nodes
        Per-panel 1-based node IDs to mark with dots (e.g. source nodes).
    layout
        Panels per row, e.g. ``[6, 6]`` for a 2x6 grid. Default: single row.
    vmin, vmax, log_scale, cmap, cbar_label
        Color scaling. Log uses gray-under / magenta-over and decade levels.
    land
        ``"mesh"`` (ordered mesh-boundary land fill, default), ``"osm"``
        (xcoast OSM real land overlay), or ``"none"``.
    osm_region
        xcoast preset name or ``(w, s, e, n)`` bbox when ``land="osm"``.
    info_fontsize
        Font size of the per-panel info-text box (river name / period /
        vertical-level lines). Defaults to ``14`` (panel-relative, larger than
        the axis ticks so the box matches the published maps).

    Returns
    -------
    dict
        ``{"fig", "axes", "output_path"}``.
    """
    n_panels = len(fields)
    if labels is not None and len(labels) != n_panels:
        raise ValueError("labels must match fields")
    nv0 = (nv - 1) if nv.min() >= 1 else nv
    tri = Triangulation(lon, lat, nv0.T)

    if layout is None:
        layout = [n_panels]
    n_rows = len(layout)
    max_cols = max(layout)
    if figsize is None:
        figsize = (max(4 * max_cols, 5), 6 * n_rows)

    fig, axes_2d = plt.subplots(n_rows, max_cols, figsize=figsize)
    axes_arr = np.atleast_2d(axes_2d)
    if max_cols == 1 and n_rows > 1:
        axes_arr = axes_arr.reshape(n_rows, 1)

    # land geometry
    boundary_paths: list[list[int]] = []
    if land == "mesh":
        boundary_paths = _order_boundary_nodes(_extract_boundary_edges(nv0))
        boundary_paths.sort(key=len, reverse=True)
    osm_mask = None
    if land == "osm":
        import xcoast  # lazy: keep geospatial stack optional

        if osm_region is None:
            osm_region = (
                float(lon.min()),
                float(lat.min()),
                float(lon.max()),
                float(lat.max()),
            )
        osm_mask = xcoast.load(osm_region)

    # levels / norm / colormap
    if log_scale:
        levels: Any = log_contour_levels(vmin, vmax)
        norm: Normalize = BoundaryNorm(levels, ncolors=256, clip=False)
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_under("#A0A0A0")
        cmap_obj.set_over("#FF00FF")
        extend = "both"
        tcf_kw: dict[str, Any] = {"levels": levels, "cmap": cmap_obj, "norm": norm}
    else:
        levels = _linear_levels(vmin, vmax)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
        extend = "max" if vmin == 0 else "both"
        tcf_kw = {"levels": levels, "cmap": cmap_obj, "vmin": vmin, "vmax": vmax}
    tcf_kw["extend"] = extend
    tcf_kw["zorder"] = 3

    idx = 0
    panels_per_row: list[int] = []
    for row_idx, n_cols in enumerate(layout):
        in_row = min(n_cols, n_panels - idx)
        panels_per_row.append(in_row)
        for col_idx in range(max_cols):
            ax = axes_arr[row_idx, col_idx]
            if col_idx >= in_row or idx >= n_panels:
                ax.axis("off")
                continue
            _draw_panel(
                ax,
                tri,
                lon,
                lat,
                _values(fields[idx]),
                boundary_paths=boundary_paths,
                osm_mask=osm_mask,
                land=land,
                land_facecolor=land_facecolor,
                water_facecolor=water_facecolor,
                tcf_kw=tcf_kw,
                log_scale=log_scale,
                vmin=vmin,
                label=None if labels is None else labels[idx],
                markers=None if marker_nodes is None else marker_nodes[idx],
                show_y=(col_idx == 0),
                fontsize_label=fontsize_label,
                fontsize_tick=fontsize_tick,
                info_fontsize=info_fontsize,
                marker_size=marker_size,
                xlim=xlim,
                ylim=ylim,
            )
            idx += 1

    _add_row_colorbars(
        fig,
        axes_arr,
        layout,
        panels_per_row,
        norm,
        cmap_obj,
        extend,
        cbar_label,
        vmin,
        vmax,
        log_scale,
        fontsize_label,
        fontsize_tick,
    )

    out_path = None
    if output is not None:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return {"fig": fig, "axes": axes_arr, "output_path": out_path}


def _draw_panel(
    ax: Any,
    tri: Triangulation,
    lon: np.ndarray,
    lat: np.ndarray,
    data: np.ndarray,
    *,
    boundary_paths: list[list[int]],
    osm_mask: Any,
    land: str,
    land_facecolor: str,
    water_facecolor: str,
    tcf_kw: dict[str, Any],
    log_scale: bool,
    vmin: float,
    label: str | None,
    markers: Sequence[int] | None,
    show_y: bool,
    fontsize_label: float,
    fontsize_tick: float,
    info_fontsize: float,
    marker_size: float,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
) -> None:
    if land == "mesh" and boundary_paths:
        ax.set_facecolor(land_facecolor)
        ext = boundary_paths[0]
        coords = [(lon[n], lat[n]) for n in ext] + [(lon[ext[0]], lat[ext[0]])]
        ax.fill(
            [p[0] for p in coords],
            [p[1] for p in coords],
            color=water_facecolor,
            zorder=1,
        )
        for path in boundary_paths[1:]:
            coords = [(lon[n], lat[n]) for n in path] + [(lon[path[0]], lat[path[0]])]
            ax.fill(
                [p[0] for p in coords],
                [p[1] for p in coords],
                color=land_facecolor,
                zorder=2,
            )
    elif land == "osm":
        ax.set_facecolor(water_facecolor)

    plot_data = np.clip(data, vmin / 100, None) if log_scale else data
    ax.tricontourf(tri, plot_data, **tcf_kw)

    if land == "mesh" and boundary_paths:
        for path in boundary_paths:
            lons = [lon[n] for n in path] + [lon[path[0]]]
            lats = [lat[n] for n in path] + [lat[path[0]]]
            ax.plot(lons, lats, color="black", linewidth=0.8, zorder=10)
    elif land == "osm" and osm_mask is not None:
        osm_mask.add_to_plain_mpl(ax, facecolor=land_facecolor, zorder=11)

    if markers:
        ii = [int(n) - 1 for n in markers]
        ax.scatter(
            lon[ii],
            lat[ii],
            s=marker_size,
            facecolors="0.2",
            edgecolors="white",
            linewidths=0.8,
            zorder=100,
            clip_on=False,
        )

    ax.set_xlabel("Longitude", fontsize=fontsize_label)
    if show_y:
        ax.set_ylabel("Latitude", fontsize=fontsize_label)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(axis="both", labelsize=fontsize_tick)
    if label:
        ax.text(
            0.98,
            0.02,
            label,
            transform=ax.transAxes,
            fontsize=info_fontsize,
            fontweight="bold",
            va="bottom",
            ha="right",
            zorder=101,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="lightgray",
                edgecolor="none",
                alpha=0.9,
            ),
        )
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_aspect("equal")


def _add_row_colorbars(
    fig: "Figure",
    axes_arr: np.ndarray,
    layout: list[int],
    panels_per_row: list[int],
    norm: Normalize,
    cmap_obj: Any,
    extend: str,
    cbar_label: str,
    vmin: float,
    vmax: float,
    log_scale: bool,
    fontsize_label: float,
    fontsize_tick: float,
) -> None:
    fig_w = fig.get_size_inches()[0]
    fig.tight_layout(rect=(0, 0, 1 - 1.0 / fig_w, 1))
    fig.canvas.draw()
    gap, width = 0.15 / fig_w, 0.18 / fig_w
    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    for row_idx in range(len(layout)):
        last = panels_per_row[row_idx] - 1
        if last < 0:
            continue
        pos = axes_arr[row_idx, last].get_position()
        cax = fig.add_axes((pos.x1 + gap, pos.y0, width, pos.height))
        cb = fig.colorbar(sm, cax=cax, extend=extend)
        cb.set_label(cbar_label, fontsize=fontsize_label)
        cb.ax.tick_params(labelsize=fontsize_tick)
        if log_scale:
            majors, mlabels, minors = log_colorbar_ticks(vmin, vmax)
            cb.set_ticks(majors)
            cb.set_ticklabels(mlabels)
            if minors:
                cb.ax.minorticks_on()
                cb.set_ticks(minors, minor=True)
        elif vmin == 0 and vmax == 100:
            cb.set_ticks([0, 20, 40, 60, 80, 100])
        elif vmin == 0 and vmax == 20:
            cb.set_ticks([0, 5, 10, 15, 20])
        elif vmin == 0 and vmax == 50:
            cb.set_ticks([0, 10, 20, 30, 40, 50])
