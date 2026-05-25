"""Freshwater-supply node map driven by a river/sewer map YAML + label CSV.

This is the reusable, region-agnostic engine behind the TB-FVCOM
freshwater-node map.  Unlike the legacy :mod:`xfvcom.plot.source_map`
(whose ``SOURCE_NODES`` are hardcoded and go stale when the BC gains
new rivers/STPs), the source nodes here are derived from the
**authoritative river/sewer map YAML** (schema v4), so the plot always
matches the boundary condition that is actually built.

Two public entry points:

``source_nodes_from_river_map(yaml_path)``
    Parse a schema-v4 river/sewer map into
    ``{entity: {"nodes": [1-based ...], "kind": "River"|"Sewer"}}``.
    Several FVCOM outflows that share one logical source (e.g. four
    Arakawa mesh nodes) are grouped under their ``source.entity``.

``plot_freshwater_node_map(nc_file, entity_nodes, ...)``
    Draw a single-panel mesh + coastline with one marker per outflow
    node (coloured by River/Sewer, "new" entities ringed) and one
    label per entity.  Label text + position come from an editable CSV
    (``label_csv``); on first run the CSV is written with default
    positions (callers may supply a region-specific
    ``default_position_fn``; otherwise a generic radial fan is used),
    and existing human-edited rows are preserved on subsequent runs.

Grid + coastline helpers are reused from :mod:`xfvcom.plot.source_map`.
All figure text is ASCII-only.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon

from .source_map import (
    extract_boundary_edges,
    load_grid_coordinates,
    order_boundary_nodes,
)

CSV_FIELDS = [
    "entity",
    "label",
    "label_lon",
    "label_lat",
    "node_lon",
    "node_lat",
    "leader",
]

PositionFn = Callable[[dict], dict]


def source_nodes_from_river_map(
    yaml_path: str | Path, *, sewer_domain: str = "wastewater"
) -> dict[str, dict]:
    """Group a schema-v4 river/sewer map into per-entity node lists.

    Returns ``{entity: {"nodes": [1-based node ids], "kind": "River"|"Sewer"}}``.
    Classification: ``source.domain == sewer_domain`` (default
    ``"wastewater"``) or a ``kind: constant`` row -> ``"Sewer"``; otherwise
    ``"River"``.  The grouping key is ``source.entity`` (falling back to the
    FVCOM ``name`` for constant rows without a logical source).
    """
    with Path(yaml_path).open() as fh:
        doc = yaml.safe_load(fh)
    out: dict[str, dict] = {}
    for e in doc["rivers"]:
        node = int(e["node"])
        src = e.get("source", {}) or {}
        domain = src.get("domain")
        entity = src.get("entity", e["name"])
        if domain == sewer_domain or e.get("kind") == "constant":
            kind = "Sewer"
            if e.get("kind") == "constant":
                entity = e["name"]
        else:
            kind = "River"
        rec = out.setdefault(entity, {"nodes": [], "kind": kind})
        rec["nodes"].append(node)
        rec["kind"] = kind  # last wins; entities are type-consistent
    return out


def _build_info(coords: dict, entity_nodes: dict, new_entities) -> dict:
    lon, lat = coords["lon"], coords["lat"]
    info = {}
    for ent, d in entity_nodes.items():
        idx0 = [n - 1 for n in d["nodes"]]
        info[ent] = dict(
            clon=float(np.mean([lon[i] for i in idx0])),
            clat=float(np.mean([lat[i] for i in idx0])),
            kind=d["kind"],
            is_new=ent in set(new_entities),
            idx0=idx0,
        )
    return info


def _generic_default_positions(info: dict) -> dict:
    """Region-agnostic default: push each label radially out from the
    centroid of all sources by ~8% of the domain span."""
    clons = [d["clon"] for d in info.values()]
    clats = [d["clat"] for d in info.values()]
    bc_lon, bc_lat = float(np.mean(clons)), float(np.mean(clats))
    span = max(max(clons) - min(clons), max(clats) - min(clats), 1e-3)
    off = 0.10 * span
    pos = {}
    for e, d in info.items():
        dlon, dlat = d["clon"] - bc_lon, d["clat"] - bc_lat
        norm = max((dlon**2 + dlat**2) ** 0.5, 1e-6)
        pos[e] = (d["clon"] + off * dlon / norm, d["clat"] + off * dlat / norm)
    return pos


def load_or_init_label_csv(
    csv_path: str | Path,
    info: dict,
    new_entities,
    default_position_fn: PositionFn | None = None,
) -> dict:
    """Load label text/positions from *csv_path*, initialising defaults.

    Existing rows are preserved verbatim (human edits win); entities in
    *info* but missing from the CSV are appended with default positions.
    The CSV is (re)written only when missing or extended, so manual edits
    are never clobbered.  Columns: see :data:`CSV_FIELDS`.
    """
    csv_path = Path(csv_path)
    existing: dict[str, dict] = {}
    if csv_path.exists():
        with csv_path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                existing[r["entity"]] = r
    defaults = (default_position_fn or _generic_default_positions)(info)
    out, dirty = {}, not csv_path.exists()
    for e, d in info.items():
        if e in existing:
            r = existing[e]
            out[e] = dict(
                label=r.get("label", e),
                label_lon=float(r["label_lon"]),
                label_lat=float(r["label_lat"]),
                node_lon=float(r.get("node_lon") or d["clon"]),
                node_lat=float(r.get("node_lat") or d["clat"]),
                leader=int(float(r.get("leader", 1))),
            )
        else:
            llon, llat = defaults[e]
            out[e] = dict(
                label=e + (" *" if d["is_new"] else ""),
                label_lon=round(float(llon), 4),
                label_lat=round(float(llat), 4),
                node_lon=round(d["clon"], 4),
                node_lat=round(d["clat"], 4),
                leader=1,
            )
            dirty = True
    if dirty:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            for e in sorted(out):
                w.writerow({"entity": e, **out[e]})
    return out


def plot_freshwater_node_map(
    nc_file: str | Path,
    entity_nodes: dict,
    *,
    label_csv: str | Path | None = None,
    new_entities=frozenset(),
    out: str | Path | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (12, 13),
    river_color: str = "#1f5fbf",
    sewer_color: str = "#cc2b2b",
    ocean_color: str = "#eaf2f8",
    land_color: str = "#f0ead6",
    default_position_fn: PositionFn | None = None,
    dpi: int = 300,
):
    """Plot a single-panel freshwater-supply node map.

    Parameters
    ----------
    nc_file : path
        Any FVCOM NetCDF carrying ``lon``/``lat``/``nv`` (a static-mesh
        output file is fine) -- used for node coords + coastline.
    entity_nodes : dict
        ``{entity: {"nodes": [1-based], "kind": "River"|"Sewer"}}``, e.g.
        from :func:`source_nodes_from_river_map`.
    label_csv : path, optional
        Editable label CSV (text + position).  Created with defaults if
        absent; existing rows preserved.  If ``None``, defaults are used
        in-memory (no file written).
    new_entities : iterable
        Entity names to highlight (bold + black-ringed markers + ``*``).
    default_position_fn : callable, optional
        ``fn(info) -> {entity: (lon, lat)}`` for default label positions,
        where ``info[entity]`` has ``clon``/``clat``/``kind``/``is_new``.
        Defaults to a generic radial fan.

    Returns
    -------
    dict with ``fig``, ``ax``, ``info``, ``labels``.
    """
    coords = load_grid_coordinates(nc_file)
    lon, lat, nv = coords["lon"], coords["lat"], coords["nv"]
    paths = sorted(
        order_boundary_nodes(extract_boundary_edges(nv)), key=len, reverse=True
    )
    info = _build_info(coords, entity_nodes, new_entities)

    if label_csv is not None:
        labels = load_or_init_label_csv(
            label_csv, info, new_entities, default_position_fn
        )
    else:
        dpos = (default_position_fn or _generic_default_positions)(info)
        labels = {
            e: dict(
                label=e + (" *" if info[e]["is_new"] else ""),
                label_lon=dpos[e][0],
                label_lat=dpos[e][1],
                node_lon=info[e]["clon"],
                node_lat=info[e]["clat"],
                leader=1,
            )
            for e in info
        }

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(ocean_color)
    if paths:
        ext = paths[0]
        ax.add_patch(
            MplPolygon(
                np.column_stack([lon[ext], lat[ext]]),
                closed=True,
                facecolor=land_color,
                edgecolor="none",
                zorder=0,
            )
        )
    for p in paths:
        ax.plot(lon[p], lat[p], color="#555555", lw=0.8, zorder=1)

    for ent, d in info.items():
        mk, col = ("^", river_color) if d["kind"] == "River" else ("s", sewer_color)
        for i in d["idx0"]:
            ax.scatter(
                lon[i],
                lat[i],
                marker=mk,
                s=70,
                c=col,
                edgecolors=("black" if d["is_new"] else "white"),
                linewidths=(1.6 if d["is_new"] else 0.5),
                zorder=4,
            )

    for ent, d in info.items():
        L = labels[ent]
        col = river_color if d["kind"] == "River" else sewer_color
        kw = dict(
            fontsize=8.5,
            fontweight=("bold" if d["is_new"] else "normal"),
            color=col,
            ha="center",
            va="center",
            zorder=5,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
        )
        if int(L["leader"]):
            ax.annotate(
                L["label"],
                xy=(L["node_lon"], L["node_lat"]),
                xytext=(L["label_lon"], L["label_lat"]),
                arrowprops=dict(
                    arrowstyle="-", color=col, lw=0.6, alpha=0.85, shrinkA=1, shrinkB=3
                ),
                **kw,
            )
        else:
            ax.text(L["label_lon"], L["label_lat"], L["label"], **kw)

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    mid_lat = float(np.mean(ax.get_ylim()))
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mid_lat)))
    ax.tick_params(rotation=0)
    if title:
        ax.set_title(title, fontsize=11)

    handles = [
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor=river_color,
            markeredgecolor="white",
            markersize=11,
            label="River mouth",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=sewer_color,
            markeredgecolor="white",
            markersize=11,
            label="STP (sewer) outfall",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.6,
            markersize=12,
            label="Newly added",
        ),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return dict(fig=fig, ax=ax, info=info, labels=labels)
