"""Generic, project-agnostic figure helpers.

Kept in xfvcom so per-project plotting scripts stay short:
  - ``fixed_stack``        : stacked panels with an IDENTICAL plot-box size
                             (x/y axis length in inches) so panels from
                             different figures line up when juxtaposed.
  - ``figsize_for_extent`` : figure size matching a lon/lat extent so a
                             cartopy GeoAxes fills it without letter-boxing
                             (ticks then sit on the axis line, not far from it).
  - ``geoaxes_decorate``   : axis ticks + Longitude/Latitude labels, NO
                             graticule, font size, and the map frame raised
                             ABOVE the filled land so the border stays visible.
  - ``place_labels``       : non-overlapping point labels CLIPPED to the axes
                             (never drawn outside the frame).

See the common panel spec in the TB-FVCOM memory ``feedback_plot_panel_spec``.
"""

from __future__ import annotations

import numpy as np


def fixed_stack(n_panels: int, geom: dict):
    """``n_panels`` stacked sub-axes, each a FIXED ``axes_w_in`` x ``axes_h_in``
    inch plot box with fixed margins. Save with ``savefig(dpi=...)`` and do NOT
    use ``tight_layout`` / ``bbox_inches='tight'`` (they break the fixed box)."""
    import matplotlib.pyplot as plt

    aw = float(geom.get("axes_w_in", 12.0))
    ah = float(geom.get("axes_h_in", 1.8))
    L = float(geom.get("left_in", 1.3))
    R = float(geom.get("right_in", 0.3))
    T = float(geom.get("top_in", 0.46))
    B = float(geom.get("bottom_in", 0.62))
    fig_w = L + aw + R
    slab = T + ah + B
    fig_h = max(slab, n_panels * slab)
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = []
    for i in range(n_panels):
        bottom = fig_h - (i + 1) * slab + B
        axes.append(fig.add_axes([L / fig_w, bottom / fig_h, aw / fig_w, ah / fig_h]))
    return fig, axes


def figsize_for_extent(extent, height_in: float = 10.0, max_w: float = 16.0):
    """Figure size whose aspect matches a (lon0, lon1, lat0, lat1) extent in a
    Mercator-like projection, so a GeoAxes fills the figure (no letter-box)."""
    lon0, lon1, lat0, lat1 = extent
    latm = 0.5 * (lat0 + lat1)
    w = (lon1 - lon0) * np.cos(np.radians(latm))
    h = lat1 - lat0
    aspect = (w / h) if h > 0 else 1.0
    width = min(max_w, height_in * aspect)
    return (width, width / aspect)


def geoaxes_decorate(ax, extent, data_crs, fs: int = 12, nbins: int = 5):
    """Axis ticks + Longitude/Latitude labels (NO graticule); font ``fs``; map
    frame raised ABOVE filled land (so the border is not hidden)."""
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LatitudeFormatter, LongitudeFormatter

    lon_t = [
        x
        for x in mticker.MaxNLocator(nbins).tick_values(extent[0], extent[1])
        if extent[0] <= x <= extent[1]
    ]
    lat_t = [
        y
        for y in mticker.MaxNLocator(nbins).tick_values(extent[2], extent[3])
        if extent[2] <= y <= extent[3]
    ]
    ax.set_xticks(lon_t, crs=data_crs)
    ax.set_yticks(lat_t, crs=data_crs)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis="both", labelsize=fs, pad=2.0)
    ax.set_xlabel("Longitude", fontsize=fs)
    ax.set_ylabel("Latitude", fontsize=fs)
    for sp in ax.spines.values():  # frame above land (zorder 4)
        sp.set_zorder(30)
        sp.set_linewidth(0.8)


def place_labels(
    ax,
    rows,
    get_label,
    data_crs,
    fontsize: int = 12,
    min_sep_px: float = 30.0,
    avoid_xy=None,
    leader: bool = True,
    overrides=None,
    pad_px: float = 3.0,
    seed_boxes=None,
    return_boxes: bool = False,
    marker_pad_px: float = 8.0,
):
    """Annotate ``rows`` (each a dict with 'lon','lat') with ``get_label(row)``
    so that label text boxes do not overlap one another or any marker point.

    For every row an estimated text bounding box (from line count + longest
    line at ``fontsize``) is tested against the already-placed boxes and the
    marker points; the first of many candidate offsets (8 compass directions
    x several radii) that is free AND inside the axes frame is used. When only
    a far candidate fits, a thin leader line connects the marker to the label
    (matplotlib annotate arrowprops) — the project convention for crowded
    maps. All labels are clipped to the axes so none draws outside the frame.

    Parameters
    ----------
    avoid_xy : optional iterable of (lon, lat)
        Extra marker points (e.g. MPOS stars) that labels must not cover, in
        addition to every row's own point.
    leader : bool
        Draw a leader line for far placements (radius beyond the near ring).
    overrides : optional dict {label_text: (dx_pt, dy_pt)}
        A PREFERRED first offset for a specific label (still validated; falls
        back to the automatic candidates if it would collide). Use for the
        few labels that need a hand-picked direction.
    seed_boxes : optional list of (x0, y0, x1, y1) display-pixel boxes
        Pre-occupied regions (e.g. labels drawn by a PRIOR place_labels call,
        such as MPOS station names at a larger font) that this call must avoid.
        Pass the previous call's ``return_boxes=True`` result here so two label
        layers with different fonts do not collide.
    return_boxes : bool
        If True, return the list of placed (x0, y0, x1, y1) boxes instead of
        the placed-count int (feed into a later call's ``seed_boxes``).
    marker_pad_px : float
        Minimum clear gap (display pixels) between a marker and its label:
        added to every candidate offset radius so the label never sits ON
        the marker. Use a larger value for big markers (e.g. ~14 for the
        MPOS stars, ~8 for small obs markers).
    """
    import numpy as np  # noqa: F401  (kept for parity with module imports)

    fig = ax.figure
    fig.canvas.draw()  # finalise transforms + extent
    p2x = fig.dpi / 72.0  # points -> pixels
    bbox = ax.get_window_extent()
    transform = data_crs._as_mpl_transform(ax)
    overrides = overrides or {}

    def to_px(lon, lat):
        return ax.transData.transform(
            ax.projection.transform_point(lon, lat, src_crs=data_crs)
        )

    avoid = [to_px(r["lon"], r["lat"]) for r in rows]
    if avoid_xy:
        for lon, lat in avoid_xy:
            avoid.append(to_px(lon, lat))

    def sgn(v):  # quantise a direction component
        return 1 if v > 0.35 else -1 if v < -0.35 else 0

    def box_for(cx, cy, sx, sy, w, h):
        if sx > 0:
            x0, x1 = cx, cx + w
        elif sx < 0:
            x0, x1 = cx - w, cx
        else:
            x0, x1 = cx - w / 2, cx + w / 2
        if sy > 0:
            y0, y1 = cy, cy + h
        elif sy < 0:
            y0, y1 = cy - h, cy
        else:
            y0, y1 = cy - h / 2, cy + h / 2
        return (x0, y0, x1, y1)

    def overlaps(b, others, pad):
        for o in others:
            if (
                b[0] - pad < o[2]
                and b[2] + pad > o[0]
                and b[1] - pad < o[3]
                and b[3] + pad > o[1]
            ):
                return True
        return False

    def hits_marker(b, pad):
        for px, py in avoid:
            if b[0] - pad < px < b[2] + pad and b[1] - pad < py < b[3] + pad:
                return True
        return False

    # 16 compass directions x several radii -> a dense candidate grid so that
    # even tight marker clusters get distinct, non-overlapping label slots
    # (labels that land far out are connected by a leader line below).
    import math

    units = [(math.cos(a), math.sin(a)) for a in [i * math.pi / 8.0 for i in range(16)]]
    # marker_pad_px shifts the whole candidate ring outward so the nearest
    # label edge always clears the marker by at least that gap.
    radii = [
        marker_pad_px + r
        for r in (6.0, 18.0, 32.0, 50.0, 72.0, 98.0, 128.0, 163.0, 203.0)
    ]
    placed = list(seed_boxes) if seed_boxes else []
    new_boxes = []
    n_ok = 0
    for r in rows:
        text = get_label(r)
        lines = text.split("\n")
        w = max((len(ln) for ln in lines), default=1) * fontsize * 0.60 * p2x
        h = len(lines) * fontsize * 1.30 * p2x
        x_px, y_px = to_px(r["lon"], r["lat"])

        cands = []
        if text in overrides:
            dx_pt, dy_pt = overrides[text]
            cands.append((dx_pt * p2x, dy_pt * p2x, sgn(dx_pt), sgn(dy_pt)))
        for rad in radii:
            for ux, uy in units:
                cands.append((ux * rad, uy * rad, sgn(ux), sgn(uy)))

        chosen = None
        for ox, oy, sx, sy in cands:
            b = box_for(x_px + ox, y_px + oy, sx, sy, w, h)
            inside = (
                b[0] > bbox.x0 + 2
                and b[2] < bbox.x1 - 2
                and b[1] > bbox.y0 + 2
                and b[3] < bbox.y1 - 2
            )
            if (
                inside
                and not overlaps(b, placed, pad_px)
                and not hits_marker(b, pad_px)
            ):
                chosen = (ox, oy, sx, sy, b)
                break
        if chosen is None:  # last resort: near-NE, accept it
            b = box_for(x_px + 8, y_px + 8, 1, 1, w, h)
            chosen = (8.0, 8.0, 1, 1, b)

        ox, oy, sx, sy, b = chosen
        placed.append(b)
        new_boxes.append(b)
        ha = "left" if sx > 0 else "right" if sx < 0 else "center"
        va = "bottom" if sy > 0 else "top" if sy < 0 else "center"
        arrow = {}
        if leader and (ox * ox + oy * oy) ** 0.5 > 24.0:
            arrow = dict(
                arrowprops=dict(
                    arrowstyle="-", lw=0.5, color="0.45", shrinkA=0, shrinkB=2
                )
            )
        ax.annotate(
            text,
            xy=(r["lon"], r["lat"]),
            xycoords=transform,
            xytext=(ox / p2x, oy / p2x),
            textcoords="offset points",
            fontsize=fontsize,
            fontweight="bold",
            color="black",
            ha=ha,
            va=va,
            zorder=12,
            annotation_clip=True,
            clip_on=True,
            **arrow,
        )
        n_ok += 1
    return new_boxes if return_boxes else n_ok
