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
        axes.append(fig.add_axes([L / fig_w, bottom / fig_h,
                                  aw / fig_w, ah / fig_h]))
    return fig, axes


def figsize_for_extent(extent, height_in: float = 10.0, max_w: float = 16.0):
    """Figure size whose aspect matches a (lon0, lon1, lat0, lat1) extent in a
    Mercator-like projection, so a GeoAxes fills the figure (no letter-box)."""
    lon0, lon1, lat0, lat1 = extent
    latm = 0.5 * (lat0 + lat1)
    w = (lon1 - lon0) * np.cos(np.radians(latm))
    h = (lat1 - lat0)
    aspect = (w / h) if h > 0 else 1.0
    width = min(max_w, height_in * aspect)
    return (width, width / aspect)


def geoaxes_decorate(ax, extent, data_crs, fs: int = 12, nbins: int = 5):
    """Axis ticks + Longitude/Latitude labels (NO graticule); font ``fs``; map
    frame raised ABOVE filled land (so the border is not hidden)."""
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
    lon_t = [x for x in mticker.MaxNLocator(nbins).tick_values(extent[0], extent[1])
             if extent[0] <= x <= extent[1]]
    lat_t = [y for y in mticker.MaxNLocator(nbins).tick_values(extent[2], extent[3])
             if extent[2] <= y <= extent[3]]
    ax.set_xticks(lon_t, crs=data_crs)
    ax.set_yticks(lat_t, crs=data_crs)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis="both", labelsize=fs, pad=2.0)
    ax.set_xlabel("Longitude", fontsize=fs)
    ax.set_ylabel("Latitude", fontsize=fs)
    for sp in ax.spines.values():           # frame above land (zorder 4)
        sp.set_zorder(30)
        sp.set_linewidth(0.8)


def place_labels(ax, rows, get_label, data_crs, fontsize: int = 12,
                 min_sep_px: float = 30.0):
    """Annotate ``rows`` (each a dict with 'lon','lat') with ``get_label(row)``,
    trying 4 cardinal offsets to avoid overlap, biased to keep labels INSIDE
    the axes; all labels are clipped to the axes so none draws outside."""
    transform = data_crs._as_mpl_transform(ax)
    bbox = ax.get_window_extent()
    placed = []
    offsets = [(6, 6), (-6, 6), (6, -6), (-6, -6)]
    n_ok = 0
    for r in rows:
        x_px, y_px = ax.transData.transform(
            ax.projection.transform_point(r["lon"], r["lat"], src_crs=data_crs))
        chosen = None
        for ox, oy in offsets:
            cx, cy = x_px + ox * 6, y_px + oy * 6   # rough label centre
            inside = (bbox.x0 + 6 < cx < bbox.x1 - 60
                      and bbox.y0 + 6 < cy < bbox.y1 - 6)
            far = all((cx - px) ** 2 + (cy - py) ** 2 > min_sep_px ** 2
                      for px, py in placed)
            if inside and far:
                chosen = (ox, oy, cx, cy)
                break
        if chosen is None:
            ox, oy = offsets[0]
            chosen = (ox, oy, x_px + ox * 6, y_px + oy * 6)
        placed.append((chosen[2], chosen[3]))
        ax.annotate(get_label(r), xy=(r["lon"], r["lat"]), xycoords=transform,
                    xytext=(chosen[0], chosen[1]), textcoords="offset points",
                    fontsize=fontsize, fontweight="bold", color="black",
                    zorder=12, annotation_clip=True, clip_on=True)
        n_ok += 1
    return n_ok
