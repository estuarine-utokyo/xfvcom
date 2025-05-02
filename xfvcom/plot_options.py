# xfvcom/plot_options.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cartopy.crs as ccrs
from cartopy.io.img_tiles import GoogleTiles


@dataclass
class FvcomPlotOptions:
    """
    One-stop option container for every plotting method in the xfvcom package.
    All fields have sensible defaults so you can override only what you need.
    """

    # ------------------------------------------------------------
    # 1. Color & scaling
    # ------------------------------------------------------------
    cmap: str = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    levels: int | list[float] = 20
    extend: str = "both"
    norm: Any | None = None  # Custom matplotlib.colors.Normalize

    # ------------------------------------------------------------
    # 2. Date / time axis
    # ------------------------------------------------------------
    date_fmt: str = "%Y-%m-%d"

    # ------------------------------------------------------------
    # 3. Figure / axis basics
    # ------------------------------------------------------------
    figsize: tuple[float, float] | None = None
    dpi: int | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    title: str | None = None

    # ------------------------------------------------------------
    # 4. Spatial extent & projection
    # ------------------------------------------------------------
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    use_latlon: bool = True
    projection: ccrs.Projection = field(default_factory=lambda: ccrs.Mercator())

    # ------------------------------------------------------------
    # 5. 2-D mesh / map specific
    # ------------------------------------------------------------
    with_mesh: bool = False
    coastlines: bool = False
    obclines: bool = False
    plot_grid: bool = False
    add_tiles: bool = False
    tile_provider: GoogleTiles = field(
        default_factory=lambda: GoogleTiles(style="satellite")
    )
    tile_zoom: int = 12
    mesh_linewidth: float = 0.5
    mesh_color: str = "#36454F"
    coastline_color: str = "gray"
    obcline_color: str = "blue"
    grid_linestyle: str = "--"
    grid_linewidth: float = 0.5

    # ------------------------------------------------------------
    # 6. Colorbar
    # ------------------------------------------------------------
    colorbar: bool = True
    cbar_label: str | None = None
    cbar_size: str | None = None
    cbar_pad: float | None = None
    cbar_kwargs: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------
    # 7. Rolling / smoothing (ts_ 系)
    # ------------------------------------------------------------
    rolling_window: int | None = None
    min_periods: int | None = None

    # ------------------------------------------------------------
    # 8. Vector (ts_vector) options
    # ------------------------------------------------------------
    arrow_width: float = 0.002
    arrow_color: str = "k"
    arrow_alpha: float = 0.7
    arrow_angles: str = "uv"
    arrow_headlength: int = 5
    arrow_headwidth: int = 3
    arrow_headaxislength: float = 4.5
    scale: float | str | None = None
    scale_units: str = "y"
    show_vec_legend: bool = True
    vec_legend_speed: float | None = None  # None → 0.3*max
    vec_legend_loc: tuple[float, float] = (0.75, 0.1)
    with_magnitude: bool = True
    skip: int | str | None = None  # sampling interval for quiver arrows
    vec_zorder: int = 2  # z-order between contour and annotations
    # ------------------------------------------------------------
    # 8-B. Vector-map specific parameters (added)
    # ------------------------------------------------------------
    plot_vec2d: bool = False  # call vector-map in plot_2d
    vec_time: int | slice | list[int] | tuple[int, ...] | None = None
    vec_siglay: int | slice | list[int] | tuple[int, ...] | None = None
    # e.g. {"time":"mean","siglay":"mean"|"thickness"}
    vec_reduce: dict[str, str] | None = None
    # vec_scale      : float | str = None          # custom quiver scale
    # ------------------------------------------------------------
    # 9. ts_contourf / ts_contourf_z
    # ------------------------------------------------------------
    add_contour: bool = False
    label_contours: bool = False
    plot_surface: bool = False
    surface_kwargs: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------
    # 10. Section plots
    # ------------------------------------------------------------
    spacing: float = 200.0
    land_color: str = "#A0522D"

    # ------------------------------------------------------------
    # 11. Miscellaneous
    # ------------------------------------------------------------
    verbose: bool = False
    log_scale: bool = False

    # ------------------------------------------------------------
    # 12. Future / rarely-used kwargs bucket
    # ------------------------------------------------------------
    extra: dict[str, Any] = field(default_factory=dict, repr=False)
    da_is_scalar: bool = False

    # ------------------------------------------------------------
    # Helper constructor to keep backward compatibility with **kwargs
    # ------------------------------------------------------------
    @classmethod
    def from_kwargs(cls, **kwargs) -> FvcomPlotOptions:
        """
        Convert legacy keyword arguments into FvcomPlotOptions,
        unknown fields are stored in `extra`.
        """
        field_names = {f.name for f in cls.__dataclass_fields__.values()}

        core: dict[str, Any] = {}
        for key in list(kwargs):
            if key in field_names:
                core[key] = kwargs.pop(key)
        return cls(**core, extra=kwargs)
