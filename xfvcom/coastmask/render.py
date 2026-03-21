"""Matplotlib / Cartopy rendering helpers for coastmask.

Provides add_land_to_axes() for GeoAxes and add_land_to_plain_axes() for plain Axes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import geopandas as gpd
    from matplotlib.axes import Axes


def add_land_to_axes(
    ax: Axes,
    land_gdf: gpd.GeoDataFrame,
    *,
    facecolor: str = "lightgray",
    edgecolor: str = "black",
    linewidth: float = 0.5,
    zorder: int = 5,
    **kwargs: Any,
) -> None:
    """Add land mask to a Cartopy GeoAxes.

    Parameters
    ----------
    ax : GeoAxes
        Cartopy-aware matplotlib Axes.
    land_gdf : GeoDataFrame
        True-land polygons in EPSG:4326.
    facecolor : str
        Fill color for land.
    edgecolor : str
        Edge color for coastline.
    linewidth : float
        Edge line width.
    zorder : int
        Drawing order.
    **kwargs
        Extra keyword arguments passed to ``ax.add_geometries``.
    """
    import cartopy.crs as ccrs

    ax.add_geometries(
        land_gdf.geometry,
        crs=ccrs.PlateCarree(),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )


def add_land_to_plain_axes(
    ax: Axes,
    land_gdf: gpd.GeoDataFrame,
    *,
    facecolor: str = "lightgray",
    edgecolor: str = "black",
    linewidth: float = 0.5,
    zorder: int = 5,
    **kwargs: Any,
) -> None:
    """Add land mask to a plain (non-Cartopy) matplotlib Axes.

    Parameters
    ----------
    ax : Axes
        Standard matplotlib Axes (not GeoAxes).
    land_gdf : GeoDataFrame
        True-land polygons in EPSG:4326.
    facecolor : str
        Fill color for land.
    edgecolor : str
        Edge color for coastline.
    linewidth : float
        Edge line width.
    zorder : int
        Drawing order.
    **kwargs
        Extra keyword arguments passed to ``geopandas.plot``.
    """
    land_gdf.plot(
        ax=ax,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )
