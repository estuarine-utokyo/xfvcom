"""Coastmask: OSM-derived land masking for coastal plots.

Public API
----------
load(name_or_bbox, *, config)
    Load (or create + cache) a CoastMask for a named region or custom bbox.

CoastMask
    Container holding true-land, raw-land, and water GeoDataFrames with
    rendering helpers for Matplotlib/Cartopy.

CoastmaskConfig
    Global configuration (shapefile paths, cache directory, tolerances).

PRESETS
    Built-in bbox presets for Japanese coastal areas (Tokyo Bay, Osaka Bay, etc.).
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd

from .cache import CoastmaskCache
from .cleanup import cleanup_land, fill_small_holes, remove_orphan_islands
from .config import PRESETS, BboxPreset, CoastmaskConfig
from .data import (
    compute_true_land,
    load_land_polygons,
    load_water_polygons,
    load_water_polygons_auto,
)
from .render import add_land_to_axes, add_land_to_plain_axes

__all__ = [
    "CoastMask",
    "CoastmaskConfig",
    "BboxPreset",
    "PRESETS",
    "load",
    "add_land_to_axes",
    "add_land_to_plain_axes",
    "cleanup_land",
    "fill_small_holes",
    "remove_orphan_islands",
]


class CoastMask:
    """Container for processed coastmask data.

    Parameters
    ----------
    land_gdf : GeoDataFrame
        True land (land minus inland water).
    raw_land_gdf : GeoDataFrame
        Original coastline-derived land polygons.
    water_gdf : GeoDataFrame
        Inland water body polygons.
    bbox : tuple
        (west, south, east, north) bounding box.
    name : str
        Region name.
    """

    def __init__(
        self,
        land_gdf: gpd.GeoDataFrame,
        raw_land_gdf: gpd.GeoDataFrame,
        water_gdf: gpd.GeoDataFrame,
        bbox: tuple[float, float, float, float],
        name: str = "",
    ) -> None:
        self.land_gdf = land_gdf
        self.raw_land_gdf = raw_land_gdf
        self.water_gdf = water_gdf
        self.bbox = bbox
        self.name = name

    @property
    def region(self) -> list[float]:
        """GMT-style region [west, east, south, north]."""
        w, s, e, n = self.bbox
        return [w, e, s, n]

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def add_to_mpl(
        self,
        ax: Any,
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
        facecolor, edgecolor, linewidth, zorder
            Styling parameters.
        **kwargs
            Extra arguments forwarded to ``ax.add_geometries``.
        """
        add_land_to_axes(
            ax,
            self.land_gdf,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
            **kwargs,
        )

    def add_to_plain_mpl(
        self,
        ax: Any,
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
            Standard matplotlib Axes.
        facecolor, edgecolor, linewidth, zorder
            Styling parameters.
        **kwargs
            Extra arguments forwarded to ``geopandas.plot``.
        """
        add_land_to_plain_axes(
            ax,
            self.land_gdf,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
            **kwargs,
        )

    def mpl_post_process(self, ax: Any, **_: Any) -> None:
        """Compatible with xfvcom's ``post_process_func`` signature.

        Usage::

            fig = plotter.plot_2d(
                ...,
                post_process_func=mask.mpl_post_process,
            )
        """
        self.add_to_mpl(ax)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def to_shapefile(self, path: str | Any) -> None:
        """Export true-land polygons to ESRI Shapefile."""
        self.land_gdf.to_file(str(path), driver="ESRI Shapefile")

    def to_geopackage(self, path: str | Any) -> None:
        """Export true-land polygons to GeoPackage."""
        self.land_gdf.to_file(str(path), driver="GPKG")


def load(
    name_or_bbox: str | tuple[float, float, float, float],
    *,
    config: CoastmaskConfig | None = None,
    force: bool = False,
) -> CoastMask:
    """Load a CoastMask for the given region.

    Checks the cache first; if not cached (or *force* is True), reads
    both shapefiles, computes true land, and saves to cache.

    Parameters
    ----------
    name_or_bbox : str or tuple
        Either a preset name (e.g. ``"tokyo_bay"``) or a custom bbox
        tuple ``(west, south, east, north)``.
    config : CoastmaskConfig or None
        Configuration with shapefile paths.  If None, a default config
        is used (which requires cached data to exist).
    force : bool
        If True, re-process even if cache exists.

    Returns
    -------
    CoastMask
        Loaded or freshly computed coastmask.
    """
    if config is None:
        config = CoastmaskConfig()

    # Resolve name to bbox
    if isinstance(name_or_bbox, str):
        name = name_or_bbox
        if name not in PRESETS:
            raise ValueError(
                f"Unknown preset '{name}'. " f"Available: {sorted(PRESETS.keys())}"
            )
        bbox = PRESETS[name].bbox
    else:
        bbox = name_or_bbox
        name = f"custom_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"

    # Build cache key that reflects water-filtering options
    cache_name = name
    parts: list[str] = []
    if config.water_source != "geofabrik":
        parts.append(config.water_source)
    if not config.subtract_water:
        parts.append("nolake")
    if not config.subtract_river:
        parts.append("noriver")
    if config.min_water_area_deg2 > 0:
        parts.append(f"minarea{config.min_water_area_deg2:.0e}")
    if config.fill_small_holes:
        parts.append(f"fillhole{config.min_hole_area_deg2:.0e}")
    if config.remove_orphan_islands:
        parts.append(f"noisland{config.min_island_area_deg2:.0e}")
    if parts:
        cache_name += "_" + "_".join(parts)

    cache = CoastmaskCache(config.cache_dir, cache_name)

    # Return cached if available
    if cache.exists() and not force:
        return CoastMask(
            land_gdf=cache.load_land(),
            raw_land_gdf=cache.load_raw_land(),
            water_gdf=cache.load_water(),
            bbox=bbox,
            name=name,
        )

    # Need shapefile paths for fresh processing (Geofabrik mode)
    if config.water_source == "geofabrik" and config.land_shp_path is None:
        raise ValueError(
            "land_shp_path must be set in CoastmaskConfig "
            "when no cache exists for this region."
        )
    # Overpass mode still needs land_shp_path for the coastline
    if config.water_source == "overpass" and config.land_shp_path is None:
        raise ValueError(
            "land_shp_path must be set in CoastmaskConfig "
            "when no cache exists for this region."
        )

    # Load raw data
    raw_land_gdf = load_land_polygons(bbox, config)

    # Load water polygons (dispatch by water_source)
    if config.water_source == "overpass":
        water_gdf = load_water_polygons_auto(bbox, config)
    else:
        need_water = (
            config.subtract_water or config.subtract_river
        ) and config.water_shp_path is not None
        if need_water:
            water_gdf = load_water_polygons_auto(bbox, config)
        else:
            water_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Compute true land (= raw land minus water bodies)
    land_gdf = compute_true_land(
        raw_land_gdf,
        water_gdf,
        bbox,
        simplify_tolerance=config.simplify_tolerance,
    )

    # Post-processing cleanup
    if config.fill_small_holes or config.remove_orphan_islands:
        # Determine column names based on water source
        class_col = "water_type" if config.water_source == "overpass" else "fclass"
        land_gdf = cleanup_land(
            land_gdf,
            water_gdf,
            do_fill_holes=config.fill_small_holes,
            min_hole_area_deg2=config.min_hole_area_deg2,
            do_remove_islands=config.remove_orphan_islands,
            min_island_area_deg2=config.min_island_area_deg2,
            protected_classes=config.protected_water_classes,
            protected_names=config.protected_water_names,
            class_column=class_col,
            name_column="name",
        )

    # Save to cache
    cache.save(
        land_gdf,
        raw_land_gdf,
        water_gdf,
        bbox,
        water_source=config.water_source,
    )

    return CoastMask(
        land_gdf=land_gdf,
        raw_land_gdf=raw_land_gdf,
        water_gdf=water_gdf,
        bbox=bbox,
        name=name,
    )
