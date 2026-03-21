"""Configuration and bbox presets for coastmask.

Provides CoastmaskConfig for paths/tolerances and PRESETS for Japanese bays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CACHE_DIR = Path.home() / ".coastmask"


@dataclass
class BboxPreset:
    """A named bounding box with description."""

    bbox: tuple[float, float, float, float]  # (west, south, east, north)
    description: str = ""


# Built-in bbox presets for Japanese coastal areas
PRESETS: dict[str, BboxPreset] = {
    "tokyo_bay": BboxPreset(
        bbox=(139.5, 34.9, 140.2, 35.7),
        description="Tokyo Bay and surrounding area",
    ),
    "tokyo_bay_inner": BboxPreset(
        bbox=(139.7, 35.2, 140.0, 35.65),
        description="Inner Tokyo Bay (fishery areas)",
    ),
    "ise_bay": BboxPreset(
        bbox=(136.5, 34.5, 137.3, 35.2),
        description="Ise Bay (Mikawa Bay included)",
    ),
    "osaka_bay": BboxPreset(
        bbox=(134.8, 34.2, 135.5, 34.8),
        description="Osaka Bay",
    ),
    "seto_inland_sea": BboxPreset(
        bbox=(131.5, 33.0, 135.5, 34.8),
        description="Seto Inland Sea",
    ),
    "ariake_sea": BboxPreset(
        bbox=(129.8, 32.5, 130.8, 33.3),
        description="Ariake Sea",
    ),
}


@dataclass
class CoastmaskConfig:
    """Global configuration for coastmask processing.

    Parameters
    ----------
    land_shp_path : Path or None
        Path to OSM land polygon shapefile
        (land-polygons-split-4326/land_polygons.shp).
    water_shp_path : Path or None
        Path to Geofabrik water polygon shapefile
        (gis_osm_water_a_free_1.shp).
    cache_dir : Path
        Cache directory for processed GeoPackage files.
    bbox_margin : float
        Buffer margin (degrees) added to bbox when reading shapefiles.
    simplify_tolerance : float
        Simplify tolerance (degrees). 0 = no simplification.
        0.0001 ~ 10 m at mid-latitudes. Useful for faster rendering.
    subtract_water : bool
        If False, skip water subtraction entirely (land = raw coastline
        polygons). Useful when inland water bodies are not relevant.
        Note: even when False, rivers are still subtracted if
        ``subtract_river`` is True (the default).
    subtract_river : bool
        If True (default), always subtract river polygons
        (``fclass == "riverbank"`` in Geofabrik data) from land,
        regardless of ``subtract_water``.
    water_fclasses : tuple of str
        Geofabrik ``fclass`` values treated as "lake/pond" water
        (controlled by ``subtract_water``).  Default:
        ``("water", "reservoir", "wetland", "dock")``.
        Rivers (``"riverbank"``) are handled separately via
        ``subtract_river``.
    min_water_area_deg2 : float
        Minimum water-body area in degrees-squared to subtract from
        land. Water polygons smaller than this are ignored (treated as
        land). Set to 0 to subtract all water bodies.
        Applies to both rivers and lakes.

        Approximate area conversions at 35 deg N:

        - ``1e-7`` deg^2 ~ 1,000 m^2 (small pond)
        - ``1e-6`` deg^2 ~ 10,000 m^2 (large pond)
        - ``1e-5`` deg^2 ~ 100,000 m^2 (small lake)
        - ``1e-4`` deg^2 ~ 1 km^2 (lake)
    """

    land_shp_path: Path | None = None
    water_shp_path: Path | None = None
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    bbox_margin: float = 0.05
    simplify_tolerance: float = 0.0
    subtract_water: bool = True
    subtract_river: bool = True
    water_fclasses: tuple[str, ...] = ("water", "reservoir", "wetland", "dock")
    min_water_area_deg2: float = 0.0

    def __post_init__(self) -> None:
        if self.land_shp_path is not None:
            self.land_shp_path = Path(self.land_shp_path).expanduser()
        if self.water_shp_path is not None:
            self.water_shp_path = Path(self.water_shp_path).expanduser()
        self.cache_dir = Path(self.cache_dir).expanduser()
