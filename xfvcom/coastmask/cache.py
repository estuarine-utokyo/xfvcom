"""GeoPackage cache management for processed coastmask data.

Stores computed land/water GeoDataFrames to avoid repeated OSM downloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd


class CoastmaskCache:
    """Cache manager for processed coastmask GeoPackage files.

    Cache layout::

        {cache_dir}/{name}/
            land.gpkg       # true land (land - water)
            land.shp        # shapefile export for PyGMT
            raw_land.gpkg   # original land polygons
            water.gpkg      # inland water polygons
            metadata.json   # bbox, source paths, timestamps

    Parameters
    ----------
    cache_dir : Path
        Root cache directory (default: ~/.coastmask/).
    name : str
        Cache name (e.g. "tokyo_bay").
    """

    def __init__(self, cache_dir: Path, name: str) -> None:
        self.root = Path(cache_dir) / name
        self.name = name

    @property
    def land_gpkg(self) -> Path:
        return self.root / "land.gpkg"

    @property
    def land_shp(self) -> Path:
        return self.root / "land.shp"

    @property
    def raw_land_gpkg(self) -> Path:
        return self.root / "raw_land.gpkg"

    @property
    def water_gpkg(self) -> Path:
        return self.root / "water.gpkg"

    @property
    def metadata_json(self) -> Path:
        return self.root / "metadata.json"

    def exists(self) -> bool:
        """Return True if a valid cache exists."""
        return self.land_gpkg.exists() and self.metadata_json.exists()

    def save(
        self,
        land_gdf: gpd.GeoDataFrame,
        raw_land_gdf: gpd.GeoDataFrame,
        water_gdf: gpd.GeoDataFrame,
        bbox: tuple[float, float, float, float],
    ) -> None:
        """Save all layers and metadata to cache.

        Parameters
        ----------
        land_gdf : GeoDataFrame
            True land (land - water).
        raw_land_gdf : GeoDataFrame
            Original coastline-derived land polygons.
        water_gdf : GeoDataFrame
            Inland water polygons.
        bbox : tuple
            (west, south, east, north).
        """
        self.root.mkdir(parents=True, exist_ok=True)

        land_gdf.to_file(self.land_gpkg, driver="GPKG")
        raw_land_gdf.to_file(self.raw_land_gpkg, driver="GPKG")

        if len(water_gdf) > 0:
            water_gdf.to_file(self.water_gpkg, driver="GPKG")
        else:
            # Write empty GeoDataFrame
            empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            empty.to_file(self.water_gpkg, driver="GPKG")

        # Shapefile export for PyGMT compatibility
        land_gdf.to_file(self.land_shp, driver="ESRI Shapefile")

        metadata: dict[str, Any] = {
            "name": self.name,
            "bbox": list(bbox),
            "n_land_polygons": len(land_gdf),
            "n_raw_land_polygons": len(raw_land_gdf),
            "n_water_polygons": len(water_gdf),
        }
        self.metadata_json.write_text(json.dumps(metadata, indent=2))

    def load_land(self) -> gpd.GeoDataFrame:
        """Load cached true-land GeoDataFrame."""
        return gpd.read_file(self.land_gpkg)

    def load_raw_land(self) -> gpd.GeoDataFrame:
        """Load cached raw land GeoDataFrame."""
        return gpd.read_file(self.raw_land_gpkg)

    def load_water(self) -> gpd.GeoDataFrame:
        """Load cached water GeoDataFrame."""
        return gpd.read_file(self.water_gpkg)

    def load_metadata(self) -> dict[str, Any]:
        """Load cached metadata."""
        return json.loads(self.metadata_json.read_text())  # type: ignore[no-any-return]
