"""Load shapefiles and compute true land (land minus inland water)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import shapely
from shapely.geometry import MultiPolygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid

if TYPE_CHECKING:
    from .config import CoastmaskConfig


def _expand_bbox(
    bbox: tuple[float, float, float, float], margin: float
) -> tuple[float, float, float, float]:
    """Expand bbox by *margin* degrees on each side."""
    w, s, e, n = bbox
    return (w - margin, s - margin, e + margin, n + margin)


def load_land_polygons(
    bbox: tuple[float, float, float, float],
    config: CoastmaskConfig,
) -> gpd.GeoDataFrame:
    """Read OSM land polygons within *bbox*.

    Parameters
    ----------
    bbox : tuple
        (west, south, east, north) in EPSG:4326.
    config : CoastmaskConfig
        Must have ``land_shp_path`` set.

    Returns
    -------
    GeoDataFrame
        Land polygons clipped to the buffered bbox.
    """
    if config.land_shp_path is None:
        raise ValueError("land_shp_path is not set in CoastmaskConfig")

    read_bbox = _expand_bbox(bbox, config.bbox_margin)
    gdf: gpd.GeoDataFrame = gpd.read_file(config.land_shp_path, bbox=read_bbox)
    return gdf


def load_water_polygons(
    bbox: tuple[float, float, float, float],
    config: CoastmaskConfig,
) -> gpd.GeoDataFrame:
    """Read Geofabrik inland water polygons within *bbox*.

    Filters to Polygon/MultiPolygon geometries only (the source file
    may contain Points and LineStrings).

    Parameters
    ----------
    bbox : tuple
        (west, south, east, north) in EPSG:4326.
    config : CoastmaskConfig
        Must have ``water_shp_path`` set.

    Returns
    -------
    GeoDataFrame
        Water polygons (Polygon/MultiPolygon only).
    """
    if config.water_shp_path is None:
        raise ValueError("water_shp_path is not set in CoastmaskConfig")

    read_bbox = _expand_bbox(bbox, config.bbox_margin)
    gdf: gpd.GeoDataFrame = gpd.read_file(config.water_shp_path, bbox=read_bbox)

    # Keep only polygon geometries
    mask = gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
    return gdf.loc[mask].copy()


def _extract_polygons(geom: shapely.Geometry) -> list[shapely.Geometry]:
    """Extract Polygon/MultiPolygon parts from any geometry."""
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return [geom]
    if geom.geom_type == "GeometryCollection":
        parts: list[shapely.Geometry] = []
        for g in geom.geoms:
            parts.extend(_extract_polygons(g))
        return parts
    return []


def compute_true_land(
    land_gdf: gpd.GeoDataFrame,
    water_gdf: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
    simplify_tolerance: float = 0.0,
) -> gpd.GeoDataFrame:
    """Compute true land = land polygons minus inland water bodies.

    Parameters
    ----------
    land_gdf : GeoDataFrame
        Raw land polygons (from ``load_land_polygons``).
    water_gdf : GeoDataFrame
        Inland water polygons (from ``load_water_polygons``).
    bbox : tuple
        (west, south, east, north) for final clipping.
    simplify_tolerance : float
        Simplify tolerance in degrees. 0 = no simplification.

    Returns
    -------
    GeoDataFrame
        True-land polygons (land minus water), clipped to *bbox*.
    """
    land_union = make_valid(unary_union(land_gdf.geometry))

    if len(water_gdf) > 0:
        water_union = make_valid(unary_union(water_gdf.geometry))
        true_land = land_union.difference(water_union)
    else:
        true_land = land_union

    true_land = make_valid(true_land)

    # Clip to bbox
    clip_box = box(*bbox)
    true_land = true_land.intersection(clip_box)
    true_land = make_valid(true_land)

    # Extract only polygonal geometries (difference can produce lines)
    polys = _extract_polygons(true_land)
    if not polys:
        result = MultiPolygon()
    elif len(polys) == 1 and polys[0].geom_type == "MultiPolygon":
        result = polys[0]
    else:
        # Flatten any MultiPolygons into individual Polygons
        flat: list[shapely.Geometry] = []
        for p in polys:
            if p.geom_type == "MultiPolygon":
                flat.extend(p.geoms)
            else:
                flat.append(p)
        result = MultiPolygon(flat)

    if simplify_tolerance > 0:
        result = result.simplify(simplify_tolerance, preserve_topology=True)

    gdf = gpd.GeoDataFrame(
        {"geometry": [result]},
        crs="EPSG:4326",
    )
    # Explode MultiPolygon into individual rows for efficient rendering
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    return gdf
