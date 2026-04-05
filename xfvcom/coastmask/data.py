"""Load shapefiles and compute true land (land minus inland water).

Handles OSM land polygon loading, water subtraction, and geometry validation.
"""

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
    may contain Points and LineStrings).  Selects water body classes
    based on ``config.subtract_water`` and ``config.subtract_river``:

    - Rivers (``fclass == "riverbank"``): included when
      ``subtract_river`` is True.
    - Lakes/ponds (``fclass`` in ``water_fclasses``): included when
      ``subtract_water`` is True.

    Optionally filters out small water bodies below
    ``config.min_water_area_deg2``.

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
    type_mask = gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
    gdf = gdf.loc[type_mask].copy()

    # Filter by fclass: rivers vs lakes/ponds
    if "fclass" in gdf.columns:
        keep_classes: list[str] = []
        if config.subtract_river:
            keep_classes.append("riverbank")
        if config.subtract_water:
            keep_classes.extend(config.water_fclasses)
        if keep_classes:
            gdf = gdf.loc[gdf["fclass"].isin(keep_classes)].copy()
        else:
            gdf = gdf.iloc[:0].copy()  # empty

    # Filter by minimum area
    if config.min_water_area_deg2 > 0 and len(gdf) > 0:
        areas = gdf.geometry.area
        gdf = gdf.loc[areas >= config.min_water_area_deg2].copy()

    return gdf


def load_water_polygons_auto(
    bbox: tuple[float, float, float, float],
    config: CoastmaskConfig,
) -> gpd.GeoDataFrame:
    """Load water polygons, dispatching by ``config.water_source``.

    - ``"geofabrik"``: delegates to :func:`load_water_polygons`.
    - ``"overpass"``: delegates to
      :func:`~xfvcom.coastmask.overpass.load_water_polygons_overpass`,
      then filters by keep/fill types and ``min_river_area_deg2``.

    Parameters
    ----------
    bbox : tuple
        (west, south, east, north) in EPSG:4326.
    config : CoastmaskConfig
        Configuration with water source settings.

    Returns
    -------
    GeoDataFrame
        Water polygons to subtract from land.
    """
    if config.water_source == "geofabrik":
        return load_water_polygons(bbox, config)

    # Overpass mode
    from .overpass import load_water_polygons_overpass

    all_water = load_water_polygons_overpass(bbox, config)

    if len(all_water) == 0:
        return all_water

    # Keep only water types in overpass_keep_types
    keep_mask = all_water["water_type"].isin(config.overpass_keep_types)
    water_gdf = all_water.loc[keep_mask].copy()

    # Apply minimum river area filter
    if config.min_river_area_deg2 > 0 and len(water_gdf) > 0:
        areas = water_gdf.geometry.area
        water_gdf = water_gdf.loc[areas >= config.min_river_area_deg2].copy()

    return water_gdf


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
