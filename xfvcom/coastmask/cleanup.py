"""Post-processing cleanup for land mask polygons.

Fill small holes (enclosed water) and remove orphan islands that appear
after hole-filling.  Extracted from sediment/plot/contour.py for reuse.
"""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


def _fill_polygon_holes(
    geom: Polygon,
    min_hole_area: float,
    protected: Polygon | MultiPolygon | None,
) -> Polygon:
    """Remove interior rings smaller than *min_hole_area* from a Polygon.

    A hole is kept if its area >= *min_hole_area* **or** more than half
    of it overlaps *protected* water geometry.
    """
    if not geom.interiors:
        return geom

    kept_rings: list[Polygon] = []
    for ring in geom.interiors:
        hole = Polygon(ring)
        if hole.area >= min_hole_area:
            kept_rings.append(hole)
        elif (
            protected is not None
            and protected.intersection(hole).area > hole.area * 0.5
        ):
            kept_rings.append(hole)

    return Polygon(geom.exterior, [r.exterior for r in kept_rings])


def fill_small_holes(
    land_gdf: gpd.GeoDataFrame,
    water_gdf: gpd.GeoDataFrame,
    *,
    min_hole_area_deg2: float = 1e-3,
    protected_classes: tuple[str, ...] = ("riverbank", "dock"),
    protected_names: tuple[str, ...] = ("\u904b\u6cb3",),
    class_column: str = "fclass",
    name_column: str = "name",
) -> gpd.GeoDataFrame:
    """Fill small interior holes in land polygons.

    Parameters
    ----------
    land_gdf : GeoDataFrame
        True-land polygons (individual Polygon rows).
    water_gdf : GeoDataFrame
        Water body polygons with classification columns.
    min_hole_area_deg2 : float
        Holes smaller than this (degrees-squared) are filled.
    protected_classes : tuple of str
        Class values whose water bodies are never filled.
    protected_names : tuple of str
        Name substrings whose water bodies are never filled.
    class_column : str
        Column name for water body classification.
    name_column : str
        Column name for water body name.

    Returns
    -------
    GeoDataFrame
        Land polygons with small holes filled.
    """
    # Build protected water geometry
    protected = _build_protected_geometry(
        water_gdf,
        protected_classes=protected_classes,
        protected_names=protected_names,
        class_column=class_column,
        name_column=name_column,
    )

    filled_geoms: list[Polygon | MultiPolygon] = []
    for geom in land_gdf.geometry:
        if geom.geom_type == "Polygon":
            filled_geoms.append(
                _fill_polygon_holes(geom, min_hole_area_deg2, protected)
            )
        elif geom.geom_type == "MultiPolygon":
            parts = [
                _fill_polygon_holes(p, min_hole_area_deg2, protected)
                for p in geom.geoms
            ]
            filled_geoms.append(MultiPolygon(parts))
        else:
            filled_geoms.append(geom)

    result = land_gdf.copy()
    result["geometry"] = filled_geoms
    return result


def remove_orphan_islands(
    original_gdf: gpd.GeoDataFrame,
    filled_gdf: gpd.GeoDataFrame,
    *,
    min_island_area_deg2: float = 1e-3,
) -> gpd.GeoDataFrame:
    """Remove small islands that lie entirely within filled areas.

    When holes are filled, small land fragments (e.g. islands inside a
    former lake) can become orphans.  This removes polygons that are
    smaller than *min_island_area_deg2* **and** fully contained within
    the union of newly-filled regions.

    Parameters
    ----------
    original_gdf : GeoDataFrame
        Land polygons **before** hole-filling.
    filled_gdf : GeoDataFrame
        Land polygons **after** hole-filling.
    min_island_area_deg2 : float
        Islands smaller than this (degrees-squared) are removed.

    Returns
    -------
    GeoDataFrame
        *filled_gdf* with orphan islands removed.
    """
    # Compute the union of areas that became land (filled regions)
    diffs: list[Polygon | MultiPolygon] = []
    for new_geom, old_geom in zip(filled_gdf.geometry, original_gdf.geometry):
        diff = new_geom.difference(old_geom)
        if not diff.is_empty:
            diffs.append(diff)

    if not diffs:
        return filled_gdf

    filled_union = unary_union(diffs)

    # Remove small polygons fully contained in filled_union
    keep_mask: list[bool] = []
    for geom in filled_gdf.geometry:
        if geom.area < min_island_area_deg2 and filled_union.contains(geom):
            keep_mask.append(False)
        else:
            keep_mask.append(True)

    return filled_gdf.loc[keep_mask].reset_index(drop=True)


def _build_protected_geometry(
    water_gdf: gpd.GeoDataFrame,
    *,
    protected_classes: tuple[str, ...],
    protected_names: tuple[str, ...],
    class_column: str,
    name_column: str,
) -> Polygon | MultiPolygon | None:
    """Build a union of protected water geometries."""
    if len(water_gdf) == 0:
        return None

    masks = []

    if class_column in water_gdf.columns and protected_classes:
        masks.append(water_gdf[class_column].isin(protected_classes))

    if name_column in water_gdf.columns and protected_names:
        name_series = water_gdf[name_column].fillna("")
        name_mask = name_series.apply(
            lambda n: any(sub in n for sub in protected_names)
        )
        masks.append(name_mask)

    if not masks:
        return None

    import functools
    import operator

    combined = functools.reduce(operator.or_, masks)
    protected_gdf = water_gdf.loc[combined]

    if len(protected_gdf) == 0:
        return None

    return unary_union(protected_gdf.geometry)


def cleanup_land(
    land_gdf: gpd.GeoDataFrame,
    water_gdf: gpd.GeoDataFrame,
    *,
    do_fill_holes: bool = False,
    min_hole_area_deg2: float = 1e-3,
    do_remove_islands: bool = False,
    min_island_area_deg2: float = 1e-3,
    protected_classes: tuple[str, ...] = ("riverbank", "dock"),
    protected_names: tuple[str, ...] = ("\u904b\u6cb3",),
    class_column: str = "fclass",
    name_column: str = "name",
) -> gpd.GeoDataFrame:
    """Convenience wrapper: fill holes then remove orphan islands.

    Parameters
    ----------
    land_gdf : GeoDataFrame
        True-land polygons.
    water_gdf : GeoDataFrame
        Water body polygons.
    do_fill_holes : bool
        Whether to fill small holes.
    min_hole_area_deg2 : float
        Threshold for hole-filling.
    do_remove_islands : bool
        Whether to remove orphan islands.
    min_island_area_deg2 : float
        Threshold for island removal.
    protected_classes, protected_names : tuple of str
        Water bodies to protect from filling.
    class_column, name_column : str
        Column names for classification and name.

    Returns
    -------
    GeoDataFrame
        Cleaned land polygons.
    """
    if not do_fill_holes and not do_remove_islands:
        return land_gdf

    original = land_gdf

    if do_fill_holes:
        land_gdf = fill_small_holes(
            land_gdf,
            water_gdf,
            min_hole_area_deg2=min_hole_area_deg2,
            protected_classes=protected_classes,
            protected_names=protected_names,
            class_column=class_column,
            name_column=name_column,
        )

    if do_remove_islands:
        land_gdf = remove_orphan_islands(
            original,
            land_gdf,
            min_island_area_deg2=min_island_area_deg2,
        )

    return land_gdf
