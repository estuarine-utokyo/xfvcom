"""Tests for xfvcom.coastmask.cleanup module.

Uses synthetic polygons to verify hole-filling and orphan island removal.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from xfvcom.coastmask.cleanup import (
    cleanup_land,
    fill_small_holes,
    remove_orphan_islands,
)


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
@pytest.fixture
def land_with_holes() -> gpd.GeoDataFrame:
    """Land polygon (0,0)-(10,10) with two holes inside.

    - Small hole at (2,2)-(2.01,2.01): area ~1e-4 deg^2
    - Large hole at (5,5)-(8,8):       area = 9 deg^2
    """
    exterior = box(0, 0, 10, 10).exterior.coords
    small_hole = Polygon([(2, 2), (2.01, 2), (2.01, 2.01), (2, 2.01)])
    large_hole = Polygon([(5, 5), (8, 5), (8, 8), (5, 8)])
    poly = Polygon(exterior, [small_hole.exterior.coords, large_hole.exterior.coords])
    gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
    return gdf.explode(index_parts=False).reset_index(drop=True)


@pytest.fixture
def water_with_classes() -> gpd.GeoDataFrame:
    """Water polygons with fclass and name columns."""
    return gpd.GeoDataFrame(
        {
            "fclass": ["riverbank", "water", "dock"],
            "name": ["多摩川", "池", "品川運河"],
            "geometry": [
                box(2, 2, 2.01, 2.01),  # overlaps small hole
                box(5, 5, 8, 8),  # overlaps large hole
                box(9, 9, 9.5, 9.5),  # not in any hole
            ],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def empty_water() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


# ------------------------------------------------------------------ #
# fill_small_holes tests
# ------------------------------------------------------------------ #
class TestFillSmallHoles:
    def test_fills_small_hole_only(
        self,
        land_with_holes: gpd.GeoDataFrame,
        empty_water: gpd.GeoDataFrame,
    ) -> None:
        """Small hole is filled, large hole is kept."""
        result = fill_small_holes(land_with_holes, empty_water, min_hole_area_deg2=1.0)
        # The land polygon should now have only the large hole
        geom = result.geometry.union_all()
        if geom.geom_type == "Polygon":
            n_holes = len(geom.interiors)
        else:
            n_holes = sum(len(p.interiors) for p in geom.geoms)
        assert n_holes == 1

    def test_no_fill_when_threshold_zero(
        self,
        land_with_holes: gpd.GeoDataFrame,
        empty_water: gpd.GeoDataFrame,
    ) -> None:
        """With threshold 0, no holes are filled."""
        result = fill_small_holes(land_with_holes, empty_water, min_hole_area_deg2=0.0)
        geom = result.geometry.union_all()
        if geom.geom_type == "Polygon":
            n_holes = len(geom.interiors)
        else:
            n_holes = sum(len(p.interiors) for p in geom.geoms)
        assert n_holes == 2

    def test_fills_all_when_threshold_large(
        self,
        land_with_holes: gpd.GeoDataFrame,
        empty_water: gpd.GeoDataFrame,
    ) -> None:
        """With large threshold, all holes are filled."""
        result = fill_small_holes(
            land_with_holes, empty_water, min_hole_area_deg2=100.0
        )
        geom = result.geometry.union_all()
        if geom.geom_type == "Polygon":
            n_holes = len(geom.interiors)
        else:
            n_holes = sum(len(p.interiors) for p in geom.geoms)
        assert n_holes == 0

    def test_protected_class_keeps_hole(
        self,
        land_with_holes: gpd.GeoDataFrame,
        water_with_classes: gpd.GeoDataFrame,
    ) -> None:
        """Hole overlapping protected riverbank water is kept."""
        # small_hole overlaps riverbank water → protected
        result = fill_small_holes(
            land_with_holes,
            water_with_classes,
            min_hole_area_deg2=1.0,  # would fill small hole
            protected_classes=("riverbank",),
            protected_names=(),
        )
        geom = result.geometry.union_all()
        if geom.geom_type == "Polygon":
            n_holes = len(geom.interiors)
        else:
            n_holes = sum(len(p.interiors) for p in geom.geoms)
        # Both holes should remain: small protected, large above threshold
        assert n_holes == 2

    def test_protected_name_keeps_hole(
        self,
        land_with_holes: gpd.GeoDataFrame,
    ) -> None:
        """Hole overlapping water with protected name substring is kept."""
        water = gpd.GeoDataFrame(
            {
                "fclass": ["water"],
                "name": ["東京運河"],
                "geometry": [box(2, 2, 2.01, 2.01)],
            },
            crs="EPSG:4326",
        )
        result = fill_small_holes(
            land_with_holes,
            water,
            min_hole_area_deg2=1.0,
            protected_classes=(),
            protected_names=("運河",),
        )
        geom = result.geometry.union_all()
        if geom.geom_type == "Polygon":
            n_holes = len(geom.interiors)
        else:
            n_holes = sum(len(p.interiors) for p in geom.geoms)
        assert n_holes == 2

    def test_no_holes_is_noop(self, empty_water: gpd.GeoDataFrame) -> None:
        """Land with no holes passes through unchanged."""
        land = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        result = fill_small_holes(land, empty_water, min_hole_area_deg2=1.0)
        assert result.geometry.area.sum() == pytest.approx(1.0)


# ------------------------------------------------------------------ #
# remove_orphan_islands tests
# ------------------------------------------------------------------ #
class TestRemoveOrphanIslands:
    def test_removes_small_island_in_filled_area(self) -> None:
        """A tiny island inside a filled region is removed."""
        # Original: large land with a big hole, plus a tiny island in the hole
        exterior = box(0, 0, 10, 10).exterior.coords
        hole = Polygon([(3, 3), (7, 3), (7, 7), (3, 7)])
        land_with_hole = Polygon(exterior, [hole.exterior.coords])
        tiny_island = box(4.5, 4.5, 4.6, 4.6)  # area = 0.01

        original = gpd.GeoDataFrame(
            geometry=[land_with_hole, tiny_island], crs="EPSG:4326"
        )

        # Filled: hole is gone, island is still there
        filled = gpd.GeoDataFrame(
            geometry=[box(0, 0, 10, 10), tiny_island], crs="EPSG:4326"
        )

        result = remove_orphan_islands(original, filled, min_island_area_deg2=0.1)
        # tiny_island (area=0.01 < 0.1) inside filled region → removed
        assert len(result) == 1

    def test_keeps_large_island(self) -> None:
        """An island larger than the threshold is kept."""
        exterior = box(0, 0, 10, 10).exterior.coords
        hole = Polygon([(3, 3), (7, 3), (7, 7), (3, 7)])
        land_with_hole = Polygon(exterior, [hole.exterior.coords])
        large_island = box(4, 4, 6, 6)  # area = 4

        original = gpd.GeoDataFrame(
            geometry=[land_with_hole, large_island], crs="EPSG:4326"
        )
        filled = gpd.GeoDataFrame(
            geometry=[box(0, 0, 10, 10), large_island], crs="EPSG:4326"
        )

        result = remove_orphan_islands(original, filled, min_island_area_deg2=0.1)
        assert len(result) == 2

    def test_no_change_when_no_filling(self) -> None:
        """When nothing was filled, all polygons are kept."""
        land = gpd.GeoDataFrame(
            geometry=[box(0, 0, 5, 5), box(6, 6, 7, 7)], crs="EPSG:4326"
        )
        result = remove_orphan_islands(land, land, min_island_area_deg2=0.1)
        assert len(result) == 2


# ------------------------------------------------------------------ #
# cleanup_land wrapper tests
# ------------------------------------------------------------------ #
class TestCleanupLand:
    def test_noop_when_both_false(
        self,
        land_with_holes: gpd.GeoDataFrame,
        empty_water: gpd.GeoDataFrame,
    ) -> None:
        result = cleanup_land(
            land_with_holes,
            empty_water,
            do_fill_holes=False,
            do_remove_islands=False,
        )
        assert result is land_with_holes

    def test_fill_only(
        self,
        land_with_holes: gpd.GeoDataFrame,
        empty_water: gpd.GeoDataFrame,
    ) -> None:
        result = cleanup_land(
            land_with_holes,
            empty_water,
            do_fill_holes=True,
            min_hole_area_deg2=100.0,
            do_remove_islands=False,
        )
        geom = result.geometry.union_all()
        if geom.geom_type == "Polygon":
            n_holes = len(geom.interiors)
        else:
            n_holes = sum(len(p.interiors) for p in geom.geoms)
        assert n_holes == 0

    def test_fill_and_remove(self) -> None:
        """Full pipeline: fill holes then remove orphan islands."""
        exterior = box(0, 0, 10, 10).exterior.coords
        hole = Polygon([(3, 3), (7, 3), (7, 7), (3, 7)])
        land_main = Polygon(exterior, [hole.exterior.coords])
        tiny_island = box(4.5, 4.5, 4.6, 4.6)

        land = gpd.GeoDataFrame(geometry=[land_main, tiny_island], crs="EPSG:4326")
        water = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        result = cleanup_land(
            land,
            water,
            do_fill_holes=True,
            min_hole_area_deg2=100.0,
            do_remove_islands=True,
            min_island_area_deg2=0.1,
        )
        # Hole filled → tiny island now inside filled region → removed
        assert len(result) == 1
        # Remaining polygon should be ~100 deg^2 (solid square)
        assert result.geometry.area.sum() == pytest.approx(100.0, abs=0.01)
