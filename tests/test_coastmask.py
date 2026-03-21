"""Tests for xfvcom.coastmask module using synthetic polygons."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from xfvcom.coastmask import CoastMask, CoastmaskConfig, load
from xfvcom.coastmask.cache import CoastmaskCache
from xfvcom.coastmask.config import PRESETS, BboxPreset
from xfvcom.coastmask.data import (
    _extract_polygons,
    compute_true_land,
)


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
@pytest.fixture
def simple_land() -> gpd.GeoDataFrame:
    """A square land polygon covering (0,0)-(10,10)."""
    return gpd.GeoDataFrame(
        geometry=[box(0, 0, 10, 10)],
        crs="EPSG:4326",
    )


@pytest.fixture
def simple_water() -> gpd.GeoDataFrame:
    """A small river polygon inside the land area."""
    return gpd.GeoDataFrame(
        geometry=[box(3, 3, 5, 5)],
        crs="EPSG:4326",
    )


@pytest.fixture
def empty_water() -> gpd.GeoDataFrame:
    """Empty water GeoDataFrame."""
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


@pytest.fixture
def mixed_water() -> gpd.GeoDataFrame:
    """Water GeoDataFrame with one small and one large polygon."""
    small_pond = box(1, 1, 1.01, 1.01)  # area ~1e-4 deg^2
    large_lake = box(6, 6, 8, 8)  # area = 4 deg^2
    return gpd.GeoDataFrame(
        geometry=[small_pond, large_lake],
        crs="EPSG:4326",
    )


# ------------------------------------------------------------------ #
# Config tests
# ------------------------------------------------------------------ #
class TestConfig:
    def test_presets_exist(self) -> None:
        assert "tokyo_bay" in PRESETS
        assert "ise_bay" in PRESETS
        assert len(PRESETS) >= 6

    def test_bbox_preset(self) -> None:
        tb = PRESETS["tokyo_bay"]
        assert isinstance(tb, BboxPreset)
        w, s, e, n = tb.bbox
        assert w < e
        assert s < n

    def test_config_defaults(self) -> None:
        cfg = CoastmaskConfig()
        assert cfg.land_shp_path is None
        assert cfg.water_shp_path is None
        assert cfg.bbox_margin == 0.05
        assert cfg.simplify_tolerance == 0.0
        assert cfg.subtract_water is True
        assert cfg.min_water_area_deg2 == 0.0
        assert cfg.cache_dir == Path.home() / ".coastmask"

    def test_config_expands_user(self) -> None:
        cfg = CoastmaskConfig(land_shp_path=Path("~/foo/bar.shp"))
        assert cfg.land_shp_path is not None
        assert "~" not in str(cfg.land_shp_path)


# ------------------------------------------------------------------ #
# Data processing tests
# ------------------------------------------------------------------ #
class TestComputeTrueLand:
    def test_land_minus_water(
        self, simple_land: gpd.GeoDataFrame, simple_water: gpd.GeoDataFrame
    ) -> None:
        bbox = (0.0, 0.0, 10.0, 10.0)
        result = compute_true_land(simple_land, simple_water, bbox)
        assert len(result) > 0

        # Total area should be land - water = 100 - 4 = 96
        total_area = result.geometry.area.sum()
        assert abs(total_area - 96.0) < 0.01

    def test_no_water(
        self, simple_land: gpd.GeoDataFrame, empty_water: gpd.GeoDataFrame
    ) -> None:
        bbox = (0.0, 0.0, 10.0, 10.0)
        result = compute_true_land(simple_land, empty_water, bbox)
        total_area = result.geometry.area.sum()
        assert abs(total_area - 100.0) < 0.01

    def test_bbox_clipping(
        self, simple_land: gpd.GeoDataFrame, empty_water: gpd.GeoDataFrame
    ) -> None:
        # Clip to smaller bbox: (0,0)-(5,5) → area = 25
        bbox = (0.0, 0.0, 5.0, 5.0)
        result = compute_true_land(simple_land, empty_water, bbox)
        total_area = result.geometry.area.sum()
        assert abs(total_area - 25.0) < 0.01

    def test_simplify(
        self, simple_land: gpd.GeoDataFrame, empty_water: gpd.GeoDataFrame
    ) -> None:
        bbox = (0.0, 0.0, 10.0, 10.0)
        result = compute_true_land(
            simple_land, empty_water, bbox, simplify_tolerance=0.5
        )
        assert len(result) > 0


class TestWaterFiltering:
    def test_min_area_filters_small_water(
        self,
        simple_land: gpd.GeoDataFrame,
        mixed_water: gpd.GeoDataFrame,
    ) -> None:
        """With high threshold, only large lake is subtracted."""
        bbox = (0.0, 0.0, 10.0, 10.0)
        # large_lake area = 4 deg^2, small_pond area ~1e-4 deg^2
        # Threshold 1.0 keeps only large_lake
        result = compute_true_land(simple_land, mixed_water, bbox)
        area_all_subtracted = result.geometry.area.sum()

        # Now filter: keep only water >= 1.0 deg^2
        big_only = mixed_water[mixed_water.geometry.area >= 1.0].copy()
        result_big = compute_true_land(simple_land, big_only, bbox)
        area_big_subtracted = result_big.geometry.area.sum()

        # Subtracting only big lake should leave more land than subtracting both
        assert area_big_subtracted > area_all_subtracted

    def test_subtract_water_false(
        self,
        simple_land: gpd.GeoDataFrame,
        mixed_water: gpd.GeoDataFrame,
    ) -> None:
        """subtract_water=False means no water subtracted at all."""
        bbox = (0.0, 0.0, 10.0, 10.0)
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        result = compute_true_land(simple_land, empty, bbox)
        total_area = result.geometry.area.sum()
        assert abs(total_area - 100.0) < 0.01


class TestExtractPolygons:
    def test_polygon(self) -> None:
        p = box(0, 0, 1, 1)
        assert len(_extract_polygons(p)) == 1

    def test_multipolygon(self) -> None:
        mp = MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])
        result = _extract_polygons(mp)
        assert len(result) == 1  # MultiPolygon returned as-is

    def test_empty(self) -> None:
        from shapely.geometry import Point

        empty = Point().buffer(0)
        assert len(_extract_polygons(empty)) == 0

    def test_geometry_collection(self) -> None:
        from shapely.geometry import GeometryCollection, LineString

        gc = GeometryCollection([box(0, 0, 1, 1), LineString([(0, 0), (1, 1)])])
        result = _extract_polygons(gc)
        assert len(result) == 1  # Only the polygon


# ------------------------------------------------------------------ #
# Cache tests
# ------------------------------------------------------------------ #
class TestCache:
    def test_save_and_load(
        self,
        tmp_path: Path,
        simple_land: gpd.GeoDataFrame,
        simple_water: gpd.GeoDataFrame,
    ) -> None:
        cache = CoastmaskCache(tmp_path, "test_region")
        bbox = (0.0, 0.0, 10.0, 10.0)

        assert not cache.exists()

        cache.save(simple_land, simple_land, simple_water, bbox)
        assert cache.exists()

        loaded = cache.load_land()
        assert len(loaded) == len(simple_land)
        assert abs(loaded.geometry.area.sum() - 100.0) < 0.01

    def test_metadata(
        self,
        tmp_path: Path,
        simple_land: gpd.GeoDataFrame,
        simple_water: gpd.GeoDataFrame,
    ) -> None:
        cache = CoastmaskCache(tmp_path, "test_region")
        bbox = (0.0, 0.0, 10.0, 10.0)

        cache.save(simple_land, simple_land, simple_water, bbox)

        meta = cache.load_metadata()
        assert meta["name"] == "test_region"
        assert meta["bbox"] == [0.0, 0.0, 10.0, 10.0]

    def test_empty_water_cache(
        self,
        tmp_path: Path,
        simple_land: gpd.GeoDataFrame,
        empty_water: gpd.GeoDataFrame,
    ) -> None:
        cache = CoastmaskCache(tmp_path, "no_water")
        bbox = (0.0, 0.0, 10.0, 10.0)

        cache.save(simple_land, simple_land, empty_water, bbox)
        assert cache.exists()

        water = cache.load_water()
        assert len(water) == 0


# ------------------------------------------------------------------ #
# CoastMask class tests
# ------------------------------------------------------------------ #
class TestCoastMask:
    def test_region_property(
        self, simple_land: gpd.GeoDataFrame, empty_water: gpd.GeoDataFrame
    ) -> None:
        mask = CoastMask(
            simple_land, simple_land, empty_water, (139.5, 34.9, 140.2, 35.7)
        )
        assert mask.region == [139.5, 140.2, 34.9, 35.7]

    def test_export_shapefile(
        self,
        tmp_path: Path,
        simple_land: gpd.GeoDataFrame,
        empty_water: gpd.GeoDataFrame,
    ) -> None:
        mask = CoastMask(simple_land, simple_land, empty_water, (0, 0, 10, 10))
        out = tmp_path / "export.shp"
        mask.to_shapefile(out)
        assert out.exists()

    def test_export_geopackage(
        self,
        tmp_path: Path,
        simple_land: gpd.GeoDataFrame,
        empty_water: gpd.GeoDataFrame,
    ) -> None:
        mask = CoastMask(simple_land, simple_land, empty_water, (0, 0, 10, 10))
        out = tmp_path / "export.gpkg"
        mask.to_geopackage(out)
        assert out.exists()


# ------------------------------------------------------------------ #
# load() integration tests
# ------------------------------------------------------------------ #
class TestLoad:
    def test_load_from_cache(
        self,
        tmp_path: Path,
        simple_land: gpd.GeoDataFrame,
        simple_water: gpd.GeoDataFrame,
    ) -> None:
        # Pre-populate cache for tokyo_bay
        cache = CoastmaskCache(tmp_path, "tokyo_bay")
        bbox = PRESETS["tokyo_bay"].bbox
        cache.save(simple_land, simple_land, simple_water, bbox)

        config = CoastmaskConfig(cache_dir=tmp_path)
        mask = load("tokyo_bay", config=config)

        assert isinstance(mask, CoastMask)
        assert mask.name == "tokyo_bay"
        assert mask.bbox == bbox

    def test_load_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preset"):
            load("nonexistent_bay")

    def test_load_no_shp_no_cache_raises(self, tmp_path: Path) -> None:
        config = CoastmaskConfig(cache_dir=tmp_path)
        with pytest.raises(ValueError, match="land_shp_path must be set"):
            load("tokyo_bay", config=config)

    def test_load_custom_bbox_from_cache(
        self,
        tmp_path: Path,
        simple_land: gpd.GeoDataFrame,
        empty_water: gpd.GeoDataFrame,
    ) -> None:
        custom_bbox = (1.0, 2.0, 3.0, 4.0)
        name = f"custom_{custom_bbox[0]}_{custom_bbox[1]}_{custom_bbox[2]}_{custom_bbox[3]}"
        cache = CoastmaskCache(tmp_path, name)
        cache.save(simple_land, simple_land, empty_water, custom_bbox)

        config = CoastmaskConfig(cache_dir=tmp_path)
        mask = load(custom_bbox, config=config)
        assert mask.bbox == custom_bbox
