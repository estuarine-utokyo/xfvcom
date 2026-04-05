"""Tests for xfvcom.coastmask.overpass module.

All tests use fixture JSON — no network calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from xfvcom.coastmask.overpass import (
    build_overpass_query,
    load_overpass_geojson,
    overpass_response_to_gdf,
    save_overpass_geojson,
)


# ------------------------------------------------------------------ #
# Fixture: minimal Overpass JSON response
# ------------------------------------------------------------------ #
@pytest.fixture
def overpass_way_response() -> dict:
    """Overpass response with a single closed way (canal)."""
    return {
        "elements": [
            {
                "type": "way",
                "id": 100001,
                "tags": {
                    "natural": "water",
                    "water": "canal",
                    "name": "品川運河",
                },
                "geometry": [
                    {"lat": 35.0, "lon": 139.0},
                    {"lat": 35.0, "lon": 139.1},
                    {"lat": 35.1, "lon": 139.1},
                    {"lat": 35.1, "lon": 139.0},
                    {"lat": 35.0, "lon": 139.0},
                ],
            }
        ]
    }


@pytest.fixture
def overpass_relation_response() -> dict:
    """Overpass response with a relation (river with an island hole)."""
    return {
        "elements": [
            {
                "type": "relation",
                "id": 200001,
                "tags": {
                    "natural": "water",
                    "water": "river",
                    "name": "多摩川",
                },
                "members": [
                    {
                        "type": "way",
                        "ref": 300001,
                        "role": "outer",
                        "geometry": [
                            {"lat": 35.0, "lon": 139.0},
                            {"lat": 35.0, "lon": 139.2},
                            {"lat": 35.2, "lon": 139.2},
                            {"lat": 35.2, "lon": 139.0},
                            {"lat": 35.0, "lon": 139.0},
                        ],
                    },
                    {
                        "type": "way",
                        "ref": 300002,
                        "role": "inner",
                        "geometry": [
                            {"lat": 35.05, "lon": 139.05},
                            {"lat": 35.05, "lon": 139.1},
                            {"lat": 35.1, "lon": 139.1},
                            {"lat": 35.1, "lon": 139.05},
                            {"lat": 35.05, "lon": 139.05},
                        ],
                    },
                ],
            }
        ]
    }


@pytest.fixture
def overpass_multi_way_relation_response() -> dict:
    """Overpass response with a relation whose outer is split into 2 ways."""
    return {
        "elements": [
            {
                "type": "relation",
                "id": 200002,
                "tags": {"natural": "water", "water": "lake", "name": "諏訪湖"},
                "members": [
                    {
                        "type": "way",
                        "ref": 400001,
                        "role": "outer",
                        "geometry": [
                            {"lat": 36.0, "lon": 138.0},
                            {"lat": 36.0, "lon": 138.1},
                            {"lat": 36.1, "lon": 138.1},
                        ],
                    },
                    {
                        "type": "way",
                        "ref": 400002,
                        "role": "outer",
                        "geometry": [
                            {"lat": 36.1, "lon": 138.1},
                            {"lat": 36.1, "lon": 138.0},
                            {"lat": 36.0, "lon": 138.0},
                        ],
                    },
                ],
            }
        ]
    }


@pytest.fixture
def overpass_mixed_response(
    overpass_way_response: dict,
    overpass_relation_response: dict,
) -> dict:
    """Combined response with both a way and a relation."""
    return {
        "elements": (
            overpass_way_response["elements"] + overpass_relation_response["elements"]
        )
    }


# ------------------------------------------------------------------ #
# build_overpass_query
# ------------------------------------------------------------------ #
class TestBuildOverpassQuery:
    def test_basic_query(self) -> None:
        bbox = (139.5, 34.9, 140.2, 35.7)
        query = build_overpass_query(bbox, ("river", "canal"))
        assert "[out:json]" in query
        assert "[timeout:60]" in query
        assert "[bbox:34.9,139.5,35.7,140.2]" in query
        assert '"water"~"^(river|canal)$"' in query
        assert "way" in query
        assert "relation" in query
        assert "out geom;" in query

    def test_single_type(self) -> None:
        query = build_overpass_query((0, 0, 1, 1), ("reservoir",))
        assert '"water"~"^(reservoir)$"' in query

    def test_all_default_types(self) -> None:
        types = ("river", "reservoir", "lake", "pond", "canal", "basin")
        query = build_overpass_query((0, 0, 1, 1), types)
        assert "river|reservoir|lake|pond|canal|basin" in query


# ------------------------------------------------------------------ #
# overpass_response_to_gdf
# ------------------------------------------------------------------ #
class TestOverpassResponseToGdf:
    def test_way_to_polygon(self, overpass_way_response: dict) -> None:
        gdf = overpass_response_to_gdf(overpass_way_response)
        assert len(gdf) == 1
        assert gdf.iloc[0]["water_type"] == "canal"
        assert gdf.iloc[0]["name"] == "品川運河"
        assert gdf.iloc[0]["osm_id"] == 100001
        assert gdf.geometry.iloc[0].geom_type == "Polygon"
        assert gdf.crs.to_epsg() == 4326

    def test_relation_to_polygon_with_hole(
        self, overpass_relation_response: dict
    ) -> None:
        gdf = overpass_response_to_gdf(overpass_relation_response)
        assert len(gdf) == 1
        assert gdf.iloc[0]["water_type"] == "river"
        assert gdf.iloc[0]["name"] == "多摩川"

        geom = gdf.geometry.iloc[0]
        assert geom.geom_type == "Polygon"
        # Should have 1 interior ring (the island)
        assert len(geom.interiors) == 1

    def test_multi_way_relation(
        self, overpass_multi_way_relation_response: dict
    ) -> None:
        """Two outer ways are merged into one closed polygon."""
        gdf = overpass_response_to_gdf(overpass_multi_way_relation_response)
        assert len(gdf) == 1
        assert gdf.iloc[0]["water_type"] == "lake"
        assert gdf.geometry.iloc[0].geom_type == "Polygon"
        assert gdf.geometry.iloc[0].area > 0

    def test_mixed_elements(self, overpass_mixed_response: dict) -> None:
        gdf = overpass_response_to_gdf(overpass_mixed_response)
        assert len(gdf) == 2
        assert set(gdf["water_type"]) == {"canal", "river"}

    def test_empty_response(self) -> None:
        gdf = overpass_response_to_gdf({"elements": []})
        assert len(gdf) == 0
        assert "water_type" in gdf.columns
        assert "name" in gdf.columns
        assert "osm_id" in gdf.columns

    def test_skips_unclosed_way(self) -> None:
        """A way that is not a closed ring is skipped."""
        response = {
            "elements": [
                {
                    "type": "way",
                    "id": 999,
                    "tags": {"natural": "water", "water": "river"},
                    "geometry": [
                        {"lat": 0, "lon": 0},
                        {"lat": 0, "lon": 1},
                        {"lat": 1, "lon": 1},
                    ],
                }
            ]
        }
        gdf = overpass_response_to_gdf(response)
        # 3 points → auto-closed to 4 → not enough for a valid polygon
        assert len(gdf) == 0

    def test_missing_tags_uses_defaults(self) -> None:
        """Elements without name/water tags get empty strings."""
        response = {
            "elements": [
                {
                    "type": "way",
                    "id": 888,
                    "tags": {"natural": "water"},
                    "geometry": [
                        {"lat": 0, "lon": 0},
                        {"lat": 0, "lon": 1},
                        {"lat": 1, "lon": 1},
                        {"lat": 1, "lon": 0},
                        {"lat": 0, "lon": 0},
                    ],
                }
            ]
        }
        gdf = overpass_response_to_gdf(response)
        assert len(gdf) == 1
        assert gdf.iloc[0]["water_type"] == ""
        assert gdf.iloc[0]["name"] == ""


# ------------------------------------------------------------------ #
# Save / Load GeoJSON
# ------------------------------------------------------------------ #
class TestSaveLoadGeojson:
    def test_round_trip(self, tmp_path: Path, overpass_mixed_response: dict) -> None:
        """Save and reload preserves geometry and attributes."""
        gdf = overpass_response_to_gdf(overpass_mixed_response)
        out_path = tmp_path / "water.geojson"

        save_overpass_geojson(gdf, out_path)
        assert out_path.exists()

        loaded = load_overpass_geojson(out_path)
        assert len(loaded) == len(gdf)
        assert set(loaded["water_type"]) == set(gdf["water_type"])
        assert set(loaded["name"]) == set(gdf["name"])

        # Geometry areas should match closely
        for orig, reloaded in zip(gdf.geometry, loaded.geometry):
            assert abs(orig.area - reloaded.area) < 1e-10

    def test_empty_gdf_round_trip(self, tmp_path: Path) -> None:
        gdf = overpass_response_to_gdf({"elements": []})
        out_path = tmp_path / "empty.geojson"
        save_overpass_geojson(gdf, out_path)

        loaded = load_overpass_geojson(out_path)
        assert len(loaded) == 0
