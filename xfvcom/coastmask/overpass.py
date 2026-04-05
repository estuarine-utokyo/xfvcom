"""Overpass API client for fetching OSM water polygons.

Fetches ``natural=water`` + ``water=*`` tagged polygons and converts
them to a GeoDataFrame.  Supports live API calls and offline GeoJSON.

No external HTTP libraries required — uses :mod:`urllib.request`.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import linemerge, polygonize
from shapely.validation import make_valid

if TYPE_CHECKING:
    from .config import CoastmaskConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Query building
# ------------------------------------------------------------------ #
def build_overpass_query(
    bbox: tuple[float, float, float, float],
    water_types: tuple[str, ...],
    timeout: int = 60,
) -> str:
    """Build an Overpass QL query for water polygons.

    Parameters
    ----------
    bbox : tuple
        (west, south, east, north) in EPSG:4326.
    water_types : tuple of str
        OSM ``water=*`` tag values to fetch (e.g. ``("river", "canal")``).
    timeout : int
        Overpass query timeout in seconds.

    Returns
    -------
    str
        Overpass QL query string.
    """
    w, s, e, n = bbox
    types_regex = "|".join(water_types)
    return (
        f"[out:json][timeout:{timeout}][bbox:{s},{w},{n},{e}];\n"
        f"(\n"
        f'  way["natural"="water"]["water"~"^({types_regex})$"];\n'
        f'  relation["natural"="water"]["water"~"^({types_regex})$"];\n'
        f");\n"
        f"out geom;"
    )


# ------------------------------------------------------------------ #
# API communication
# ------------------------------------------------------------------ #
def fetch_overpass(
    query: str,
    endpoint: str = "https://overpass-api.de/api/interpreter",
) -> dict[str, Any]:
    """Execute an Overpass QL query and return parsed JSON.

    Parameters
    ----------
    query : str
        Overpass QL query string.
    endpoint : str
        Overpass API endpoint URL.

    Returns
    -------
    dict
        Parsed Overpass JSON response.

    Raises
    ------
    urllib.error.HTTPError
        On HTTP errors from the Overpass API.
    """
    data = urllib.request.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, method="POST")
    req.add_header("User-Agent", "xfvcom-coastmask/1.0")

    logger.info("Fetching from Overpass API: %s", endpoint)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ------------------------------------------------------------------ #
# Save / Load
# ------------------------------------------------------------------ #
def save_overpass_geojson(gdf: gpd.GeoDataFrame, path: str | Path) -> None:
    """Save water polygons GeoDataFrame as GeoJSON for offline use.

    Parameters
    ----------
    gdf : GeoDataFrame
        Output from :func:`overpass_response_to_gdf`.
    path : str or Path
        Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(path), driver="GeoJSON")
    logger.info("Saved Overpass GeoJSON to %s (%d features)", path, len(gdf))


def load_overpass_geojson(path: str | Path) -> gpd.GeoDataFrame:
    """Load water polygons from a saved GeoJSON file.

    Parameters
    ----------
    path : str or Path
        Path to GeoJSON file.

    Returns
    -------
    GeoDataFrame
        Water polygons with ``water_type``, ``name``, ``osm_id`` columns.
    """
    gdf: gpd.GeoDataFrame = gpd.read_file(str(path))
    return gdf


# ------------------------------------------------------------------ #
# Geometry conversion
# ------------------------------------------------------------------ #
def _coords_to_tuples(
    coords: list[dict[str, float]],
) -> list[tuple[float, float]]:
    """Convert ``[{"lat": ..., "lon": ...}, ...]`` to ``[(lon, lat), ...]``."""
    return [(c["lon"], c["lat"]) for c in coords]


def _way_to_polygon(element: dict[str, Any]) -> Polygon | None:
    """Convert an Overpass way element to a Polygon.

    Returns None if the way is not a closed ring or has too few points.
    """
    coords = _coords_to_tuples(element["geometry"])
    if len(coords) < 4:
        return None
    # Close ring if not already closed
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty or poly.geom_type not in ("Polygon", "MultiPolygon"):
            return None
        return poly if poly.geom_type == "Polygon" else poly
    except Exception:
        return None


def _relation_to_geometry(
    element: dict[str, Any],
) -> Polygon | MultiPolygon | None:
    """Convert an Overpass relation element to Polygon/MultiPolygon.

    Assembles outer and inner way members into closed rings using
    :func:`shapely.ops.linemerge` and :func:`shapely.ops.polygonize`.
    """
    outer_lines: list[LineString] = []
    inner_lines: list[LineString] = []

    for member in element.get("members", []):
        if member.get("type") != "way" or "geometry" not in member:
            continue
        coords = _coords_to_tuples(member["geometry"])
        if len(coords) < 2:
            continue
        line = LineString(coords)
        if member.get("role") == "inner":
            inner_lines.append(line)
        else:
            outer_lines.append(line)

    if not outer_lines:
        return None

    # Merge and polygonize outer rings
    merged_outer = linemerge(outer_lines)
    outer_polys = list(polygonize(merged_outer))
    if not outer_polys:
        return None

    # If no inners, return outer polygons directly
    if not inner_lines:
        if len(outer_polys) == 1:
            return outer_polys[0]
        return MultiPolygon(outer_polys)

    # Merge and polygonize inner rings
    merged_inner = linemerge(inner_lines)
    inner_polys = list(polygonize(merged_inner))

    # Subtract inner holes from their containing outer polygons
    result_polys: list[Polygon] = []
    for op in outer_polys:
        inner_rings = [ip.exterior for ip in inner_polys if op.contains(ip)]
        if inner_rings:
            result_polys.append(Polygon(op.exterior, inner_rings))
        else:
            result_polys.append(op)

    if len(result_polys) == 1:
        return result_polys[0]
    return MultiPolygon(result_polys)


def overpass_response_to_gdf(
    response: dict[str, Any],
) -> gpd.GeoDataFrame:
    """Convert an Overpass JSON response to a GeoDataFrame.

    Parameters
    ----------
    response : dict
        Parsed Overpass JSON (from :func:`fetch_overpass`).

    Returns
    -------
    GeoDataFrame
        Water polygons with columns ``water_type``, ``name``, ``osm_id``.
    """
    records: list[dict[str, Any]] = []

    for element in response.get("elements", []):
        tags = element.get("tags", {})
        osm_id = element.get("id", 0)
        water_type = tags.get("water", "")
        name = tags.get("name", "")

        if element["type"] == "way":
            geom = _way_to_polygon(element)
        elif element["type"] == "relation":
            geom = _relation_to_geometry(element)
        else:
            continue

        if geom is None or geom.is_empty:
            continue

        geom = make_valid(geom)
        records.append(
            {
                "osm_id": osm_id,
                "water_type": water_type,
                "name": name,
                "geometry": geom,
            }
        )

    if not records:
        return gpd.GeoDataFrame(
            {"osm_id": [], "water_type": [], "name": [], "geometry": []},
            crs="EPSG:4326",
        )

    return gpd.GeoDataFrame(records, crs="EPSG:4326")


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #
def load_water_polygons_overpass(
    bbox: tuple[float, float, float, float],
    config: CoastmaskConfig,
) -> gpd.GeoDataFrame:
    """Load water polygons using Overpass mode.

    If ``config.overpass_geojson_path`` is set, reads from the local
    file.  Otherwise, queries the Overpass API live.

    Parameters
    ----------
    bbox : tuple
        (west, south, east, north) in EPSG:4326.
    config : CoastmaskConfig
        Configuration with Overpass settings.

    Returns
    -------
    GeoDataFrame
        Water polygons with ``water_type``, ``name``, ``osm_id`` columns.
    """
    if config.overpass_geojson_path is not None:
        logger.info("Loading Overpass data from %s", config.overpass_geojson_path)
        return load_overpass_geojson(config.overpass_geojson_path)

    query = build_overpass_query(bbox, config.overpass_water_types, timeout=60)
    response = fetch_overpass(query, endpoint=config.overpass_endpoint)
    return overpass_response_to_gdf(response)
