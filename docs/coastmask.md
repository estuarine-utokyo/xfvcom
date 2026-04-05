# Coastmask: OSM-Derived Land Masking for Coastal and Ocean Models

[Back to README](../README.md)

## Overview

The `xfvcom.coastmask` module creates land/water mask polygons from OpenStreetMap (OSM) data for coastal and estuarine modeling applications. It provides accurate, modern coastline representations — including reclaimed land, harbors, and canals — that are missing from traditional datasets like GSHHG.

```python
from xfvcom.coastmask import load, CoastmaskConfig

mask = load("tokyo_bay")
mask.add_to_axes(ax)  # Cartopy GeoAxes
```

---

## OSM Data Architecture

The land mask is constructed from two complementary OSM datasets:

### 1. Land Polygons (`land_polygons.shp`)

Source: [osmdata.openstreetmap.de](https://osmdata.openstreetmap.de/data/land-polygons.html)

Generated from `natural=coastline` ways by the [osmcoastline](https://github.com/osmcode/osmcoastline) tool. This dataset defines the **land/ocean boundary only** — inland water bodies are NOT represented as holes. All land area (including lakes, rivers, ponds) is solid.

### 2. Inland Water Polygons (Geofabrik extract)

Source: [Geofabrik Shapefiles](https://www.geofabrik.de/data/shapefiles.html)

The file `gis_osm_water_a_free_1.shp` contains inland water bodies as polygons. Geofabrik classifies them using a `fclass` column:

| `fclass` | Current OSM tags | Description |
|----------|-----------------|-------------|
| `riverbank` | `natural=water` + `water=river` | River water surface (polygon). **Deprecated tag**; modern OSM uses `water=river` |
| `reservoir` | `natural=water` + `water=reservoir` | Dammed lakes, reservoirs. **Deprecated tag**; modern OSM uses `water=reservoir` |
| `water` | `natural=water` + `water=lake\|pond\|canal\|basin\|...` | **Catch-all** for lakes, ponds, canals, basins, etc. Cannot distinguish subtypes from Geofabrik data alone |
| `wetland` | `natural=wetland` | Marshes, swamps |
| `dock` | `waterway=dock` | Harbor docks, port basins |

**Note on Geofabrik limitations**: The `fclass=water` category mixes lakes, ponds, and canals. Canals (`water=canal`) cannot be distinguished from lakes (`water=lake`) without the original `water=*` tag, which Geofabrik does not export.

### 3. Waterways (LineString only — not used for masking)

The file `gis_osm_waterways_free_1.shp` contains rivers, streams, canals, and drains as **LineStrings**. These cannot be used directly for area masking because they have no width information. The `fclass` values are: `stream`, `river`, `canal`, `drain`.

---

## Processing Pipeline

```
land_polygons.shp          gis_osm_water_a_free_1.shp
       │                            │
       ▼                            ▼
 load_land_polygons()        load_water_polygons()
       │                            │
       │    ┌───── filter by fclass (riverbank, water, ...)
       │    │      filter by area (min_water_area_deg2)
       │    │
       ▼    ▼
 compute_true_land()
   land_union - water_union → clip to bbox → explode
       │
       ▼
 cleanup_land()  [optional]
   fill_small_holes() → remove_orphan_islands()
       │
       ▼
 CoastMask object
   .land_gdf        (true land polygons)
   .raw_land_gdf    (original coastline)
   .water_gdf       (subtracted water bodies)
```

---

## Hole-Filling and Cleanup

When `fill_small_holes=True`, the module removes small interior holes (ponds, reservoirs, small lakes) from the land polygons. This is essential for clean map rendering and ocean model grid generation.

### How It Works

1. **Hole-size threshold** (`min_hole_area_deg2`, default `1e-3 deg² ≈ 10 km²`):
   - Holes **smaller** than this threshold are filled (become land)
   - Holes **larger** than this threshold are preserved (remain water)
   - This naturally distinguishes major river systems (Tama, Arakawa: > 10 km² when merged) from small inland features (ponds, small streams: < 10 km²)

2. **Protected water features** (`protected_water_classes`, `protected_water_names`):
   - Holes that overlap > 50% with protected water polygons are preserved regardless of size
   - Default protected classes: `("riverbank", "dock")`
   - Default protected names: `("運河",)` (substring match on the `name` column)

3. **Orphan island removal** (`remove_orphan_islands=True`):
   - When a hole is filled, small land polygons that were islands within that hole become redundant
   - These orphan islands (entirely contained within filled areas) are removed to prevent rendering artifacts caused by geometric gaps and anti-aliasing

### Choosing the Right Threshold

The `min_hole_area_deg2` threshold controls the tradeoff between map cleanliness and water feature detail:

| Threshold | Approx. area | Effect |
|-----------|-------------|--------|
| `1e-6` | 10,000 m² (1 ha) | Fills only tiny ponds |
| `1e-5` | 100,000 m² | Fills small ponds and narrow streams |
| `1e-4` | 1 km² | Fills most inland lakes and reservoirs |
| `1e-3` | 10 km² | Fills all but major river systems. **Recommended for ocean modeling** |

At `1e-3`, the union of water features along a major river (e.g., Tama River at 6.75 km² as a single polygon, but > 10 km² when merged with adjacent water bodies) forms a hole large enough to be preserved, while isolated inland features are filled.

---

## Configuration Reference

```python
config = CoastmaskConfig(
    # === Data sources ===
    land_shp_path=Path("land_polygons.shp"),      # OSM land polygons
    water_shp_path=Path("water_a_free_1.shp"),     # Geofabrik water polygons

    # === Water subtraction ===
    subtract_water=True,         # Subtract lakes/ponds/docks
    subtract_river=True,         # Subtract rivers (fclass="riverbank")
    water_fclasses=("water", "reservoir", "wetland", "dock"),
    min_water_area_deg2=0.0,     # Min water body area to subtract (0 = all)

    # === Cleanup ===
    fill_small_holes=True,       # Fill small interior holes as land
    min_hole_area_deg2=1e-3,     # Holes < this are filled (~10 km²)
    remove_orphan_islands=True,  # Remove artifacts from hole-filling
    min_island_area_deg2=1e-3,   # Island size threshold for removal
    protected_water_classes=("riverbank", "dock"),  # Protect these from filling
    protected_water_names=("運河",),                # Protect by name match

    # === Geometry ===
    bbox_margin=0.05,            # Buffer (degrees) when reading shapefiles
    simplify_tolerance=0.0,      # Simplify tolerance (0 = no simplification)

    # === Cache ===
    cache_dir=Path("~/.coastmask"),
)
```

### Preset Regions

| Name | Bbox (W, S, E, N) | Description |
|------|-------------------|-------------|
| `tokyo_bay` | 139.5, 34.9, 140.2, 35.7 | Tokyo Bay |
| `tokyo_bay_inner` | 139.7, 35.2, 140.0, 35.65 | Inner Tokyo Bay |
| `ise_bay` | 136.5, 34.5, 137.3, 35.2 | Ise Bay |
| `osaka_bay` | 134.8, 34.2, 135.5, 34.8 | Osaka Bay |
| `seto_inland_sea` | 131.5, 33.0, 135.5, 34.8 | Seto Inland Sea |
| `ariake_sea` | 129.8, 32.5, 130.8, 33.3 | Ariake Sea |

---

## Usage Examples

### Basic Usage

```python
from xfvcom.coastmask import load

# Load with built-in preset
mask = load("tokyo_bay")

# Load with custom bbox (west, south, east, north)
mask = load((135.0, 34.0, 136.0, 35.0))
```

### Clean Map for Ocean Modeling

```python
from xfvcom.coastmask import load, CoastmaskConfig

config = CoastmaskConfig(
    land_shp_path=data_dir / "land_polygons.shp",
    water_shp_path=data_dir / "gis_osm_water_a_free_1.shp",
    subtract_water=True,
    subtract_river=True,
    fill_small_holes=True,
    min_hole_area_deg2=1e-3,
    remove_orphan_islands=True,
    protected_water_classes=("dock",),
    protected_water_names=("運河",),
)
mask = load("tokyo_bay", config=config)
```

This configuration:
- Subtracts all water features from land (rivers, lakes, docks, etc.)
- Fills small holes (< 10 km²) — removes inland ponds, reservoirs, small streams
- Preserves major river systems — merged water areas > 10 km² remain as water
- Protects docks and named canals (e.g., 京浜運河) from filling
- Removes orphan island artifacts

### Rendering

```python
# Cartopy GeoAxes
mask.add_to_axes(ax, facecolor="lightgray", edgecolor="k", linewidth=0.5)

# Plain matplotlib Axes
mask.add_to_plain_axes(ax)

# Export for PyGMT
mask.to_shapefile("land.shp")
```

---

## Caching

Processed coastmask data is cached in GeoPackage format under `~/.coastmask/`:

```
~/.coastmask/
└── tokyo_bay/
    ├── land.gpkg         # True land polygons (after cleanup)
    ├── land.shp          # Shapefile export for PyGMT
    ├── raw_land.gpkg     # Original coastline polygons
    ├── water.gpkg        # Subtracted water polygons
    └── metadata.json     # Bbox, polygon counts, config
```

The cache key encodes water subtraction options (e.g., `tokyo_bay_nolake`, `tokyo_bay_noriver`). Use `force=True` to regenerate:

```python
mask = load("tokyo_bay", config=config, force=True)
```

**Important**: When changing `CoastmaskConfig` options that affect cleanup (e.g., `protected_water_classes`, `min_hole_area_deg2`), delete the cache directory or use `force=True` to ensure the new settings take effect.

---

## Future: Overpass API Integration

The Geofabrik `fclass` system has limitations — it uses deprecated OSM tags and cannot distinguish canal polygons from lake polygons. A planned extension will support the [Overpass API](https://overpass-api.de/) to fetch water features with the current OSM tagging scheme (`natural=water` + `water=*`):

| `water=*` tag | Description | Ocean model usage |
|---------------|-------------|-------------------|
| `river` | River water surface | Keep (estuaries, river-ocean coupling) |
| `canal` | Canal water surface | Keep (port/harbor channels) |
| `reservoir` | Dammed lake | Fill as land |
| `lake` | Natural lake | Fill as land |
| `pond` | Pond | Fill as land |
| `basin` | Artificial basin | Fill as land |

This will enable per-type control of water features, with both live API fetching and offline GeoJSON file support for HPC environments without internet access.
