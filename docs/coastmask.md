# Coastmask (moved to `xcoast`)

[Back to README](../README.md)

The OSM-derived land masking feature that lived under `xfvcom.coastmask` has
moved to a standalone package: **[`xcoast`](https://github.com/estuarine-utokyo/xcoast)**.

The `xfvcom.coastmask` module is now a thin deprecation shim that re-exports
the public API from `xcoast` and emits a `DeprecationWarning`. New code
should import from `xcoast` directly:

```python
from xcoast import load, CoastmaskConfig

mask = load("tokyo_bay")
mask.add_to_mpl(ax)
```

## Why was it moved?

The coastmask functionality is broadly useful for any coastal modeling or
monitoring workflow — not only for FVCOM preprocessing. Splitting it into its
own package keeps `xfvcom` focused on FVCOM I/O and analysis, lets `xcoast`
grow features such as OceanMesh shoreline preparation, observation-station
maps, and water-quality visualization, and slims down `xfvcom`'s dependency
footprint by removing the geospatial stack (`geopandas`, `shapely`, `pyogrio`)
from its required dependencies.

## Installation

`xcoast` is on GitHub. Install it inside the activated `xfvcom` environment:

```bash
pip install git+https://github.com/estuarine-utokyo/xcoast.git
# or, for editable development against a local clone:
pip install -e /path/to/xcoast
```

## Full reference

For configuration options, the processing pipeline, hole-filling cleanup, the
Overpass mode, and the cache layout, see the documentation in the `xcoast`
repository:

- [`xcoast/docs/coastmask.md`](https://github.com/estuarine-utokyo/xcoast/blob/main/docs/coastmask.md)
- [`xcoast/README.md`](https://github.com/estuarine-utokyo/xcoast/blob/main/README.md)
