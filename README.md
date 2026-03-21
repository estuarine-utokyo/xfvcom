# xfvcom

[![Python Version](https://img.shields.io/badge/python-3.13+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python toolkit for [FVCOM](https://github.com/FVCOM-GitHub/FVCOM) ocean model data analysis and visualization, built on [xarray](https://docs.xarray.dev/).

---

## Installation

### Recommended: mamba (Miniforge)

Requires [Miniforge](https://github.com/conda-forge/miniforge) (mamba).

```bash
git clone https://github.com/jsasaki-utokyo/xfvcom.git
cd xfvcom
bash setup.sh            # Creates "xfvcom" environment (Python 3.13)
conda activate xfvcom
```

For Intel MPI on supercomputers:

```bash
bash setup.sh --impi     # Switches MPI backend to Intel MPI
```

See `bash setup.sh --help` for all options.

> **Note**: Use `mamba` (via `setup.sh`) to avoid conflicts with Intel oneAPI Python on supercomputers.
> Activation works with both `conda activate` and `mamba activate`.

### Alternative: pip (not recommended)

```bash
pip install -e .[dev]
```

pip does not install MPI, NetCDF CLI tools (nco, cdo), or other system-level packages included in `environment.yml`.
Use pip only for packages unavailable on conda-forge.

---

## Documentation

| Topic | Description |
|-------|-------------|
| [Plotting](docs/plot_2d.md) | 2D horizontal plots, time series, vertical sections |
| [Forcing Files](docs/forcing_generator.md) | River, meteorological, groundwater forcing generation |

---

## Command-Line Tools

### Validation

```bash
xfvcom-check-met met_forcing.nc                    # Check for NaN/Inf/out-of-bounds
xfvcom-check-met met_forcing.nc -o anomalies.csv   # Export to CSV
xfvcom-check-met met_forcing.nc --var uwind_speed  # Check specific variable
xfvcom-check-met met_forcing.nc --uniform          # Check spatial uniformity
```

### Forcing File Generation

```bash
# River forcing
xfvcom-make-river-nml river_data.csv --output rivers.nml
xfvcom-make-river-nc rivers.nml --start 2025-01-01T00:00Z --end 2025-12-31T23:00Z

# Meteorological forcing (constant/CSV)
xfvcom-make-met-nc grid.nc --start 2025-01-01T00:00Z --end 2025-01-07T00:00Z \
  --ts wind.csv:uwind,vwind --air-temperature 20.0 --utm-zone 54

# Meteorological forcing (GWO-AMD)
xfvcom-make-met-nc grid.dat --start 2020 --gwo-dir /path/to/GWO/Hourly \
  --station-map "slht:Tokyo,kous:Tokyo,clod:Tokyo,*:Chiba" --wind-factor 1.8 --utm-zone 54

# Meteorological forcing with gap filling
xfvcom-make-met-nc grid.dat --start 2020 --gwo-dir /path/to/GWO/Hourly \
  --station-map "*:Chiba" --fill-gaps --fallback-stations "Chiba:Tokyo,Yokohama" \
  --correlation-file gwo_correlations.yaml --solar-model empirical --utm-zone 54

# Compute station correlations for gap filling
xfvcom-calc-gwo-corr --gwo-dir /path/to/GWO/Hourly \
  --stations Tokyo,Chiba,Yokohama,Tateyama --years 2015-2020 \
  --include-solar-model -o gwo_correlations.yaml
```

**Note on GWO cloud cover data**: Cloud cover (`clod`) in GWO-AMD has 3-hourly observation intervals. Values at non-observed times are interpolated and marked with RMK=2 (no observation). The meteorological forcing generator now correctly includes these interpolated values as valid data, improving temporal resolution from 3-hourly to hourly.

```bash
# Groundwater forcing
xfvcom-make-groundwater-nc grid.nc --start 2025-01-01T00:00Z --end 2025-12-31T23:00Z \
  --flux 0.001 --temperature 15.0 --salinity 0.0
```

### Dye Time Series

```bash
xfvcom-dye-ts --base-dir /path/to/TB-FVCOM --basename tb_w18_r16 \
  --years 2020 2021 --members 1,2,3,4,5 --nodes 100,200,300 --output dye.nc
```

---

## Usage

### Load Data

```python
import xfvcom

loader = xfvcom.FvcomDataLoader("path/to/data", ncfile="output.nc")
ds = loader.ds

analyzer = xfvcom.FvcomAnalyzer(ds)
node_idx = analyzer.nearest_neighbor(lon=140.0, lat=35.0)
```

### Visualization

```python
cfg = xfvcom.FvcomPlotConfig(figsize=(12, 6))
plotter = xfvcom.FvcomPlotter(ds, cfg)

# Time series
fig = plotter.plot_timeseries("temperature", index=node_idx)

# 2D plot with satellite tiles
opts = xfvcom.FvcomPlotOptions(add_tiles=True, tile_provider="satellite")
fig = plotter.plot_2d("salinity", time="2020-07-01", siglay=0, opts=opts)
```

### Area Calculations

```python
from xfvcom import FvcomInputLoader

grid_loader = FvcomInputLoader(grid_file="grid.dat", utm_zone=54)
cv_area = grid_loader.calculate_node_area_median_dual([100, 200, 300], index_base=1)
elem_areas = grid_loader.calculate_element_area([10, 11, 12], index_base=1)
```

### Coast Masking

```python
from xfvcom.coastmask import load, CoastmaskConfig

mask = load("tokyo_bay")  # Built-in preset
mask.add_to_axes(ax)       # Add land mask to a Cartopy GeoAxes
```

### Ensemble Analysis

```python
from xfvcom.plot import plot_ensemble_timeseries, plot_dye_timeseries_stacked

fig, ax = plot_ensemble_timeseries(ds, var_name="dye", cfg=cfg)
result = plot_dye_timeseries_stacked(ds, cfg=cfg, output="stacked.png")
```

### Animation

```python
from xfvcom.plot.utils import create_anim_2d_plot

create_anim_2d_plot(plotter, processes=4, var_name="temperature", siglay=0, fps=10)
```

---

## API Reference

### Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `FvcomDataLoader` | `xfvcom.io` | Load FVCOM NetCDF output |
| `FvcomInputLoader` | `xfvcom.io` | Load grid files with area calculations |
| `FvcomAnalyzer` | `xfvcom.analysis` | Spatial analysis and physics calculations |
| `FvcomPlotter` | `xfvcom.plot` | Visualization engine |
| `FvcomPlotConfig` | `xfvcom.plot` | Plot styling configuration |
| `FvcomPlotOptions` | `xfvcom` | Per-plot options |
| `CoastMask` | `xfvcom.coastmask` | OSM-derived land masking for coastal plots |
| `CoastmaskConfig` | `xfvcom.coastmask` | Coastmask configuration (paths, tolerances) |
| `MetValidator` | `xfvcom.validation` | Forcing file validation |
| `GWOReader` | `xfvcom.io` | GWO-AMD meteorological data reader |
| `GWOForcingSource` | `xfvcom.io` | GWO data source for forcing generation |
| `GapFiller` | `xfvcom.io` | 4-step gap filling for GWO data |
| `StationCorrelations` | `xfvcom.io` | Station correlation management |
| `SolarEstimator` | `xfvcom.io` | Solar radiation estimation (pvlib) |

### Key Functions

| Function | Module | Description |
|----------|--------|-------------|
| `plot_ensemble_timeseries()` | `xfvcom.plot` | Ensemble line plots |
| `plot_dye_timeseries_stacked()` | `xfvcom.plot` | Stacked area plots |
| `create_anim_2d_plot()` | `xfvcom.plot.utils` | GIF/MP4 animations |
| `load()` | `xfvcom.coastmask` | Load/create coastmask for a region |
| `calculate_node_area()` | `xfvcom.grid` | Node area calculation |
| `extend_timeseries_*()` | `xfvcom.utils.timeseries_utils` | Time series extension |

---

## Development

```bash
conda activate xfvcom

# Code quality (CI equivalent)
black --check . && isort --check-only . && mypy . && pytest -m "not png"

# Auto-fix formatting, then check types
black . && isort . && mypy .

# Run all tests (including PNG regression)
pytest
```

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for development guidelines.

---

## Links

- [FVCOM](http://fvcom.smast.umassd.edu/fvcom/)
- [xarray](https://docs.xarray.dev/)
- [Cartopy](https://scitools.org.uk/cartopy/)
