# xfvcom

[![Python Version](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A comprehensive Python toolkit for FVCOM ocean model data analysis and visualization**

xfvcom streamlines preprocessing and postprocessing workflows for the Finite Volume Community Ocean Model ([FVCOM](https://github.com/FVCOM-GitHub/FVCOM)). Built on [xarray](https://docs.xarray.dev/en/stable/), it provides powerful tools for ocean model data analysis, forcing file generation, and publication-quality visualizations.

---

## Key Features

### Data I/O & Processing
- **FVCOM NetCDF Support**: Load and process model output files with automatic mesh completion
- **Multi-format Grid Support**: Handle both ASCII (.dat) and NetCDF grid formats
- **Coordinate Systems**: Seamless conversion between UTM and geographic (lat/lon) coordinates
- **Depth Calculations**: Automatic depth variable computation from sigma layers and bathymetry
- **Time Zone Handling**: Intelligent timezone conversion (default: Asia/Tokyo to UTC)

### Analysis Tools
- **Spatial Analysis**: KDTree-based nearest neighbor search for efficient spatial queries
- **Physics Calculations**: Layer averages, tidal decomposition, variable filtering by dimensions
- **Time Series Processing**: Extension methods (seasonal, linear, forward-fill), interpolation, resampling
- **Grid Utilities**: Mesh connectivity analysis and validation tools
- **Area Calculations**:
  - **Median-dual control volumes**: FVCOM-standard node areas with shoreline-aware boundary closure
  - **Triangle sums**: Legacy area calculation using surrounding triangles
  - **Element areas**: Per-cell triangle areas for targeted diagnostics
- **Ensemble Analysis**: Multi-member dye tracer analysis, linearity verification, source identification

### Visualization
- **Static Plots**: Time series, 2D contours, vector fields, vertical sections
- **Animations**: Generate GIF and MP4 animations for temporal data
- **Interactive Plots**: Plotly integration for web-based visualizations (optional)
- **Map Projections**: Full Cartopy support with tile providers (OSM, Google Satellite, etc.)
- **Triangular Mesh**: Native support for FVCOM's unstructured grid visualization
- **Enhanced Display**: Advanced clipping control for node markers and text labels on geographic maps
- **Ensemble Plots**: Line plots with automatic colormap selection, stacked area plots for multi-member data

### Forcing File Generation
- **River Forcing**: Generate river discharge and temperature/salinity inputs from CSV or constants
- **Meteorological Forcing**: Create atmospheric forcing files with comprehensive variables
- **Groundwater Forcing**: Support for groundwater flux with temperature, salinity, and optional dye tracers
- **Flexible Input**: Mix constants with CSV time series, node-specific or global values

### Validation Tools
- **Meteorological Forcing Validation**: Check for NaN, Inf, and out-of-bounds values
- **Detailed Reporting**: Node/element locations (1-based indexing) with lon/lat coordinates
- **CSV Export**: Export anomaly reports for further analysis

---

## Documentation

- [2-D Horizontal Plotting](docs/plot_2d.md) - Spatial visualization techniques
- [Time-series & Depth-profiles](docs/plot_ts.md) - Temporal data analysis
- [Vertical Sections](docs/plot_section.md) - Cross-sectional views
- [Forcing File Generator](docs/forcing_generator.md) - Input file creation

---

## Installation

### Prerequisites
- Python 3.13 or later
- Conda with mamba (Miniforge recommended)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/jsasaki-utokyo/xfvcom.git
cd xfvcom

# Run setup script (creates conda environment and installs xfvcom)
./setup.sh

# Activate environment
conda activate xfvcom
```

### Setup Script Options

```bash
# Show help
./setup.sh --help

# Custom environment name
./setup.sh --env-name myenv

# Force recreate existing environment
./setup.sh --force

# Skip development dependencies
./setup.sh --no-dev
```

### Manual Installation

If you prefer manual installation:

```bash
# Create environment from environment.yml
mamba env create -f environment.yml

# Activate and verify
conda activate xfvcom
python -c "import xfvcom; print(f'xfvcom {xfvcom.__version__}')"
```

---

## Quick Start

### Load and Analyze Data

```python
import xfvcom

# Load FVCOM output
loader = xfvcom.FvcomDataLoader("path/to/data", ncfile="output.nc")
ds = loader.ds

# Spatial analysis
analyzer = xfvcom.FvcomAnalyzer(ds)
node_idx = analyzer.nearest_neighbor(lon=140.0, lat=35.0)
```

### Area Calculations

```python
from xfvcom import FvcomInputLoader, calculate_node_area

# Load grid
grid_loader = FvcomInputLoader(grid_file="grid.dat", utm_zone=54)

# Median-dual control volumes (FVCOM standard)
cv_area = grid_loader.calculate_node_area_median_dual([100, 200, 300], index_base=1)
print(f"Control volume area: {cv_area/1e6:.2f} km2")

# Triangle sum method (legacy)
tri_area = calculate_node_area("grid.dat", [100, 200, 300], utm_zone=54, index_base=1)
print(f"Triangle area: {tri_area/1e6:.2f} km2")

# Element areas
elem_areas = grid_loader.calculate_element_area([10, 11, 12], index_base=1)
print(f"Element areas: {[f'{a:.1f}' for a in elem_areas]} m2")
```

### Create Visualizations

```python
# Configure plot style
cfg = xfvcom.FvcomPlotConfig(
    figsize=(12, 6),
    fontsize={"xlabel": 14, "ylabel": 14, "title": 15},
    linewidth={"plot": 1.8}
)

# Create plotter
plotter = xfvcom.FvcomPlotter(ds, cfg)

# Time series plot
fig = plotter.plot_timeseries("temperature", index=node_idx)

# 2D spatial plot
opts = xfvcom.FvcomPlotOptions(
    add_tiles=True,
    tile_provider="satellite",  # or "osm"
    mesh_color="lightgray"
)
fig = plotter.plot_2d("salinity", time="2020-07-01", siglay=0, opts=opts)
```

### Ensemble Time Series Analysis

```python
from xfvcom.plot import plot_ensemble_timeseries, plot_dye_timeseries_stacked

# Line plot with automatic colormap selection
# (tab20 for <=20 members, hsv for >20 members)
fig, ax = plot_ensemble_timeseries(
    ds,
    var_name="dye",
    cfg=cfg,
    legend_outside=True,
    title="Ensemble Dye Concentration"
)

# Stacked area plot
result = plot_dye_timeseries_stacked(
    ds,
    cfg=cfg,
    member_ids=[1, 2, 3, 4, 5],
    title="Dye Concentration (Stacked)",
    output="dye_stacked.png"
)
```

### Create Animations

```python
from xfvcom.plot.utils import create_anim_2d_plot

# Generate animated GIF
create_anim_2d_plot(
    plotter=plotter,
    var_name="temperature",
    siglay=0,  # Surface layer
    fps=10,
    output_format="gif",
    plot_kwargs={"vmin": 10, "vmax": 30, "cmap": "RdYlBu_r"}
)
```

---

## Command-Line Tools

### Meteorological Forcing Validation

```bash
# Check for abnormal values (NaN, Inf, out-of-bounds)
xf-check-met met_forcing.nc

# Quiet mode with CSV export
xf-check-met met_forcing.nc -q -o anomalies.csv

# Custom bounds for specific variable
xf-check-met met_forcing.nc --bound 'air_temperature:-40:50'

# Check specific variables only
xf-check-met met_forcing.nc --var uwind_speed --var vwind_speed

# Disable progress bar
xf-check-met met_forcing.nc --no-progress
```

### River Forcing

```bash
# Generate river namelist from CSV
xfvcom-make-river-nml river_data.csv --output rivers.nml

# Create river forcing NetCDF
xfvcom-make-river-nc rivers.nml \
  --start 2025-01-01T00:00Z \
  --end 2025-12-31T23:00Z \
  --dt 3600
```

### Meteorological Forcing

```bash
# Mix constants with CSV data
xfvcom-make-met-nc grid.nc \
  --start 2025-01-01T00:00Z \
  --end 2025-01-07T00:00Z \
  --ts wind.csv:uwind,vwind \
  --air-temperature 20.0 \
  --humidity 0.7
```

### Groundwater Forcing

```bash
# Constant values
xfvcom-make-groundwater-nc grid.nc \
  --start 2025-01-01T00:00Z \
  --end 2025-12-31T23:00Z \
  --flux 0.001 \
  --temperature 15.0 \
  --salinity 0.0

# With dye tracer
xfvcom-make-groundwater-nc grid.nc \
  --start 2025-01-01T00:00Z \
  --end 2025-12-31T23:00Z \
  --flux groundwater.csv:datetime,node_id,flux \
  --dye-concentration 1.0
```

### Dye Time Series Extraction

```bash
# Extract and aggregate dye concentrations from ensemble runs
xfvcom-dye-ts \
  --base-dir /path/to/TB-FVCOM \
  --basename tb_w18_r16 \
  --years 2020 2021 \
  --members 1,2,3,4,5 \
  --nodes 100,200,300 \
  --output dye_timeseries.nc
```

---

## Examples

### Time Series Extension

```python
from xfvcom.utils.timeseries_utils import (
    extend_timeseries_seasonal,
    extend_timeseries_linear,
    interpolate_missing_values
)

# Extend using seasonal patterns
extended_df = extend_timeseries_seasonal(
    df,
    extend_to="2025-12-31",
    period="1Y"  # Repeat yearly pattern
)

# Linear extrapolation
extended_df = extend_timeseries_linear(
    df,
    extend_to="2025-12-31",
    lookback_periods=30
)

# Fill gaps
filled_df = interpolate_missing_values(
    df,
    method="linear",
    limit=7  # Max consecutive NaNs to fill
)
```

### Enhanced Node Visualization

```python
from xfvcom import make_node_marker_post

# Advanced clipping control for markers and text
pp = make_node_marker_post(
    nodes=[1, 100, 500, 1000],
    plotter=plotter,
    marker_kwargs={"color": "red", "markersize": 5},
    text_kwargs={"color": "yellow", "fontsize": 10, "clip_on": True},
    index_base=1,
    marker_clip_buffer=0.002,   # Include markers slightly outside bounds
    text_clip_buffer=-0.001     # Hide text near edges (fixes Cartopy bug)
)

opts = xfvcom.FvcomPlotOptions(xlim=(139.85, 139.95), ylim=(35.36, 35.45))
ax = plotter.plot_2d(post_process_func=pp, opts=opts)
```

### Member-Node Mapping Analysis

```python
from xfvcom.ensemble_analysis import (
    extract_member_node_mapping,
    get_member_summary,
    export_member_mapping
)

# Extract dye release locations for each ensemble member
mapping = extract_member_node_mapping(
    dye_files=["dye_member1.nc", "dye_member2.nc", ...],
    threshold=1e-6,
    grid_file="grid.dat"
)

# Get summary statistics
summary = get_member_summary(mapping)
print(f"Total unique nodes: {summary['total_nodes']}")

# Export to CSV
export_member_mapping(mapping, "member_mapping.csv", grid_file="grid.dat")
```

---

## API Reference

### Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `FvcomDataLoader` | `xfvcom.io` | Load and preprocess FVCOM NetCDF output files |
| `FvcomInputLoader` | `xfvcom.io` | Load grid files with area calculation methods |
| `FvcomAnalyzer` | `xfvcom.analysis` | Physics calculations and spatial analysis |
| `FvcomPlotter` | `xfvcom.plot` | Main visualization engine |
| `FvcomPlotConfig` | `xfvcom.plot` | Centralized plot styling configuration |
| `FvcomPlotOptions` | `xfvcom` | Per-plot customization options |
| `FvcomGrid` | `xfvcom.grid` | Grid manipulation and validation |

### Validation Classes

| Class | Module | Description |
|-------|--------|-------------|
| `MetValidator` | `xfvcom.validation` | Validate meteorological forcing files |
| `AnomalyReport` | `xfvcom.validation` | Container for validation results |
| `AnomalyRecord` | `xfvcom.validation` | Single anomaly with location and value |
| `PhysicalBounds` | `xfvcom.validation` | Define min/max bounds for variables |

### Ensemble Analysis

| Function | Module | Description |
|----------|--------|-------------|
| `extract_member_node_mapping()` | `xfvcom.ensemble_analysis` | Extract dye release nodes from ensemble members |
| `get_member_summary()` | `xfvcom.ensemble_analysis` | Get summary statistics for member mapping |
| `export_member_mapping()` | `xfvcom.ensemble_analysis` | Export mapping to CSV with coordinates |

### Dye Time Series

| Function | Module | Description |
|----------|--------|-------------|
| `collect_member_files()` | `xfvcom.dye_timeseries` | Collect NetCDF files for ensemble members |
| `aggregate()` | `xfvcom.dye_timeseries` | Aggregate time series across members |
| `negative_stats()` | `xfvcom.dye_timeseries` | Analyze negative value statistics |
| `verify_linearity()` | `xfvcom.dye_timeseries` | Verify ensemble linearity assumption |

### Plotting Functions

| Function | Module | Description |
|----------|--------|-------------|
| `plot_ensemble_timeseries()` | `xfvcom.plot` | Line plots with auto colormap selection |
| `plot_ensemble_statistics()` | `xfvcom.plot` | Mean, std, and coefficient of variation |
| `plot_dye_timeseries_stacked()` | `xfvcom.plot` | Stacked area plots for ensemble data |
| `create_anim_2d_plot()` | `xfvcom.plot.utils` | Create GIF/MP4 animations |
| `make_node_marker_post()` | `xfvcom` | Enhanced node/text display with clipping |
| `make_element_boundary_post()` | `xfvcom` | Highlight element boundaries |

### Utility Functions

| Function | Module | Description |
|----------|--------|-------------|
| `calculate_node_area()` | `xfvcom.grid` | Triangle sum area calculation |
| `calculate_element_area()` | `xfvcom.grid` | Element area calculation |
| `extend_timeseries_*()` | `xfvcom.utils.timeseries_utils` | Time series extension methods |
| `interpolate_missing_values()` | `xfvcom.utils.timeseries_utils` | Interpolate gaps in data |
| `resample_timeseries()` | `xfvcom.utils.timeseries_utils` | Change time resolution |

---

## Testing

```bash
# Run all tests
pytest

# Skip PNG regression tests (CI default)
pytest -m "not png"

# Regenerate plot baselines (after intentional visual changes)
pytest --regenerate-baseline

# Run specific test module
pytest tests/test_dye_timeseries_stacked.py -v
```

---

## Development

### Code Quality

```bash
# Format code
black .

# Sort imports
isort .

# Type checking
mypy xfvcom

# All checks (what CI runs)
black --check .
isort --check-only .
mypy xfvcom
pytest -m "not png"
```

### Building Documentation

```bash
cd docs
make html
```

---

## Links

- [FVCOM Official Site](http://fvcom.smast.umassd.edu/fvcom/)
- [xarray Documentation](https://docs.xarray.dev/)
- [Cartopy Documentation](https://scitools.org.uk/cartopy/)
