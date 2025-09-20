# xfvcom

[![Python Version](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A comprehensive Python toolkit for FVCOM ocean model data analysis and visualization**

xfvcom streamlines preprocessing and postprocessing workflows for the Finite Volume Community Ocean Model ([FVCOM](https://github.com/FVCOM-GitHub/FVCOM)). Built on [xarray](https://docs.xarray.dev/en/stable/), it provides powerful tools for ocean model data analysis, forcing file generation, and publication-quality visualizations.

## 🚀 Key Features

### 📊 Data I/O & Processing
- **FVCOM NetCDF Support**: Load and process model output files with automatic mesh completion
- **Multi-format Grid Support**: Handle both ASCII (.dat) and NetCDF grid formats
- **Coordinate Systems**: Seamless conversion between UTM and geographic (lat/lon) coordinates
- **Depth Calculations**: Automatic depth variable computation from sigma layers and bathymetry
- **Time Zone Handling**: Intelligent timezone conversion with configurable defaults

### 🔬 Analysis Tools
- **Spatial Analysis**: KDTree-based nearest neighbor search for efficient spatial queries
- **Physics Calculations**: Layer averages, tidal decomposition, variable filtering by dimensions
- **Time Series Processing**: Advanced extension methods (seasonal, linear, forward-fill)
- **Grid Utilities**: Mesh connectivity analysis and validation tools
- **Area Calculations**: Compute total area of triangular elements containing specified nodes

### 🎨 Visualization
- **Static Plots**: Time series, 2D contours, vector fields, vertical sections
- **Animations**: Generate GIF and MP4 animations for temporal data
- **Interactive Plots**: Plotly integration for web-based interactive visualizations
- **Map Projections**: Full Cartopy support for geographic visualizations
- **Triangular Mesh**: Native support for FVCOM's unstructured grid visualization
- **Enhanced Node/Marker Display**: Advanced clipping control for geographic coordinates with Cartopy

### ⚡ Forcing File Generation
- **River Forcing**: Generate river discharge and temperature/salinity inputs
- **Meteorological Forcing**: Create atmospheric forcing files with comprehensive variables
- **Groundwater Forcing**: Support for groundwater flux with optional dye tracers
- **Flexible Input**: Mix constants with CSV time series, multiple data formats

---

## 📚 Documentation

### Core Documentation
- [2-D Horizontal Plotting](docs/plot_2d.md) - Spatial visualization techniques
- [Time-series & Depth-profiles](docs/plot_ts.md) - Temporal data analysis
- [Vertical Sections](docs/plot_section.md) - Cross-sectional views
- [Forcing File Generator](docs/forcing_generator.md) - Input file creation

### Quick Links
- [API Reference](#-api-reference)
- [Examples](#-examples)
- [CLI Commands](#-command-line-tools)
- [Contributing Guidelines](docs/CONTRIBUTING.md)

## 🔧 Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Conda (Miniforge/Mambaforge recommended)
- Git

### Quick Install

#### Option 1: Automated Setup (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/xfvcom.git
cd xfvcom

# Run the setup script
./setup.sh
```

#### Option 2: Using Conda Environment File
```bash
# Clone the repository
git clone https://github.com/yourusername/xfvcom.git
cd xfvcom

# Create environment from file
conda env create -f environment.yml
conda activate xfvcom
```

#### Option 3: Manual Installation
```bash
# Create conda environment
conda create -n xfvcom python=3.11 -c conda-forge \
    numpy xarray pandas matplotlib cartopy pyproj \
    scipy scikit-learn imageio moviepy tqdm netcdf4 \
    pytest mypy black isort jinja2 pyyaml

# Activate environment
conda activate xfvcom

# Install xfvcom in editable mode
pip install -e .
```

### Verify Installation
```bash
python -c "import xfvcom; print(f'xfvcom {xfvcom.__version__} installed successfully')"
```

## 💻 Quick Start

### Basic Usage

```python
import xfvcom

# Load FVCOM data
fvcom = xfvcom.FvcomDataLoader("path/to/data", ncfile="output.nc")
ds = fvcom.ds

# Analyze data
analyzer = xfvcom.FvcomAnalyzer(ds)
node_idx = analyzer.nearest_neighbor(lon=140.0, lat=35.0)

# Calculate area for selected nodes (requires grid file)
from xfvcom import calculate_node_area
area = calculate_node_area("path/to/grid.dat", [100, 200, 300], utm_zone=54)
print(f"Area: {area/1e6:.2f} km²")

# Create visualizations
cfg = xfvcom.FvcomPlotConfig(figsize=(10, 6), dpi=150)
plotter = xfvcom.FvcomPlotter(ds, cfg)

# Time series plot
fig = plotter.plot_timeseries("temperature", index=node_idx)

# 2D spatial plot
fig = plotter.plot_2d("salinity", time="2020-07-01", siglay=0)
```

### Creating Animations

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

### Interactive Visualizations

```python
from xfvcom.plot.plotly_utils import plot_timeseries_comparison

# Compare multiple variables interactively
fig = plot_timeseries_comparison(
    ds,
    variables=["temperature", "salinity"],
    node_idx=node_idx,
    start_time="2020-01-01",
    end_time="2020-12-31"
)
fig.show()
```

## 🔨 Command-Line Tools

### River Forcing Generation

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
# Basic usage with time series
xfvcom-make-met-nc grid.nc \
  --start 2025-01-01T00:00Z \
  --end 2025-01-02T00:00Z \
  --ts weather.csv:uwind,vwind,air_temperature,humidity

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

# Time-varying with dye tracer
xfvcom-make-groundwater-nc grid.nc \
  --start 2025-01-01T00:00Z \
  --end 2025-12-31T23:00Z \
  --flux groundwater.csv:datetime,node_id,flux \
  --temperature groundwater.csv:datetime,node_id,temperature \
  --salinity 0.0 \
  --dye-concentration 1.0
```

## 🔍 Examples

### Advanced Time Series Processing

```python
from xfvcom.utils.timeseries_utils import (
    extend_timeseries_seasonal,
    interpolate_missing_values
)

# Extend river data using seasonal patterns
extended_ds = extend_timeseries_seasonal(
    ds,
    target_end="2025-12-31",
    variables=["river_flux", "river_temp"]
)

# Fill gaps in observational data
filled_ds = interpolate_missing_values(
    ds,
    method="linear",
    max_gap="7D"
)
```

### Node Area Calculations

Calculate the total area of triangular elements containing specified nodes, useful for spatial analysis and domain decomposition:

```python
from xfvcom import FvcomInputLoader, calculate_node_area

# Method 1: Using existing loader
loader = FvcomInputLoader("grid.dat", utm_zone=54)
nodes = [100, 200, 300, 500, 1000]  # 1-based node indices (FVCOM convention)
area = loader.calculate_node_area(nodes, index_base=1)
print(f"Total area: {area:,.0f} m² ({area/1e6:.2f} km²)")

# Method 2: Direct calculation from grid file
area = calculate_node_area(
    grid_file="grid.dat",
    node_indices=[100, 200, 300],
    utm_zone=54,
    index_base=1  # Use 0 for zero-based indexing
)

# Calculate area for all nodes
total_area = calculate_node_area("grid.dat", None, utm_zone=54)
print(f"Total mesh area: {total_area/1e6:.2f} km²")
```

**Features:**
- Support for both 0-based and 1-based node indexing
- Efficient calculation using shoelace formula
- Returns area in square meters (assuming UTM coordinates)
- Handles overlapping triangular elements automatically

### Enhanced Node Visualization with Cartopy

xfvcom provides advanced control for displaying node markers and text labels on geographic maps, addressing known Cartopy limitations with text clipping:

```python
from xfvcom import make_node_marker_post

# Independent buffer control for markers and text
pp = make_node_marker_post(
    nodes=[1, 100, 500, 1000],  # Node indices to display
    plotter=plotter,
    marker_kwargs={"color": "red", "markersize": 4},
    text_kwargs={"color": "yellow", "fontsize": 8, "clip_on": True},
    index_base=1,  # FVCOM uses 1-based indexing
    respect_bounds=True,
    marker_clip_buffer=0.002,  # Include markers slightly outside bounds
    text_clip_buffer=-0.001,   # Hide text near edges to prevent overflow
)

# Apply to a map with specific extent
opts = FvcomPlotOptions(
    xlim=(139.85, 139.95),
    ylim=(35.36, 35.45),
    add_tiles=True
)
ax = plotter.plot_2d(post_process_func=pp, opts=opts)
```

**Key Features:**
- **`marker_clip_buffer`**: Controls marker visibility at map boundaries
  - Positive values: Include markers outside the specified extent
  - Negative values: Exclude markers near boundaries
- **`text_clip_buffer`**: Controls text label visibility (fixes Cartopy `clip_on` issues)
  - Positive values: Show text labels beyond map extent
  - Negative values: Hide text near edges to prevent overflow
- **Performance optimized**: Uses vectorized operations for large node sets
- **Cartopy workaround**: Automatically handles geographic coordinate text clipping

This feature is particularly useful for:
- Dense grids where text labels may overlap at boundaries
- Analyzing nodes at domain edges
- Creating clean visualizations with precise boundary control

### Programmatic Forcing Generation

```python
from xfvcom.io import MetNetCDFGenerator, GroundwaterNetCDFGenerator

# Meteorological forcing
met_gen = MetNetCDFGenerator(
    grid_nc="grid.nc",
    start="2025-01-01T00:00Z",
    end="2025-12-31T23:00Z",
    dt_seconds=3600,
    ts_specs=["weather.csv:all"],  # Read all columns
    data_tz="UTC"
)
met_gen.write("met_forcing.nc")

# Groundwater with spatial variation
gw_gen = GroundwaterNetCDFGenerator(
    grid_nc="grid.nc",
    start="2025-01-01T00:00Z",
    end="2025-12-31T23:00Z",
    groundwater_flux="flux_map.csv:node_id,flux",
    groundwater_temp=15.0,
    groundwater_salt=0.0
)
gw_gen.write("groundwater_forcing.nc")
```

## 📦 API Reference

### Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `FvcomDataLoader` | `xfvcom.io` | Load and preprocess FVCOM NetCDF files |
| `FvcomAnalyzer` | `xfvcom.analysis` | Physics calculations and spatial analysis |
| `FvcomPlotter` | `xfvcom.plot` | Main visualization engine |
| `FvcomPlotConfig` | `xfvcom.plot` | Configuration for plot styling |
| `FvcomGrid` | `xfvcom.grid` | Grid manipulation utilities with area calculations |

### Generator Classes

| Class | Module | Description |
|-------|--------|-------------|
| `RiverNetCDFGenerator` | `xfvcom.io` | Generate river forcing files |
| `MetNetCDFGenerator` | `xfvcom.io` | Generate meteorological forcing |
| `GroundwaterNetCDFGenerator` | `xfvcom.io` | Generate groundwater forcing |
| `RiverNamelistGenerator` | `xfvcom.io` | Create FVCOM river namelists |

### Utility Functions

| Function | Module | Description |
|----------|--------|-------------|
| `create_anim_2d_plot()` | `xfvcom.plot.utils` | Create animations from 2D data |
| `extend_timeseries_*()` | `xfvcom.utils.timeseries_utils` | Time series extension methods |
| `plot_timeseries_comparison()` | `xfvcom.plot.plotly_utils` | Interactive comparison plots |
| `make_node_marker_post()` | `xfvcom.plot.markers` | Enhanced node/text display with clipping control |
| `calculate_node_area()` | `xfvcom.grid` | Calculate total area for specified nodes |

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_plot_2d.py

# Skip PNG regression tests
pytest -m "not png"

# Regenerate plot baselines
pytest --regenerate-baseline
```

## 🛠️ Development

### Code Quality Tools

```bash
# Format code
black xfvcom/

# Sort imports
isort xfvcom/

# Type checking
mypy .

# Run all checks
pre-commit run --all-files
```

### Building Documentation

```bash
cd docs
make html
open _build/html/index.html
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/CONTRIBUTING.md) for details on:

- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit with descriptive message
6. Push to your fork
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Jun Sasaki** - *Initial work* - [jsasaki.ece@gmail.com](mailto:jsasaki.ece@gmail.com)

## 🙏 Acknowledgments

- FVCOM development team for the ocean model
- xarray developers for the excellent data structure
- All contributors who have helped improve this package

## 📚 Citation

If you use xfvcom in your research, please cite:

```bibtex
@software{xfvcom,
  author = {Sasaki, Jun},
  title = {xfvcom: A Python toolkit for FVCOM data analysis},
  year = {2024},
  url = {https://github.com/yourusername/xfvcom}
}
```

## 🔗 Links

- [FVCOM Official Site](http://fvcom.smast.umassd.edu/fvcom/)
- [xarray Documentation](https://docs.xarray.dev/)
- [Cartopy Documentation](https://scitools.org.uk/cartopy/)
- [Issue Tracker](https://github.com/yourusername/xfvcom/issues)
