# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

xfvcom is a Python package for preprocessing and postprocessing data from the Finite Volume Community Ocean Model (FVCOM). It's built on xarray and specializes in ocean model data analysis, ensemble analysis, and publication-quality visualization.

**Current Version**: 0.2.0

## Environment Setup for Claude Code

### Quick Start
```bash
# Activate the existing fvcom environment
conda activate fvcom

# Verify environment is active
python --version  # Should show Python 3.11.x or 3.12.x
python -c "import xfvcom; print(f'xfvcom {xfvcom.__version__}')"
```

### Available Environments
- `fvcom` - Main development environment with xfvcom installed in editable mode
- `xfvcom-docs` - Documentation building environment
- Other project-specific environments: `era_dl`, `moewq`, `pylag`, `river_dl`

## Essential Commands

### Development Setup (if creating new environment)
```bash
# Create and activate conda environment
conda create -n xfvcom python=3.12 -c conda-forge \
  numpy xarray pandas matplotlib cartopy pyproj scipy scikit-learn \
  imageio moviepy tqdm pytest mypy black isort jinja2 pyyaml types-pyyaml \
  netcdf4

conda activate xfvcom

# Install package in editable mode
pip install -e .
```

### Testing
```bash
# Run all tests (CI-equivalent)
pytest -m "not png"

# Run all tests including PNG regression
pytest

# Regenerate baseline images when plot appearance changes intentionally
pytest --regenerate-baseline -q

# Run specific test file
pytest tests/test_dye_timeseries_stacked.py -v

# Run specific test class or function
pytest tests/test_grid.py::TestGridCalculations
```

#### CI-equivalent checks

Run exactly what GitHub CI executes:

```bash
black --check .
isort --check-only .
mypy xfvcom
pytest -m "not png"
```

**Note**: Ruff is NOT run in CI, so ruff errors are informational only.

### Code Quality
```bash
# Format code (REQUIRED before commit)
black .
isort .

# Type check
mypy xfvcom

# View linting suggestions (informational, not enforced)
ruff check .
```

### Building Documentation
```bash
cd docs
make html
# Output: docs/_build/html/index.html
```

---

## Architecture

### Core Components

#### 1. **Data I/O** (xfvcom/io/)
- **FvcomDataLoader** (`io/core.py`): Load FVCOM NetCDF output files
  - Coordinate transformations (UTM ↔ geographic)
  - Automatic depth variable calculation
  - Mesh completion and validation

- **FvcomInputLoader** (`io/core.py`): Load grid files (.dat, .nc)
  - Area calculations (median-dual control volumes, triangle sums, element areas)
  - Mesh connectivity analysis
  - Support for 0-based and 1-based indexing

- **Force File Generators** (`io/*_generator.py`):
  - `RiverNetCDFGenerator`, `RiverNamelistGenerator`
  - `MetNetCDFGenerator`
  - `GroundwaterNetCDFGenerator`
  - Base class: `BaseForceGenerator`
  - **Important**: Use netCDF4 directly (not xarray) for FVCOM compatibility

#### 2. **Analysis** (xfvcom/analysis.py)
- **FvcomAnalyzer**: Physics calculations
  - KDTree nearest neighbor search
  - Layer averages
  - Tidal decomposition
  - Variable filtering by dimensions

#### 3. **Ensemble Analysis** (xfvcom/ensemble_analysis/)
- **Member Info** (`member_info.py`): Dye tracer source identification
  - `extract_member_node_mapping()`: Extract release locations
  - `get_member_summary()`: Statistical summaries
  - `export_member_mapping()`: Export to CSV with coordinates
  - `get_node_coordinates()`: Get lon/lat for nodes

#### 4. **Dye Time Series** (xfvcom/dye_timeseries.py)
- Multi-member ensemble aggregation
- Negative value handling (keep, clip)
- Time alignment strategies (intersection, same_calendar, climatology)
- Linearity verification
- Comprehensive data classes: `DyeCase`, `Selection`, `Paths`, `NegPolicy`, `AlignPolicy`

#### 5. **Visualization** (xfvcom/plot/)

**Core Plotting:**
- **FvcomPlotter** (`plot/core.py`): Main plotting engine
  - 2D plots: `plot_2d()` with contours, vectors, tiles
  - Time series: `plot_timeseries()`
  - Vertical sections: `plot_section()`
  - Support for post-processing functions

- **FvcomPlotConfig** (`plot/config.py`): Centralized styling configuration
  - Font sizes, line widths, colors, grid styles
  - Used by all modern plotting functions
  - Dataclass-based with sensible defaults

- **FvcomPlotOptions** (`plot_options.py`): Per-plot customization
  - Map extent (xlim, ylim)
  - Tile providers (OSM, satellite)
  - Mesh styling
  - Projection settings

**Enhanced Plotting:**
- **Ensemble Time Series** (`plot/timeseries.py`):
  - `plot_ensemble_timeseries()`: Line plots with auto colormap selection
  - `plot_ensemble_statistics()`: Mean, std, coefficient of variation
  - `apply_smart_time_ticks()`: Intelligent datetime formatting
  - Automatic colormap: tab20 (≤20 members), hsv (>20 members)

- **Stacked Plots** (`plot/dye_timeseries.py`):
  - `plot_dye_timeseries_stacked()`: Stacked area plots
  - FvcomPlotConfig support
  - Automatic colormap selection
  - Hard-fail on NaN values (design choice)

- **Markers and Boundaries** (`plot/markers.py`, `plot/boundaries.py`):
  - `make_node_marker_post()`: Enhanced node display with clipping control
  - `make_element_boundary_post()`: Element boundary highlighting
  - Cartopy workarounds for text clipping

**Animation:**
- **Utils** (`plot/utils.py`): Animation creation
  - `create_anim_2d_plot()`: GIF/MP4 from 2D data
  - Frame generation and encoding

**Interactive (Optional):**
- **Plotly Utils** (`plot/plotly_utils.py`): Web-based interactive plots
  - Requires plotly installation
  - `plot_timeseries_comparison()`, `plot_timeseries_multi_variable()`
  - Graceful import failure if plotly not available

#### 6. **Grid Utilities** (xfvcom/grid/)
- **FvcomGrid** (`grid/grid.py`): Grid manipulation
  - `calculate_node_area()`: Triangle sum method (legacy)
  - `calculate_node_area_median_dual()`: FVCOM-standard control volumes
  - `calculate_element_area()`: Per-element areas
  - Shoelace formula for area calculations

#### 7. **CLI Tools** (xfvcom/cli/)
Command-line interfaces for common tasks:
- `xfvcom-make-river-nml`: River namelist generation
- `xfvcom-make-river-nc`: River forcing NetCDF
- `xfvcom-make-met-nc`: Meteorological forcing
- `xfvcom-make-groundwater-nc`: Groundwater forcing
- `xfvcom-dye-ts`: Dye time series extraction and aggregation

#### 8. **Utilities** (xfvcom/utils/)
- **Time Series** (`utils/timeseries_utils.py`):
  - `extend_timeseries_*()`: Seasonal, linear, forward-fill extension
  - `interpolate_missing_values()`: Gap filling
  - `resample_timeseries()`: Frequency conversion

- **Helpers** (`utils/helpers.py`, `utils/helpers_utils.py`):
  - GIF creation: `create_gif()`, `create_gif_from_frames()`
  - Frame generation: `FrameGenerator`, `PlotHelperMixin`
  - Model evaluation: `evaluate_model_scores()`

### Data Flow

```
Input Files → FvcomDataLoader → xarray Dataset → FvcomAnalyzer → Analysis Results
                                       ↓
                                 FvcomPlotter + FvcomPlotConfig → Plots/Animations
                                       ↓
                              Ensemble Analysis → DyeCase → Statistics

Grid File → FvcomInputLoader → Area Calculations
                             → Mesh Connectivity

CSV/Constants → Force Generators → FVCOM Input Files (.nc, .nml)
```

### Key Dependencies

- **xarray**: Core data structure for multidimensional arrays
- **pandas**: Time series manipulation
- **numpy**: Numerical operations
- **matplotlib**: Base plotting library
- **cartopy**: Geographic projections and tile providers
- **scipy**: Scientific computations (KDTree, interpolation)
- **scikit-learn**: Machine learning utilities
- **netCDF4**: Low-level NetCDF I/O (for FVCOM compatibility)
- **imageio/moviepy**: Animation creation
- **plotly** (optional): Interactive visualizations

---

## Development Guidelines

### Code Style
- **Formatter**: Black (line length: 88 characters)
- **Import sorting**: isort with Black-compatible profile
- **Type hints**: Required for public APIs, encouraged for internal
- **Docstrings**: Google-style for public functions
- **Pre-commit hooks**: Automatically format on commit (if configured)

### Testing Strategy

**Test Organization:**
- `tests/test_*.py` - Unit tests
- `tests/baseline/` - PNG regression baseline images
- `tests/data/` - Test data files

**Test Markers:**
- `@pytest.mark.png` - PNG regression tests (skipped in CI)
- Default: All tests run locally, PNG skipped in CI

**PNG Regression Tests:**
```python
@pytest.mark.png
def test_plot_produces_expected_output(plotter, baseline_dir, regen_baseline):
    fig = plotter.plot_2d(...)
    # Compare against baseline or regenerate if --regenerate-baseline
```

**When to Regenerate Baselines:**
- Intentional visual changes (font sizes, colors, layouts)
- Updated matplotlib/cartopy versions
- New plot features
- **Command**: `pytest --regenerate-baseline -q`

### Module Relationships

**Plotting Module Structure:**
```
xfvcom/plot/
├── __init__.py           # Public API exports
├── config.py             # FvcomPlotConfig (centralized styling)
├── core.py               # FvcomPlotter (main engine)
├── timeseries.py         # Ensemble time series plots
├── dye_timeseries.py     # Stacked area plots
├── markers.py            # Node marker display
├── boundaries.py         # Element boundary display
├── utils.py              # Animation utilities
├── plotly_utils.py       # Interactive plots (optional)
├── _timeseries_utils.py  # Internal utilities (not public)
└── ...
```

**Import Patterns:**
```python
# Core imports (always available)
from xfvcom import FvcomDataLoader, FvcomAnalyzer, FvcomPlotter
from xfvcom import FvcomPlotConfig, FvcomPlotOptions

# Ensemble analysis
from xfvcom.ensemble_analysis import extract_member_node_mapping

# Enhanced plotting
from xfvcom.plot import (
    plot_ensemble_timeseries,
    plot_dye_timeseries_stacked,
    make_node_marker_post
)

# Utilities
from xfvcom.utils.timeseries_utils import extend_timeseries_seasonal
```

### FvcomPlotConfig Usage Pattern

**Modern approach (preferred):**
```python
# Create config once
cfg = FvcomPlotConfig(
    figsize=(12, 6),
    fontsize={
        "xticks": 13,
        "yticks": 13,
        "xlabel": 14,
        "ylabel": 14,
        "title": 15,
        "legend": 12,
    },
    linewidth={"plot": 1.8},
)

# Use across multiple plots for consistency
plotter = FvcomPlotter(ds, cfg)
fig = plotter.plot_timeseries(...)

# Or pass to standalone functions
fig, ax = plot_ensemble_timeseries(ds, cfg=cfg, ...)
result = plot_dye_timeseries_stacked(ds, cfg=cfg, ...)
```

**Legacy approach (still supported):**
```python
# Per-plot customization with FvcomPlotOptions
opts = FvcomPlotOptions(
    figsize=(10, 8),
    add_tiles=True,
    tile_provider="satellite"
)
fig = plotter.plot_2d(..., opts=opts)
```

---

## Common Tasks

### Adding New Plot Types

1. **Implement in FvcomPlotter** (`xfvcom/plot/core.py`):
   ```python
   def plot_new_type(self, var_name, **kwargs):
       """Plot description.

       Parameters
       ----------
       var_name : str
           Variable to plot
       **kwargs
           Additional options

       Returns
       -------
       Figure
           Matplotlib figure
       """
       fig, ax = plt.subplots(figsize=self.config.figsize)
       # Implementation...
       return fig
   ```

2. **Add test** (`tests/test_plot_new_type.py`):
   ```python
   @pytest.mark.png
   def test_new_plot_type(plotter, baseline_dir):
       fig = plotter.plot_new_type("temperature")
       # Baseline comparison...
   ```

3. **Update documentation**:
   - Add example to README.md
   - Document in appropriate docs/*.md file
   - Add to API reference table

### Adding Standalone Plotting Functions

For functions that don't need FvcomPlotter instance:

1. **Create in appropriate module** (`xfvcom/plot/timeseries.py` or new file)
2. **Accept FvcomPlotConfig** as optional parameter
3. **Export in `xfvcom/plot/__init__.py`**
4. **Add to `__all__`** in both module and package `__init__.py`

### Adding CLI Commands

1. **Create module** in `xfvcom/cli/` (e.g., `new_command.py`):
   ```python
   def main(args=None):
       parser = argparse.ArgumentParser(description="...")
       # Add arguments
       args = parser.parse_args(args)
       # Implementation
       return 0  # Success

   if __name__ == "__main__":
       sys.exit(main())
   ```

2. **Register in `pyproject.toml`**:
   ```toml
   [project.scripts]
   xfvcom-new-command = "xfvcom.cli.new_command:main"
   ```

3. **Test the CLI**:
   - Create `tests/test_new_command_cli.py`
   - Call `main()` with test arguments
   - Verify outputs

4. **Document in README.md**:
   - Add to "Command-Line Tools" section
   - Provide usage examples

### Modifying Force File Generators

**Structure:**
```
xfvcom/io/
├── base_generator.py              # BaseForceGenerator
├── river_nc_generator.py          # RiverNetCDFGenerator
├── river_nml_generator.py         # RiverNamelistGenerator
├── met_nc_generator.py            # MetNetCDFGenerator
└── groundwater_nc_generator.py    # GroundwaterNetCDFGenerator
```

**Key Points:**
1. Inherit from `BaseForceGenerator`
2. Use **netCDF4** directly (not xarray) for FVCOM compatibility
3. Default timezone: Asia/Tokyo → UTC conversion
4. Implement `write()` method
5. Support both constants and CSV time series
6. Handle FVCOM time formats: Modified Julian Day, Itime/Itime2, Times

**Example Pattern:**
```python
class NewForceGenerator(BaseForceGenerator):
    def __init__(self, grid_nc, start, end, dt_seconds, **kwargs):
        super().__init__(grid_nc, start, end, dt_seconds)
        # Initialize force-specific parameters

    def write(self, output_path):
        with nc.Dataset(output_path, 'w') as ds:
            # Create dimensions
            # Create variables
            # Write data
```

### Working with Ensemble Data

**Pattern for multi-member analysis:**
```python
from xfvcom.dye_timeseries import (
    Paths, DyeCase, Selection, NegPolicy, AlignPolicy,
    collect_member_files, aggregate
)

# Define case
paths = Paths(tb_fvcom_dir="/path/to/TB-FVCOM")
case = DyeCase(basename="tb_w18_r16", years=[2021], members=[1,2,3])
sel = Selection(nodes=[100], sigmas=[0])

# Collect and aggregate
member_map = collect_member_files(paths, case)
ds = aggregate(member_map, case, sel, NegPolicy("keep"), AlignPolicy("native_intersection"))

# Visualize
from xfvcom.plot import plot_ensemble_timeseries, plot_dye_timeseries_stacked

cfg = FvcomPlotConfig(figsize=(14, 6))
fig, ax = plot_ensemble_timeseries(ds, var_name="dye", cfg=cfg)
result = plot_dye_timeseries_stacked(ds, cfg=cfg, output="stacked.png")
```

### Documentation Maintenance

**File Organization:**
```
docs/
├── CONTRIBUTING.md              # Contribution guidelines
├── DOCUMENTATION_UPDATES.md     # Changelog for docs
├── forcing_generator.md         # Force file docs
├── plot_2d.md                   # 2D plotting guide
├── plot_section.md              # Vertical sections
├── plot_ts.md                   # Time series plotting
└── development/                 # Implementation notes (NOT user docs)
    ├── DIMENSION_TRANSPOSE_FIX.md
    ├── GROUNDWATER_FLUX_UNITS_CORRECTION.md
    └── ...
```

**When updating features:**
1. Update README.md with new feature description and example
2. Add entry to docs/DOCUMENTATION_UPDATES.md
3. Update appropriate docs/*.md file
4. Move implementation notes to docs/development/
5. Update this CLAUDE.md if architecture changes

---

## Important Notes

### Timezone Handling
- **Default input**: Asia/Tokyo (JST, UTC+9)
- **Default output**: UTC
- **Configurable**: Via `data_tz` parameter in generators
- **FVCOM convention**: Uses UTC for time variables

### Index Conventions
- **FVCOM**: 1-based indexing
- **Python/xarray**: 0-based indexing
- **Solution**: Most functions accept `index_base` parameter
  - `index_base=1`: FVCOM convention (default for grid operations)
  - `index_base=0`: Python convention

### Coordinate Systems
- **UTM**: Used for area calculations (meters)
- **Geographic**: Used for visualization (degrees)
- **Conversion**: Automatic via pyproj
- **Specify**: `utm_zone` parameter (e.g., 54 for Tokyo Bay)

### Area Calculations
Two methods available:
1. **Median-dual** (`calculate_node_area_median_dual`):
   - FVCOM-standard control volumes
   - Proper boundary closure
   - Preferred for accurate flux calculations

2. **Triangle sum** (`calculate_node_area`):
   - Legacy method
   - Sum of surrounding triangle areas
   - Simpler but less accurate at boundaries

### NaN Handling Philosophy
- **Ensemble plots**: Hard-fail on NaN (use `detect_nans_and_raise`)
  - Rationale: NaN indicates data pipeline issues
  - User should clean data before plotting
- **Other plots**: Graceful handling (skip or interpolate)

### Colormap Selection
- **Ensemble plots**: Automatic selection based on member count
  - ≤20 members: `tab20` (qualitative, distinct colors)
  - >20 members: `hsv` (continuous, evenly distributed)
- **Override**: Via `colormap` parameter
- **Custom**: Via `custom_colors` dict for specific members

---

## Troubleshooting

### Common Issues

**Issue**: Import errors after updating
```python
# Solution: Reinstall in editable mode
pip install -e .
```

**Issue**: PNG tests failing after matplotlib update
```bash
# Solution: Regenerate baselines
pytest --regenerate-baseline -q
```

**Issue**: Cartopy text clipping on geographic plots
```python
# Solution: Use make_node_marker_post with text_clip_buffer
pp = make_node_marker_post(..., text_clip_buffer=-0.001)
```

**Issue**: Force file rejected by FVCOM
```bash
# Check: Ensure netCDF4 (not xarray) was used for writing
# Check: Verify FVCOM time format (MJD, Itime/Itime2, Times)
ncdump -h force_file.nc  # Inspect structure
```

**Issue**: Slow nearest neighbor search
```python
# Solution: FvcomAnalyzer builds KDTree once, reuse instance
analyzer = FvcomAnalyzer(ds)  # Build KDTree
node1 = analyzer.nearest_neighbor(lon1, lat1)  # Fast
node2 = analyzer.nearest_neighbor(lon2, lat2)  # Fast (reuses tree)
```

### Getting Help
- Check README.md for usage examples
- Review tests/ for working code patterns
- Check docs/ for detailed guides
- Look for similar existing functionality before adding new

---

## Version History

**v0.2.0** (Current)
- Ensemble analysis module
- Dye time series aggregation
- Stacked area plots
- FvcomPlotConfig integration
- Enhanced node visualization
- Documentation reorganization

**v0.1.x**
- Core I/O and analysis
- Basic plotting
- Force file generators
- CLI tools
