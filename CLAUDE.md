# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

xfvcom is a Python package for preprocessing and postprocessing data from the Finite Volume Community Ocean Model (FVCOM). It's built on xarray and specializes in ocean model data analysis and visualization.

## Environment Setup for Claude Code

### Quick Start
```bash
# Initialize conda in the shell (required for Claude Code)
source ~/miniforge3/etc/profile.d/conda.sh

# Activate the existing fvcom environment
conda activate fvcom

# Verify environment is active
python --version  # Should show Python 3.12.x
```

### Available Environments
- `fvcom` - Main development environment with xfvcom installed
- `xfvcom-docs` - Documentation building environment
- Other project-specific environments: `era_dl`, `moewq`, `pylag`, `river_dl`

## Essential Commands

### Development Setup (if creating new environment)
```bash
# Create and activate conda environment
conda create -n xfvcom python=3.12 -c conda-forge numpy xarray pandas matplotlib cartopy pyproj scipy scikit-learn imageio moviepy tqdm pytest mypy black isort jinja2 pyyaml types-pyyaml
conda activate xfvcom

# Install package in editable mode
pip install -e .
```

### Testing
```bash
# Run all tests
pytest

# Run tests excluding PNG regression tests
pytest -m "not png"

# Regenerate baseline images when plot appearance changes
pytest --regenerate-baseline -q

# Run specific test file
pytest tests/test_grid.py
```

### Code Quality
```bash
# Format code
black xfvcom/

# Sort imports
isort xfvcom/

# Type check
mypy .

# Run all pre-commit hooks
pre-commit run --all-files
```

### Building Documentation
```bash
cd docs
make html
```

## Architecture

### Core Components

1. **FvcomDataLoader** (xfvcom/io/core.py): Loads FVCOM NetCDF files and performs preprocessing
   - Handles coordinate transformations between UTM and geographic coordinates
   - Adds depth variables based on water levels, sigma layers, and bathymetry

2. **FvcomAnalyzer** (xfvcom/analysis.py): Physics-oriented calculations
   - Nearest neighbor search
   - Layer averages
   - Tidal decomposition
   - Variable filtering by dimensions

3. **FvcomPlotter** (xfvcom/plot/core.py): Main visualization engine
   - Time series plots
   - 2D contour and vector plots
   - Vertical section plots
   - Supports both static plots and animations (GIF/MP4)

4. **CLI Tools** (xfvcom/cli/):
   - `xfvcom-make-river-nc`: Generate river forcing NetCDF files
   - `xfvcom-make-river-nml`: Generate river namelist files
   - `xfvcom-make-met-nc`: Generate meteorological forcing NetCDF files
   - `xfvcom-make-groundwater-nc`: Generate groundwater forcing NetCDF files

### Data Flow

1. Data Loading: `FvcomDataLoader` → xarray Dataset with FVCOM mesh
2. Analysis: `FvcomAnalyzer` performs calculations on the Dataset
3. Visualization: `FvcomPlotter` with `FvcomPlotConfig` creates plots
4. Animation: Helper functions in `xfvcom.plot.utils` create GIF/MP4

### Key Dependencies

- **xarray**: Core data structure for multidimensional arrays
- **cartopy**: Geographic projections for maps
- **matplotlib**: Base plotting library
- **imageio/moviepy**: Animation creation
- **scipy/scikit-learn**: Scientific computations
- **netCDF4**: NetCDF file I/O

## Development Guidelines

### Code Style
- Line length: 88 characters (Black default)
- Import sorting: isort with Black-compatible profile
- Type hints: Optional but encouraged for public APIs
- Pre-commit hooks enforce formatting automatically

### Testing
- Tests use pytest with custom markers
- PNG regression tests compare plot outputs (marked with `@pytest.mark.png`)
- Test data located in `tests/data/` and baselines in `tests/baseline/`
- When plot appearance changes intentionally, regenerate baselines

### File Relationships
- **helpers.py**: High-level API for animations and batch plotting
- **utils.py**: Combines helpers.py and core.py for animation workflows
- **plot_options.py**: FvcomPlotOptions dataclass for styling
- **config.py**: FvcomPlotConfig with defaults (fonts, projections)
- **core.py**: FvcomPlotter implementation
- **analysis.py**: FvcomAnalyzer for physics calculations

## Common Tasks

### Adding New Plot Types
1. Implement method in `FvcomPlotter` (xfvcom/plot/core.py)
2. Add corresponding test in `tests/test_plot_*.py`
3. Update documentation with example usage

### Adding CLI Commands
1. Create module in `xfvcom/cli/`
2. Add entry point in `pyproject.toml` under `[project.scripts]`
3. Follow existing patterns (argument parsing, generator usage)

### Modifying Force File Generators
1. Base class: `xfvcom/io/base_generator.py`
2. Implementations: `met_nc_generator.py`, `river_nc_generator.py`, `river_nml_generator.py`, `groundwater_nc_generator.py`
3. Time zone handling: Default input timezone is Asia/Tokyo, output is UTC
4. Important: Use netCDF4 package directly for writing FVCOM-compliant NetCDF files (not xarray)