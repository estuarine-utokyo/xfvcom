# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

xfvcom is a Python package for preprocessing and postprocessing FVCOM (Finite Volume Community Ocean Model) data. Built on xarray, it handles ocean model I/O, analysis, ensemble processing, and visualization.

## Environment Setup

```bash
# Activate existing environment
conda activate xfvcom

# Verify
python -c "import xfvcom; print(xfvcom.__version__)"
```

**Note**: Use `mamba` for installation to avoid conflicts with Intel oneAPI Python on supercomputers.

## Essential Commands

### CI-equivalent checks (run before committing)
```bash
black --check .
isort --check-only .
mypy .
pytest -m "not png"
```

**Quick one-liner to run all checks:**
```bash
python -m black --check . && python -m isort --check-only . && python -m mypy .
```

**Quick one-liner to auto-fix formatting (then manually fix mypy):**
```bash
python -m black . && python -m isort . && python -m mypy .
```

### Common mypy fixes
- Add type annotations to variables: `count: NDArray = np.zeros(...)`
- Use `| None` for optional types: `self.input_dir: Path | None = None`
- Use distinct variable names to avoid type redefinition within same scope
- Add assertions for properties that build internal state: `assert self._data is not None`
- Use `list()` copy to satisfy list invariance: `layers: list[int | None] = list(sigmas)`
- Cast pandas values explicitly: `node_id = int(row["column"])`
- Handle `None` from `importlib.util.spec_from_file_location()`: check `if spec is None`

### Development
```bash
# Format code
black . && isort .

# Run all tests (including PNG regression)
pytest

# Regenerate baseline images after intentional visual changes
pytest --regenerate-baseline -q

# Run specific test
pytest tests/test_grid.py::TestGridCalculations -v
```

**Note**: Ruff is NOT run in CI, so ruff errors are informational only.

## Architecture

### Data Flow
```
Input Files → FvcomDataLoader → xarray Dataset → FvcomAnalyzer → Results
                                      ↓
                                FvcomPlotter + FvcomPlotConfig → Plots

Grid File → FvcomInputLoader → Area Calculations / Mesh Connectivity

GWO CSV → GWOReader → GapFiller → MetNetCDFGenerator → met.nc
```

### Core Modules

| Module | Key Classes | Purpose |
|--------|-------------|---------|
| `xfvcom/io/` | `FvcomDataLoader`, `FvcomInputLoader` | Load FVCOM NetCDF and grid files |
| `xfvcom/io/*_generator.py` | `MetNetCDFGenerator`, `RiverNetCDFGenerator` | Generate FVCOM forcing files |
| `xfvcom/io/gwo_*.py` | `GWOReader`, `GapFiller`, `StationCorrelations` | JMA meteorological data processing |
| `xfvcom/analysis.py` | `FvcomAnalyzer` | KDTree search, layer averages, tidal decomposition |
| `xfvcom/plot/` | `FvcomPlotter`, `FvcomPlotConfig` | Visualization engine and styling |
| `xfvcom/dye_timeseries.py` | `DyeCase`, `Selection`, `aggregate()` | Multi-member ensemble aggregation |
| `xfvcom/grid/` | `FvcomGrid` | Area calculations (median-dual, triangle sum) |
| `xfvcom/cli/` | CLI entry points | Command-line tools (`xfvcom-*`) |

### Force File Generators

**Critical**: Use `netCDF4` directly (not xarray) for FVCOM compatibility. All generators inherit from `BaseForceGenerator`.

## Key Conventions

### Index Conventions
- **FVCOM**: 1-based indexing
- **Python/xarray**: 0-based indexing
- Most functions accept `index_base` parameter (default varies by context)

### Timezone Handling
- **Default**: Asia/Tokyo (JST) input → UTC output
- **GWO-AMD mode exception**: JST values stored directly but labeled as UTC (Tokyo Bay convention)

### Coordinate Systems
- **UTM**: For area calculations (meters) — specify `utm_zone` (e.g., 54 for Tokyo Bay)
- **Geographic**: For visualization (degrees)

### Area Calculations
- `calculate_node_area_median_dual()`: FVCOM-standard control volumes (preferred)
- `calculate_node_area()`: Legacy triangle sum method

### NaN Handling
- **Ensemble plots**: Hard-fail on NaN (design choice — user must clean data first)
- **Other plots**: Graceful handling

## Testing

### Key Fixtures (from `tests/conftest.py`)
- `fvcom_ds`: Minimal 3-D FVCOM-like xarray Dataset
- `plotter`: FvcomPlotter instance bound to fvcom_ds
- `regen_baseline`: True when `--regenerate-baseline` flag set

### PNG Regression Tests
- Marked with `@pytest.mark.png`
- Skipped in CI (baseline images in `tests/baseline/`)
- Regenerate after matplotlib/cartopy updates or intentional visual changes

## CLI Tools

Registered in `pyproject.toml` under `[project.scripts]`:
- `xfvcom-make-met-nc`: Meteorological forcing (supports GWO-AMD with gap filling)
- `xfvcom-make-river-nc`, `xfvcom-make-river-nml`: River forcing
- `xfvcom-make-groundwater-nc`: Groundwater forcing
- `xfvcom-check-met`: Validate forcing files
- `xfvcom-calc-gwo-corr`: Compute station correlations
- `xfvcom-dye-ts`: Dye time series extraction

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors after updating | `pip install -e .` |
| PNG tests failing after matplotlib update | `pytest --regenerate-baseline -q` |
| Force file rejected by FVCOM | Ensure netCDF4 (not xarray) was used; check time format |
| Cartopy text clipping | Use `make_node_marker_post(..., text_clip_buffer=-0.001)` |
| Slow nearest neighbor search | Reuse `FvcomAnalyzer` instance (KDTree built once) |
