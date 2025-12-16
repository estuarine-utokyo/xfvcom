# Documentation Updates

## December 2024 - GWO-AMD Gap Filling

### Summary

Added comprehensive gap filling support for GWO-AMD meteorological data, including station correlations, solar radiation estimation, and detailed reporting.

### New Features

1. **Gap Filling Module** (`xfvcom/io/gwo_gap_filler.py`)
   - `GapFiller`: 4-step gap filling process
     - Step 1: Temporal interpolation (short gaps ≤ max_gap_hours)
     - Step 2: Boundary interpolation (using adjacent year data)
     - Step 3: Fallback station interpolation (with correlation conversion)
     - Step 4: Solar radiation estimation (pvlib + cloud model)
   - `GapFillResult`, `MissingValueInfo`: Result dataclasses
   - `print_gap_fill_report()`, `print_missing_value_report()`: Detailed reporting

2. **Station Correlations** (`xfvcom/io/gwo_correlations.py`)
   - `StationCorrelations`: Compute/load correlation parameters from YAML
   - `CorrelationParams`: Linear regression parameters (slope, intercept, R², RMSE)
   - `get_station_coordinates()`: Station coordinate lookup for Tokyo Bay stations
   - Physical constraint definitions for validation

3. **Solar Estimation** (`xfvcom/io/solar_estimation.py`)
   - `SolarEstimator`: Clear-sky model with cloud attenuation (requires pvlib)
   - Models: `empirical`, `pvlib-kasten`, `pvlib-larson`
   - `build_empirical_model()`: Calibrate from GWO observations
   - `compare_models()`: Model validation and comparison

4. **CLI Updates** (`xfvcom-make-met-nc`)
   - `--fill-gaps`: Enable 4-step gap filling
   - `--fallback-stations`: Station fallback chain (e.g., "Chiba:Tokyo,Yokohama")
   - `--correlation-file`: Pre-computed correlation YAML file
   - `--solar-model`: Solar estimation model selection
   - `--no-extend-boundary`: Disable boundary interpolation
   - `--strict`: Fail if any gaps remain

5. **New CLI Tool** (`xfvcom-calc-gwo-corr`)
   - Compute correlations between GWO weather stations
   - Build and compare solar estimation models
   - Output to YAML for use with gap filling

### Bug Fixes

- Removed incorrect `fillna(0.0)` for solar radiation (slht) and precipitation (kous)
- These variables should retain NaN for missing values, not be zero-filled

### Files Updated

- `docs/forcing_generator.md` - Added gap filling section with examples
- `README.md` - Added new CLI commands and API classes
- `CLAUDE.md` - Added new modules to architecture, updated data flow diagram
- `pyproject.toml` - Added `pvlib>=0.10.0` dependency, new CLI script
- `environment.yml` - Added pvlib to conda dependencies
- `xfvcom/io/__init__.py` - Added exports for gap filling classes

---

## December 2024 - GWO-AMD Meteorological Forcing

### Summary

Added documentation for the new GWO-AMD (JMA Ground Weather Observation) meteorological data reader and FVCOM forcing generator.

### Changes Made

#### New Features Documented

1. **GWO-AMD Reader** (`xfvcom/io/gwo_reader.py`)
   - `GWOReader`: Read JMA GWO hourly CSV files (33-column format, no header)
   - `GWOForcingSource`: Data source adapter for MetNetCDFGenerator
   - Station mapping: Route variables to different observation stations
   - RMK code handling: Automatic masking and interpolation for missing data
   - Unit conversions: GWO raw values → FVCOM units
   - Wind conversion: Direction (16-point compass) + speed → u,v components
   - Long-wave radiation estimation: Brutsaert formula from T, RH, cloud

2. **Meteorological Forcing CLI** (`xfvcom-make-met-nc`)
   - New `--gwo-dir` option to enable GWO-AMD mode
   - `--station-map` for variable-to-station routing
   - `--wind-factor` for wind speed multiplier (default 1.8 for Tokyo Bay)
   - `--max-gap-hours` for interpolation gap limit

3. **Shell Script** (`scripts/make_fvcom_met.sh`)
   - Convenience wrapper for generating meteorological forcing
   - Configurable via environment variables

#### Timezone Handling (Important)

Documented the FVCOM timezone convention for Tokyo Bay simulations:
- GWO-AMD mode stores JST values labeled as "UTC" (no timezone conversion)
- This differs from CSV mode which converts from `data_tz` to UTC
- Rationale: Compatibility with existing FVCOM workflows

#### Files Updated

- `docs/forcing_generator.md` - Added GWO-AMD section with CLI examples, variable conversions, and timezone notes
- `README.md` - Added GWO-AMD CLI example and new API classes (`GWOReader`, `GWOForcingSource`)
- `CLAUDE.md` - Added GWO-AMD reader to architecture section, updated data flow diagram, expanded timezone handling notes

---

# Documentation Updates - October 2024

## Summary

This document tracks major updates to the xfvcom documentation, including README reorganization, feature additions, and cleanup of temporary files.

## Changes Made

### 1. README.md Reorganization

#### ✅ Added Features
- **Ensemble Analysis Module**: Documented the new `xfvcom.ensemble_analysis` subpackage with member-node mapping functionality
- **Enhanced Plotting**: Added documentation for:
  - `plot_ensemble_timeseries()` - Line plots with automatic colormap selection
  - `plot_ensemble_statistics()` - Statistical summaries for ensemble data
  - `plot_dye_timeseries_stacked()` - Stacked area plots with FvcomPlotConfig support
- **Dye Time Series CLI**: Documented `xfvcom-dye-ts` command-line tool
- **FvcomPlotConfig Integration**: Updated examples to show centralized plot styling
- **Automatic Colormap Selection**: Documented tab20 (≤20 members) vs hsv (>20 members) behavior

#### 🔄 Updated Sections
- **Installation**: Simplified to single method (removed redundant options)
- **Quick Start**: Reorganized into logical sections:
  - Load and Analyze Data
  - Area Calculations
  - Create Visualizations
  - Ensemble Time Series Analysis
  - Create Animations
- **API Reference**: Reorganized into categories:
  - Core Classes
  - Ensemble Analysis
  - Dye Time Series
  - Plotting Functions
  - Utility Functions
- **Testing**: Added specific examples for common test scenarios

#### ❌ Removed Outdated Content
- Removed `setup.sh` installation option (doesn't exist)
- Removed `environment.yml` installation option (doesn't exist)
- Removed programmatic forcing generation examples that duplicated CLI examples
- Simplified "Advanced Time Series Processing" examples to avoid redundancy

### 2. File Organization

#### Created Structure
```
docs/
├── development/              # NEW: Implementation and design notes
│   ├── DIMENSION_TRANSPOSE_FIX.md
│   ├── FINAL_FIX_REPORT.md
│   ├── GROUNDWATER_FLUX_UNITS_CORRECTION.md
│   ├── IMPLEMENTATION_COMPLETE.md
│   ├── IMPLEMENTATION_SUMMARY.md (from root)
│   ├── MEDIAN_DUAL_IMPLEMENTATION_REPORT.md
│   ├── SIMPLE_STACKED_PLOT.md
│   └── VERIFICATION_COMPLETE.md
├── CONTRIBUTING.md
├── forcing_generator.md
├── plot_2d.md
├── plot_section.md
└── plot_ts.md
```

#### Moved Files
**From repository root → `docs/development/`:**
- `DIMENSION_TRANSPOSE_FIX.md`
- `FINAL_FIX_REPORT.md`
- `IMPLEMENTATION_COMPLETE.md`
- `IMPLEMENTATION_SUMMARY.md`
- `SIMPLE_STACKED_PLOT.md`
- `VERIFICATION_COMPLETE.md`

**From `docs/` → `docs/development/`:**
- `GROUNDWATER_FLUX_UNITS_CORRECTION.md`
- `MEDIAN_DUAL_IMPLEMENTATION_REPORT.md`

#### Files Kept in Root
- `README.md` - Main documentation
- `CLAUDE.md` - Claude Code assistant instructions
- `AGENTS.md` - Agent configuration

### 3. Documentation Improvements

#### Better Organization
- **Logical Grouping**: Related features grouped together (e.g., all ensemble analysis in one section)
- **Progressive Complexity**: Simple examples first, advanced features later
- **Clear API Boundaries**: Separated public API from internal implementation details

#### Enhanced Examples
- **Ensemble Time Series**: Added complete workflow from data loading to visualization
- **Area Calculations**: Clarified median-dual vs triangle sum methods
- **FvcomPlotConfig Usage**: Consistent styling across all plot examples
- **CLI Tools**: Added `xfvcom-dye-ts` with practical examples

#### Improved Accuracy
- Removed references to non-existent installation methods
- Updated all code examples to match current API
- Corrected module paths (e.g., `xfvcom.plot` instead of `xfvcom.plot.plotly_utils`)
- Added version number to citation (0.2.0)

### 4. API Reference Updates

#### New Documented APIs

**Ensemble Analysis** (xfvcom.ensemble_analysis):
```python
- extract_member_node_mapping()
- get_member_summary()
- export_member_mapping()
- get_node_coordinates()
```

**Dye Time Series** (xfvcom.dye_timeseries):
```python
- collect_member_files()
- aggregate()
- negative_stats()
- verify_linearity()
- DyeCase, Selection, Paths, NegPolicy, AlignPolicy
```

**Enhanced Plotting** (xfvcom.plot):
```python
- plot_ensemble_timeseries()
- plot_ensemble_statistics()
- plot_dye_timeseries_stacked()
- get_member_color()
- get_member_colors()
```

#### Updated Documentation
- `FvcomPlotConfig`: Emphasized centralized styling approach
- `FvcomPlotter`: Updated with current plot methods
- Area calculation methods: Clarified median-dual vs triangle sum

### 5. Best Practices Applied

#### Documentation Structure
- ✅ **Scannable**: Used emoji markers, clear headings, horizontal rules
- ✅ **Progressive Disclosure**: Quick Start → Examples → API Reference
- ✅ **Task-Oriented**: Organized by what users want to accomplish
- ✅ **Code-Heavy**: More code examples, less prose

#### Content Guidelines
- ✅ **Accurate**: All examples tested against current API
- ✅ **Complete**: Included imports, error handling where relevant
- ✅ **Practical**: Real-world use cases, not toy examples
- ✅ **Maintainable**: Clear separation of stable vs experimental features

#### File Organization
- ✅ **Clean Root**: Minimal files in repository root
- ✅ **Logical Grouping**: Related docs in subdirectories
- ✅ **Clear Naming**: Development notes clearly separated from user docs

## Impact

### For Users
- **Easier Onboarding**: Simplified installation, clearer Quick Start
- **Better Discovery**: Enhanced API reference with categorization
- **More Examples**: Ensemble analysis and advanced plotting workflows
- **Accurate Information**: Removed outdated/incorrect content

### For Developers
- **Organized Notes**: Implementation details in `docs/development/`
- **Clear API Documentation**: What's public vs internal
- **Contribution Guide**: Updated workflow and requirements
- **CI Clarity**: Documented what checks run in CI

### For Maintainers
- **Single Source of Truth**: README as central hub
- **Versioned**: Citation includes version number
- **Traceable**: This document provides audit trail of changes

## Next Steps

### Recommended Future Updates
1. **Tutorial Series**: Step-by-step guides for common workflows
2. **Gallery**: Visual showcase of plot types with code
3. **Performance Guide**: Best practices for large datasets
4. **Migration Guide**: From previous versions (if needed)
5. **Video Tutorials**: Screen recordings for complex workflows

### Documentation Maintenance
- Update README when adding new features
- Move implementation notes to `docs/development/`
- Keep API reference synchronized with `__all__` exports
- Test all code examples during releases

## Version Information

- **Updated**: October 16, 2024
- **xfvcom Version**: 0.2.0
- **Changes By**: Documentation Review and Reorganization
- **Related PRs**: TBD

---

## Checklist for Future Documentation Updates

When adding new features or updating documentation:

- [ ] Update README.md with new feature description
- [ ] Add code example to appropriate section
- [ ] Update API Reference table if adding public API
- [ ] Add entry to this changelog
- [ ] Test all code examples
- [ ] Update version number if releasing
- [ ] Move temporary notes to `docs/development/`
- [ ] Update CLAUDE.md if changing development workflow
