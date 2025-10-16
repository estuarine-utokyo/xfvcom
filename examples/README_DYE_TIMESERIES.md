# Multi-Year Dye Time-Series Extraction

A robust tool for extracting and analyzing dye concentration time series from multi-year FVCOM ensemble runs.

## Features

- **Multi-year, multi-member aggregation**: Extract dye data across multiple years and ensemble members
- **Spatial averaging**: Average over selected nodes and sigma layers
- **Negative value handling**: Detect, report, and optionally clip numerical undershoots
- **Time alignment modes**:
  - `native_intersection`: Strict timestamp intersection across all members
  - `same_calendar`: Align by calendar position (month, day, hour)
  - `climatology`: Average over years to produce climatological means
- **Linearity verification**: Check if reference member equals sum of parts
- **Multiple output formats**: NetCDF and Zarr support

## Installation

The tool is part of the `xfvcom` package. Ensure you have installed xfvcom:

```bash
cd ~/Github/xfvcom
pip install -e .
```

**Required**: Install zarr for both NetCDF and Zarr export support:
```bash
pip install zarr
```

Note: Zarr is required for the dye time series tools. The package supports exporting to both `.nc` (NetCDF) and `.zarr` formats.

## Usage

### 1. Command Line Interface (CLI)

Basic extraction:
```bash
python examples/dye_timeseries_cli.py \
  --years 2021 \
  --members 0 1 2 18 \
  --nodes 100,200,300 \
  --sigmas 0,1 \
  --save output/dye_series.nc
```

With negative statistics and linearity check:
```bash
python examples/dye_timeseries_cli.py \
  --years 2020 2021 2022 \
  --members 0 1 2 \
  --nodes 123,456 \
  --sigmas 0 \
  --neg-policy clip_zero \
  --align native_intersection \
  --print-neg-stats \
  --verify-linearity --ref-member 0 --parts 1 2 \
  --save output/dye_linearity.nc
```

Climatology mode:
```bash
python examples/dye_timeseries_cli.py \
  --years 2020 2021 2022 \
  --members 0 1 2 \
  --nodes 100 \
  --sigmas 0 1 2 \
  --align climatology \
  --save output/dye_climatology.zarr
```

#### CLI Options

- `--years`: List of years to process (required)
- `--members`: List of member IDs to process (required)
- `--nodes`: Node indices, comma or space separated (required)
- `--sigmas`: Sigma layer indices, 0-based (required)
- `--basename`: Base name of output files (default: `tb_w18_r16`)
- `--neg-policy`: `keep` or `clip_zero` (default: `keep`)
- `--align`: `native_intersection`, `same_calendar`, or `climatology` (default: `native_intersection`)
- `--tb-fvcom-dir`: Path to TB-FVCOM (default: `../../TB-FVCOM` or `$TB_FVCOM_DIR`)
- `--save`: Output file path (`.nc` or `.zarr`)
- `--verify-linearity`: Enable linearity check
- `--ref-member`: Reference member for linearity (default: 0)
- `--parts`: Part members for linearity (default: all except ref)
- `--print-neg-stats`: Print negative value statistics
- `--verbose`: Enable debug logging

### 2. Python API

```python
from pathlib import Path
from xfvcom.dye_timeseries import (
    DyeCase, Selection, Paths, NegPolicy, AlignPolicy,
    collect_member_files, aggregate, negative_stats, verify_linearity
)

# Configuration
paths = Paths(tb_fvcom_dir=Path("~/Github/TB-FVCOM"))
case = DyeCase(basename="tb_w18_r16", years=[2021, 2022], members=[0, 1, 2])
sel = Selection(nodes=[100, 200], sigmas=[0, 1])
neg = NegPolicy(mode="keep")
align = AlignPolicy(mode="native_intersection")

# Collect files
member_map = collect_member_files(paths, case)

# Aggregate
ds = aggregate(member_map, case, sel, neg, align)

# Analyze
stats = negative_stats(ds)
linearity = verify_linearity(ds, ref_member=0, parts=[1, 2])

# Save
ds.to_netcdf("output/dye_series.nc")
```

### 3. Jupyter Notebook

An interactive demo is available:
```bash
jupyter notebook examples/notebooks/dye_timeseries_demo.ipynb
```

The notebook demonstrates:
- Basic configuration and extraction
- Negative value analysis
- Linearity verification
- Time series visualization
- Statistical summaries
- Exporting to NetCDF and Zarr

## Data Structure

### Input Files

Expected file structure:
```
TB-FVCOM/
└── goto2023/dye_run/output/
    └── <YEAR>/
        └── <MEMBER>/
            └── tb_w18_r16_<YEAR>_<MEMBER>_*.nc
```

Example:
```
TB-FVCOM/goto2023/dye_run/output/2021/0/tb_w18_r16_2021_0_0001.nc
TB-FVCOM/goto2023/dye_run/output/2021/1/tb_w18_r16_2021_1_0001.nc
```

### Output Dataset

The aggregated dataset contains:
- **Dimensions**: `time`, `ensemble`
- **Coordinates**:
  - `time`: Time coordinate (FVCOM native encoding)
  - `ensemble`: MultiIndex with (year, member)
- **Data variables**:
  - `dye`: Averaged dye concentration (time, ensemble)
- **Attributes**: Configuration metadata, statistics, alignment info

## Error Handling

### NaN Detection

If NaN values are found in DYE data, the tool raises a detailed error:
```
ValueError: NaN values detected in DYE data for (year=2021, member=0)!
  Files: ['tb_w18_r16_2021_0_0001.nc']
  Total NaNs: 10
  First occurrences:
    [1] time_idx=5 (2021-01-01T05:00:00), node=100, sigma=0
    ...
  Remediation hints:
    - Check for dry nodes (land mask)
    - Verify DYE initialization and boundary conditions
    - Review upstream preprocessing steps
```

### Missing Files

If files are not found:
```
FileNotFoundError: No files found for 3 (year, member) pair(s):
  Missing: [(2021, 0), (2021, 1), (2021, 2)]
  Expected location: /path/to/TB-FVCOM/goto2023/dye_run/output/<YEAR>/<MEMBER>/
  Expected pattern: tb_w18_r16_<YEAR>_<MEMBER>_*.nc
  Hint: Check that outputs exist and paths are correct.
```

### Time Alignment Failure

If `native_intersection` yields empty results:
```
ValueError: Time alignment failed: empty intersection!
Time ranges for each (year, member):
  (year=2020, member=0): 2020-01-01 to 2020-12-31 (8760 steps)
  (year=2021, member=0): 2021-01-01 to 2021-12-31 (8760 steps)
Hint: Consider using align='same_calendar' or 'climatology' instead.
```

## Negative Value Analysis

Dye concentrations can have negative values due to numerical undershoots. The tool provides:

1. **Statistics before any clipping**: Count, minimum value, percentage
2. **Per-member breakdown**: Individual member statistics
3. **Global summary**: Across all members

Example output:
```json
{
  "per_member": {
    "2021_0": {
      "count_neg": 0,
      "min_value": 0.0,
      "share_neg": 0.0,
      "total_samples": 745
    },
    "2021_1": {
      "count_neg": 136,
      "min_value": -0.001310,
      "share_neg": 0.183,
      "total_samples": 745
    }
  },
  "global": {
    "min_value": -0.001310,
    "count_neg": 136,
    "share_neg": 0.091,
    "total_samples": 1490
  }
}
```

## Linearity Verification

For ensemble runs where member 0 should equal the sum of other members:

```python
linearity = verify_linearity(ds, ref_member=0, parts=[1, 2, 18])
```

Metrics:
- **RMSE**: Root mean square error
- **MAE**: Mean absolute error
- **Max |Δ|**: Maximum absolute difference
- **NSE**: Nash-Sutcliffe Efficiency (1.0 = perfect, <0 = poor)

Interpretation:
- NSE > 0.99: Excellent linearity
- NSE > 0.95: Good linearity
- NSE < 0.95: Poor linearity (check configurations)

## Time Alignment Modes

### native_intersection
Strict intersection of exact timestamps across all (year, member) pairs. Best for same-year, same-period comparisons.

### same_calendar
Group by (month, day, hour) across years. Allows cross-year comparison at the same calendar position. Handles leap years gracefully.

### climatology
Average over years by (month, day, hour) to produce climatological means. Output includes:
- `dye(month, day, hour, member)`: Per-member climatology
- `clim_mean(month, day, hour)`: Averaged over both years and members

## Performance Notes

- Single files are opened directly for efficiency
- Multiple files are concatenated along time dimension
- Use `--verbose` for detailed progress logging
- Memory usage scales with: (# time steps) × (# ensemble members)

## Troubleshooting

**Issue**: `decode_times=False` warnings
**Cause**: FVCOM uses non-standard time encoding ("msec since 00:00:00")
**Solution**: This is expected and handled internally

**Issue**: Member files not found
**Cause**: Incorrect TB-FVCOM path or file naming
**Solution**: Use `--tb-fvcom-dir` to specify correct path, or set `$TB_FVCOM_DIR`

**Issue**: Empty intersection in native mode
**Cause**: Time ranges don't overlap
**Solution**: Use `--align same_calendar` or `--align climatology`

## Examples

Test the installation:
```bash
python examples/test_dye_timeseries.py
```

Quick extraction for analysis:
```bash
python examples/dye_timeseries_cli.py \
  --years 2021 \
  --members 0 1 2 \
  --nodes 100 \
  --sigmas 0 \
  --save output/test.nc
```

## API Reference

See module docstrings:
```python
from xfvcom import dye_timeseries
help(dye_timeseries)
```

Key classes:
- `DyeCase`: Case configuration (basename, years, members)
- `Selection`: Spatial/vertical selection (nodes, sigmas)
- `Paths`: Path configuration
- `NegPolicy`: Negative value handling policy
- `AlignPolicy`: Time alignment policy

Key functions:
- `collect_member_files()`: Find and validate input files
- `load_member_series()`: Load single (year, member) series
- `aggregate()`: Combine all series with alignment
- `negative_stats()`: Compute negative value statistics
- `verify_linearity()`: Check linearity assumption

## Citation

If you use this tool in research, please cite:
```
xfvcom dye_timeseries module
https://github.com/estuarine-utokyo/xfvcom
```

## License

MIT License - see xfvcom package LICENSE file
