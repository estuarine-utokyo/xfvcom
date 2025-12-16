# Forcing File Generation

[Back to README](../README.md)

## Overview

| Generator | CLI Command | Output |
|-----------|-------------|--------|
| River | `xfvcom-make-river-nc` | River discharge, temperature, salinity |
| Meteorological | `xfvcom-make-met-nc` | Wind, radiation, pressure, humidity |
| Groundwater | `xfvcom-make-groundwater-nc` | Groundwater flux, temperature, salinity, dye |

---

## River Forcing

### Generate Namelist

```bash
xfvcom-make-river-nml river_data.csv --output rivers.nml
```

### Generate NetCDF

```bash
xfvcom-make-river-nc rivers.nml \
  --start 2025-01-01T00:00Z \
  --end 2025-12-31T23:00Z \
  --dt 3600 \
  --ts Arakawa=discharge.csv:flux \
  --const Arakawa.salt=0.05
```

### Python API

```python
from xfvcom.io.river_nc_generator import RiverNetCDFGenerator

gen = RiverNetCDFGenerator(
    nml_path="rivers.nml",
    start="2025-01-01T00:00Z",
    end="2025-12-31T23:00Z",
    dt_seconds=3600,
    ts_specs=["Arakawa=discharge.csv:flux"],
    const_specs=["Arakawa.salt=0.05"],
)
gen.write("river_forcing.nc")
```

---

## Meteorological Forcing

### CLI (Constant/CSV Mode)

```bash
xfvcom-make-met-nc grid.nc \
  --start 2025-01-01T00:00Z \
  --end 2025-01-07T00:00Z \
  --utm-zone 54 \
  --ts wind.csv:uwind,vwind \
  --air-temperature 20.0 \
  --humidity 0.7
```

### CLI (GWO-AMD Mode)

Generate meteorological forcing from JMA Ground Weather Observation (GWO-AMD) data:

```bash
xfvcom-make-met-nc grid.dat --start 2020 \
  --gwo-dir /path/to/GWO/Hourly \
  --station-map "slht:Tokyo,kous:Tokyo,clod:Tokyo,*:Chiba" \
  --wind-factor 1.8 \
  --utm-zone 54 \
  -o met_forcing.nc
```

#### GWO-AMD Options

| Option | Description | Default |
|--------|-------------|---------|
| `--gwo-dir` | Base directory for GWO hourly data | (required) |
| `--station-map` | Variable-to-station mapping | `slht:Tokyo,kous:Tokyo,clod:Tokyo,*:Chiba` |
| `--wind-factor` | Wind speed multiplier | `1.8` |
| `--max-gap-hours` | Maximum gap for interpolation | `6` |

#### Station Mapping

Map specific variables to different stations:

```
slht:Tokyo,kous:Tokyo,clod:Tokyo,*:Chiba
```

- `slht` (short-wave radiation) → Tokyo
- `kous` (precipitation) → Tokyo
- `clod` (cloud cover) → Tokyo
- `*` (all others) → Chiba

#### Shell Script

A convenience script is available:

```bash
# Generate for year 2020
./scripts/make_fvcom_met.sh 2020

# With custom settings
GWO_DIR=/path/to/GWO/Hourly \
GRID_FILE=/path/to/grid.dat \
OUTPUT_DIR=/path/to/output \
./scripts/make_fvcom_met.sh 2020
```

### Variables

| Variable | Option | Units |
|----------|--------|-------|
| `uwind_speed` | `--uwind` | m/s |
| `vwind_speed` | `--vwind` | m/s |
| `air_temperature` | `--air-temperature` | C |
| `relative_humidity` | `--humidity` | fraction |
| `short_wave` | `--swrad` | W/m2 |
| `long_wave` | `--lwrad` | W/m2 |
| `air_pressure` | `--pressure` | hPa |
| `cloud_cover` | `--cloud` | fraction |
| `precipitation` | `--precip` | m/s |

### GWO-AMD Variable Conversions

| GWO Column | FVCOM Variable | Conversion |
|------------|----------------|------------|
| `kion` | `air_temperature` | × 0.1 → °C |
| `rhum` | `relative_humidity` | × 0.01 → fraction |
| `shpa` | `air_pressure` | × 0.1 → hPa |
| `muki`, `sped` | `uwind_speed`, `vwind_speed` | Direction + speed → u,v components |
| `clod` | `cloud_cover` | × 0.1 → fraction |
| `slht` | `short_wave` | × 0.01 × 1e6 / 3600 → W/m² |
| `kous` | `precipitation` | × 0.1 × 0.001 / 3600 → m/s |
| (estimated) | `long_wave` | Brutsaert formula from T, RH, cloud |

### Timezone Handling

**Important**: FVCOM meteorological forcing files follow a specific timezone convention:

- **Time values**: Stored as JST (Japan Standard Time, UTC+9) values
- **Time labels**: Labeled as "UTC" in the NetCDF attributes
- **No conversion**: When using GWO-AMD mode, data is NOT converted from JST to UTC

This convention means that a time value of `2020-01-01 09:00` in the NetCDF file represents 9:00 AM JST, even though it may be labeled as UTC. This is the standard practice for Tokyo Bay FVCOM simulations.

When using CSV time series (`--ts` option), the input timezone is controlled by `--data-tz` (default: `Asia/Tokyo`) and data is converted to UTC. For GWO-AMD mode, this conversion is skipped to maintain compatibility with existing FVCOM setups.

### Gap Filling

GWO-AMD data often contains missing values due to equipment failures or maintenance. The `--fill-gaps` option enables comprehensive gap filling using a 4-step process:

```bash
xfvcom-make-met-nc grid.dat --start 2020 \
  --gwo-dir /path/to/GWO/Hourly \
  --station-map "*:Chiba" \
  --fill-gaps \
  --fallback-stations "Chiba:Tokyo,Yokohama,Tateyama" \
  --correlation-file gwo_correlations.yaml \
  --solar-model empirical \
  -o met_forcing.nc
```

#### Gap Filling Steps

| Step | Method | Description |
|------|--------|-------------|
| 1 | Temporal Interpolation | Linear interpolation for gaps ≤ max_gap_hours (default: 6) |
| 2 | Boundary Interpolation | Uses adjacent year data for start/end gaps |
| 3 | Fallback Station | Uses correlated station data with linear conversion |
| 4 | Solar Estimation | pvlib-based estimation for solar radiation (slht only) |

#### Gap Filling Options

| Option | Description | Default |
|--------|-------------|---------|
| `--fill-gaps` | Enable 4-step gap filling | disabled |
| `--fallback-stations` | Station fallback chain (e.g., "Chiba:Tokyo,Yokohama") | none |
| `--correlation-file` | YAML file with pre-computed correlations | none |
| `--solar-model` | Solar estimation model: `empirical`, `pvlib-kasten`, `pvlib-larson` | `empirical` |
| `--no-extend-boundary` | Disable boundary interpolation (step 2) | enabled |
| `--strict` | Fail if any gaps remain after filling | disabled |

#### Computing Station Correlations

Before using fallback station gap filling, compute correlations between stations:

```bash
xfvcom-calc-gwo-corr --gwo-dir /path/to/GWO/Hourly \
  --stations Tokyo,Chiba,Yokohama,Tateyama \
  --years 2015-2020 \
  --include-solar-model \
  -o gwo_correlations.yaml
```

This generates a YAML file containing:
- Linear regression parameters (slope, intercept, R², RMSE) for each station pair and variable
- Solar model calibration parameters (if `--include-solar-model` is specified)

#### Solar Model Comparison

Compare different solar estimation models:

```bash
xfvcom-calc-gwo-corr --gwo-dir /path/to/GWO/Hourly \
  --stations Tokyo \
  --compare-solar-models \
  --train-years 2015-2019 \
  --test-year 2020 \
  -o gwo_correlations.yaml
```

Output includes R², RMSE, MAE, and bias for each model.

#### Gap Fill Report

When `--fill-gaps` is enabled, a detailed report is printed:

```
================================================================================
Gap Filling Report
================================================================================

[slht] Chiba - 24 gaps detected
  +-- Step 1 (Temporal): 12 filled
  +-- Step 2 (Boundary): 4 filled
  +-- Step 3 (Fallback Tokyo): 6 filled
  +-- Step 4 (Solar model): 2 filled
  `-- Remaining: 0

================================================================================
Fill Summary
================================================================================
Variable     Total    Step1    Step2    Step3    Step4   Remain     Rate
--------------------------------------------------------------------------------
slht            24        12        4        6        2        0   100.0%
--------------------------------------------------------------------------------
TOTAL           24        12        4        6        2        0   100.0%

[OK] All gaps filled successfully.
```

### Python API for Gap Filling

```python
from xfvcom.io import (
    GWOReader,
    GapFiller,
    StationCorrelations,
    SolarEstimator,
)

# Load correlation data
correlations = StationCorrelations.from_yaml("gwo_correlations.yaml")

# Create gap filler
reader = GWOReader("/path/to/GWO/Hourly")
filler = GapFiller(
    gwo_reader=reader,
    fallback_stations={"Chiba": ["Tokyo", "Yokohama", "Tateyama"]},
    correlations=correlations,
    max_gap_hours=6,
    solar_model="empirical",
)

# Fill gaps
df = reader.load_station_year("Chiba", 2020)
df_filled, results = filler.fill_all(
    df,
    station="Chiba",
    start=datetime(2020, 1, 1),
    end=datetime(2020, 12, 31, 23),
)

# Print report
from xfvcom.io import print_gap_fill_report
print_gap_fill_report(results)
```

---

## Groundwater Forcing

### Constant Values

```bash
xfvcom-make-groundwater-nc grid.nc \
  --start 2025-01-01T00:00Z \
  --end 2025-12-31T23:00Z \
  --flux 0.001 \
  --temperature 15.0 \
  --salinity 0.0
```

### With Dye Tracer

```bash
xfvcom-make-groundwater-nc grid.nc \
  --start 2025-01-01T00:00Z \
  --end 2025-12-31T23:00Z \
  --flux groundwater.csv:datetime,node_id,flux \
  --dye-concentration 1.0
```

---

## Time Series Format

CSV/TSV with `time` column (ISO 8601):

```csv
time,flux
2025-01-01T00:00Z,100
2025-01-01T06:00Z,105
2025-01-01T12:00Z,110
```

- Delimiter: auto-detected
- Encoding: auto-detected (UTF-8 recommended)
- Interpolation: linear only
- Timezone: `Asia/Tokyo` by default (`--data-tz` to override)

---

## Validation

Check forcing files for errors:

```bash
xfvcom-check-met met_forcing.nc
xfvcom-check-met met_forcing.nc -o anomalies.csv
```

[Back to README](../README.md)
