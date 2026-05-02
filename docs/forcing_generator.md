# Forcing File Generation

[Back to README](../README.md)

## Overview

| Generator | CLI Command | Output |
|-----------|-------------|--------|
| Meteorological | `xfvcom-make-met-nc` | Wind, radiation, pressure, humidity |
| River | `xfvcom-make-river-nc` | River discharge, temperature, salinity |
| Groundwater | `xfvcom-make-groundwater-nc` | Groundwater flux, temperature, salinity, dye |

---

## Meteorological Forcing

### Direct CLI Usage

```bash
# Constant values
xfvcom-make-met-nc grid.nc --start 2025-01-01T00:00Z --end 2025-01-07T00:00Z \
  --ts wind.csv:uwind,vwind --air_temperature 20.0 --utm-zone 54

# GWO-AMD source
xfvcom-make-met-nc grid.dat --start 2020 --gwo-dir /path/to/GWO/Hourly \
  --station-map "slht:Tokyo,kous:Tokyo,clod:Tokyo,*:Chiba" --wind-factor 1.8 --utm-zone 54

# GWO-AMD with gap filling
xfvcom-make-met-nc grid.dat --start 2020 --gwo-dir /path/to/GWO/Hourly \
  --station-map "*:Chiba" --fill-gaps --fallback-stations "Chiba:Tokyo,Yokohama" \
  --correlation-file gwo_correlations.yaml --solar-model empirical --utm-zone 54

# MPOS spatial wind for Tokyo Bay (3-station IDW)
xfvcom-make-met-nc grid.nc --start 2020-06-01 --end 2020-06-08 \
  --mpos-dir "$DATA_DIR/MPOS/nc_meteo" --utm-zone 54
```

### MPOS spatial wind (Tokyo Bay)

When `--mpos-dir` points to a directory of `Mpos_K_{stn}_{year}.nc` files
produced by `xmpos.preprocess.mpos2nc_meteo`, the generator switches to
spatial mode for the wind and air-temperature variables. Each FVCOM
element (`uwind_speed`, `vwind_speed`) and node (`air_temperature`)
receives its own time series, computed by inverse-distance weighting
from the active MPOS stations.

Defaults:

- **Stations used**: `01kemigawa`, `03urayasu`, `04chiba1gou`. The
  Kawasaki Artificial Island station (`02kawasaki`) is **excluded by
  default** because its anemometer is shielded by the adjacent "Wind
  Tower" structure to its NE; per MLIT `revise.pdf` (section 5),
  northeasterly wind speeds at this site are systematically attenuated
  and not corrected upstream. Override with
  `--mpos-stations 01kemigawa,02kawasaki,03urayasu,04chiba1gou` when an
  unfiltered field is needed.
- **IDW power**: 2 (override with `--mpos-idw-power`).
- **Time zone**: MPOS files are stored in JST; the generator converts
  to UTC during load, matching the FVCOM forcing convention. Pass
  `--mpos-keep-jst` to skip the shift (rarely needed).
- **Variables not observed by MPOS** (`rh`, `prmsl`, `swrad`, `lwrad`,
  `precip`, `cloud`) fall back to constants or `--ts` overrides, exactly
  as in legacy mode. Atmospheric pressure is **not** observed by MPOS
  and must be supplied externally if required (`metdata` package or
  ERA5).

`--mpos-dir` and `--gwo-dir` are mutually exclusive.

### Shell Script Wrapper

The script `scripts/make_fvcom_met.sh` provides a convenient wrapper with configurable defaults:

```bash
./scripts/make_fvcom_met.sh --start 2019
./scripts/make_fvcom_met.sh --start 2020 --output tb20_met.nc
./scripts/make_fvcom_met.sh --start 2020-06-01 --end 2020-08-31
./scripts/make_fvcom_met.sh --help
```

### CLI Options

#### General

| Option | Description | Default |
|--------|-------------|---------|
| `grid` | FVCOM grid file (`.dat` or `.nc`) — positional | required |
| `--start` | Start time: year (`2020`), date, or ISO datetime | required |
| `--end` | End time (optional for year/date-only start) | - |
| `--start-tz` | Timezone for naive start/end | `UTC` |
| `--dt` | Time step in seconds | `3600` |
| `--utm-zone` | UTM zone number (e.g., 54 for Tokyo Bay) | required |
| `--southern` | Southern hemisphere UTM | - |
| `-o`, `--output` | Output NetCDF file path | - |

#### Time Series Source

| Option | Description | Default |
|--------|-------------|---------|
| `--ts SPEC` | CSV/TSV time-series `path[:var1,var2,…]` (repeatable) | - |
| `--data-tz` | Timezone of input data | `Asia/Tokyo` |
| `--air_temperature`, etc. | Constant value for each met variable | (built-in defaults) |

#### GWO-AMD Options

| Option | Description | Default |
|--------|-------------|---------|
| `--gwo-dir` | GWO hourly data directory (enables GWO mode) | - |
| `--station-map` | Variable-to-station mapping | `slht:Tokyo,kous:Tokyo,clod:Tokyo,*:Chiba` |
| `--wind-factor` | Wind speed multiplier | `1.8` |
| `--max-gap-hours` | Max hours to interpolate for missing data | `6` |

#### Gap Filling (requires `--gwo-dir`)

| Option | Description | Default |
|--------|-------------|---------|
| `--fill-gaps` | Enable 4-step gap filling | `false` |
| `--fallback-stations` | Fallback chain (e.g., `"Chiba:Tokyo,Yokohama"`) | - |
| `--correlation-file` | YAML file with pre-computed correlations | - |
| `--solar-model` | `empirical`, `pvlib-kasten`, or `pvlib-larson` | `empirical` |
| `--no-extend-boundary` | Disable using prev/next year data for boundary gaps | - |
| `--strict` | Fail if gaps remain | `false` |
| `--recompute-correlations` | Force recompute even if cache exists | - |
| `--no-correlation-cache` | Disable auto-caching of correlations | - |

### Time Specification

| START | END | Period |
|-------|-----|--------|
| `2020` | (optional) | 2020-01-01T00:00 → 2021-01-01T00:00 |
| `2020` | `2022` | 2020-01-01T00:00 → 2023-01-01T00:00 |
| `2020-01-01` | `2020-01-07` | 2020-01-01T00:00 → 2020-01-08T00:00 |
| `2020-01-01T00:00:00` | `2020-01-07T00:00:00` | 2020-01-01T00:00 → 2020-01-07T00:00 |

Note: END is optional only for year format. For year/date formats, END is inclusive (adds 1 year/day).

### Station Mapping

Format: `variable:station,variable:station,*:default_station`

- `slht` (short-wave radiation), `kous` (precipitation), `clod` (cloud) → Tokyo
- `*` (all others: kion, rhum, shpa, sped, muki) → Chiba

### Gap Filling Steps

| Step | Method | Description |
|------|--------|-------------|
| 1 | Temporal | Linear interpolation for gaps ≤ max_gap_hours |
| 2 | Boundary | Uses adjacent year data for start/end gaps |
| 3 | Fallback | Uses correlated station with linear conversion |
| 4 | Solar | pvlib-based estimation (slht only) |

### Pre-compute Correlations (Optional)

```bash
xfvcom-calc-gwo-corr --gwo-dir /path/to/GWO/Hourly \
    --stations Tokyo,Chiba,Yokohama,Tateyama --years 2015-2020 \
    --include-solar-model -o gwo_correlations.yaml
```

Correlations are auto-cached when using `--fill-gaps` with `--fallback-stations`.

### GWO Variable Conversions

| GWO | FVCOM | Conversion |
|-----|-------|------------|
| kion | air_temperature | × 0.1 → °C |
| rhum | relative_humidity | × 0.01 → fraction |
| shpa | air_pressure | × 0.1 → hPa |
| muki, sped | uwind, vwind | Direction + speed → u,v |
| clod | cloud_cover | × 0.1 → fraction |
| slht | short_wave | × 0.01 × 1e6 / 3600 → W/m² |
| kous | precipitation | × 0.1 × 0.001 / 3600 → m/s |
| (estimated) | long_wave | Brutsaert formula |

---

## River Forcing

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `nml` | River namelist file — positional | required |
| `--start` | Start time (ISO-8601, UTC) | required |
| `--end` | End time (ISO-8601, UTC) | required |
| `--start-tz` | Timezone for naive start/end | `UTC` |
| `--dt` | Time step in seconds | `3600` |
| `--flux` | Default discharge | `0.0` |
| `--temp` | Default temperature | `20.0` |
| `--salt` | Default salinity | `0.0` |
| `--ts SPEC` | Time-series (`RIVER=path:var` or `path:var`, repeatable) | - |
| `--const SPEC` | Constant value (`RIVER.var=value` or `var=value`, repeatable) | - |
| `--config` | YAML config with defaults and river definitions | - |
| `--data-tz` | Timezone of CSV/TSV data | `Asia/Tokyo` |
| `-o`, `--output` | Output NetCDF file | - |

### Example

```bash
# Generate namelist first
xfvcom-make-river-nml river_data.csv --output rivers.nml

# Generate NetCDF
xfvcom-make-river-nc rivers.nml \
    --start 2025-01-01T00:00Z --end 2025-12-31T23:00Z --dt 3600 \
    --ts "Arakawa=discharge.csv:flux" --const "Arakawa.salt=0.05" \
    -o river_forcing.nc
```

---

## Groundwater Forcing

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `grid_file` | Grid file (`.nc` or `.dat`) — positional | required |
| `--start` | Start time (ISO-8601) | required |
| `--end` | End time (ISO-8601) | required |
| `--dt` | Time step in seconds | `3600` |
| `--start-tz` | Timezone for naive start/end | `UTC` |
| `--utm-zone` | UTM zone (required for `.dat` files) | - |
| `--southern` | Southern hemisphere | - |
| `--flux` | Flux velocity (m/s): constant or CSV | `0.0` |
| `--temperature` | Temperature (°C): constant or CSV | `0.0` |
| `--salinity` | Salinity (PSU): constant or CSV | `0.0` |
| `--dye` | Dye concentration: constant or CSV | - |
| `--ideal` | Use ideal time format instead of MJD | - |
| `--nml` | Also generate FVCOM namelist snippet | - |
| `-o`, `--output` | Output NetCDF file | `groundwater_forcing.nc` |

### Example

```bash
xfvcom-make-groundwater-nc grid.nc \
    --start 2025-01-01T00:00Z --end 2025-12-31T23:00Z \
    --flux 0.001 --temperature 15.0 --salinity 0.0 --dye 1.0 \
    -o groundwater_forcing.nc
```

---

## Time Series CSV Format

```csv
time,flux
2025-01-01T00:00Z,100
2025-01-01T06:00Z,105
```

- Delimiter/encoding: auto-detected
- Interpolation: linear
- Timezone: `Asia/Tokyo` (override with `--data-tz`)

---

## Validation

```bash
xfvcom-check-met met_forcing.nc                    # Check for NaN/Inf/out-of-bounds
xfvcom-check-met met_forcing.nc -o anomalies.csv   # Export to CSV
xfvcom-check-met met_forcing.nc --var uwind_speed   # Check specific variable
xfvcom-check-met met_forcing.nc --uniform           # Check spatial uniformity
```

[Back to README](../README.md)
