# Forcing File Generation

[Back to README](../README.md)

## Overview

| Generator | CLI Command | Output |
|-----------|-------------|--------|
| Meteorological | `xfvcom-make-met-nc` | Wind, radiation, pressure, humidity |
| River | `xfvcom-make-river-nc` | River discharge, temperature, salinity |
| Groundwater | `xfvcom-make-groundwater-nc` | Groundwater flux, temperature, salinity, dye |

---

## Meteorological Forcing (GWO-AMD)

### Quick Start with Command Line Arguments

The `make_fvcom_met.sh` script supports command line arguments for easy customization:

```bash
# Generate forcing for year 2019
./make_fvcom_met.sh --start 2019

# Generate forcing for year 2020 with custom output file
./make_fvcom_met.sh --start 2020 --output tb20_met.nc

# Generate forcing for a specific date range
./make_fvcom_met.sh --start 2020-06-01 --end 2020-08-31 --output summer2020.nc

# Use a different grid file
./make_fvcom_met.sh --start 2019 --grid /path/to/custom_grid.dat

# Disable gap filling
./make_fvcom_met.sh --start 2020 --no-fill-gaps

# Show all available options
./make_fvcom_met.sh --help
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--start YEAR` | Start year or datetime | 2020 |
| `--end YEAR` | End year or datetime | auto |
| `--grid FILE` | Grid file path | (see script) |
| `--output, -o FILE` | Output file | met_forcing.nc |
| `--utm-zone ZONE` | UTM zone | 54 |
| `--gwo-dir DIR` | GWO data directory | (see script) |
| `--station-map MAP` | Station mapping | slht:Tokyo,... |
| `--wind-factor F` | Wind factor | 1.8 |
| `--max-gap-hours N` | Max gap hours | 6 |
| `--fill-gaps` | Enable gap filling | (default) |
| `--no-fill-gaps` | Disable gap filling | - |
| `--fallback-stations` | Fallback stations | Chiba:Tokyo,... |
| `--solar-model MODEL` | Solar model | empirical |
| `--help, -h` | Show help | - |

### Configuration Priority

Settings can be specified in three ways (highest to lowest priority):

1. **Command line arguments**: `./make_fvcom_met.sh --start 2019`
2. **Environment variables**: `START=2019 ./make_fvcom_met.sh`
3. **Default values in script**: Edit the script directly

### Full Script Reference

The script `scripts/make_fvcom_met.sh` provides a complete template:

```bash
#!/bin/bash
# Default values can be edited directly in the script
GRID=~/Github/TB-FVCOM/goto2023/input/TokyoBay18_grd.dat
START=2020
END=
UTM_ZONE=54
OUTPUT=met_forcing.nc
GWO_DIR=${DATA_DIR}/met/JMA_DataBase/GWO/Hourly
STATION_MAP="slht:Tokyo,kous:Tokyo,clod:Tokyo,*:Chiba"
WIND_FACTOR=1.8
MAX_GAP_HOURS=6
FILL_GAPS=true
FALLBACK_STATIONS="Chiba:Tokyo,Yokohama,Tateyama"
SOLAR_MODEL=empirical

# ... (argument parsing and execution)
```

### Time Specification

| START | END | Period |
|-------|-----|--------|
| `2020` | (optional) | 2020-01-01T00:00 → 2021-01-01T00:00 |
| `2020` | `2022` | 2020-01-01T00:00 → 2023-01-01T00:00 |
| `2020-01-01` | `2020-01-07` | 2020-01-01T00:00 → 2020-01-08T00:00 |
| `2020-01-01T00:00:00` | `2020-01-07T00:00:00` | 2020-01-01T00:00 → 2020-01-07T00:00 |

Note: END is optional only for year format. For year/date formats, END is inclusive (adds 1 year/day).

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--start` | Start time (see above) | required |
| `--end` | End time (see above) | - |
| `--utm-zone` | UTM zone number (e.g., 54 for Tokyo Bay) | required |
| `--southern` | Southern hemisphere UTM | northern |
| `--gwo-dir` | GWO hourly data directory | - |
| `--station-map` | Variable-to-station mapping | `slht:Tokyo,kous:Tokyo,clod:Tokyo,*:Chiba` |
| `--wind-factor` | Wind speed multiplier | 1.8 |
| `--max-gap-hours` | Max hours to interpolate | 6 |
| `--fill-gaps` | Enable 4-step gap filling | false |
| `--fallback-stations` | Fallback chain (e.g., "Chiba:Tokyo,Yokohama") | - |
| `--solar-model` | Solar estimation: empirical, pvlib-kasten, pvlib-larson | empirical |
| `--strict` | Fail if gaps remain | false |

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
#!/bin/bash
GWO_DIR=/path/to/GWO/Hourly
STATIONS=Tokyo,Chiba,Yokohama,Tateyama
YEARS=2015-2020
OUTPUT=gwo_correlations.yaml

xfvcom-calc-gwo-corr --gwo-dir "$GWO_DIR" \
    --stations "$STATIONS" --years "$YEARS" \
    --include-solar-model -o "$OUTPUT"
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

```bash
#!/bin/bash
NML=rivers.nml
START=2025-01-01T00:00Z
END=2025-12-31T23:00Z
DT=3600
TS_SPEC="Arakawa=discharge.csv:flux"
CONST_SPEC="Arakawa.salt=0.05"
OUTPUT=river_forcing.nc

# Generate namelist first
xfvcom-make-river-nml river_data.csv --output "$NML"

# Generate NetCDF
xfvcom-make-river-nc "$NML" \
    --start "$START" --end "$END" --dt "$DT" \
    --ts "$TS_SPEC" --const "$CONST_SPEC" \
    -o "$OUTPUT"
```

---

## Groundwater Forcing

```bash
#!/bin/bash
GRID=grid.nc
START=2025-01-01T00:00Z
END=2025-12-31T23:00Z
FLUX=0.001
TEMPERATURE=15.0
SALINITY=0.0
DYE=1.0
OUTPUT=groundwater_forcing.nc

xfvcom-make-groundwater-nc "$GRID" \
    --start "$START" --end "$END" \
    --flux "$FLUX" --temperature "$TEMPERATURE" \
    --salinity "$SALINITY" --dye-concentration "$DYE" \
    -o "$OUTPUT"
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
xfvcom-check-met met_forcing.nc
xfvcom-check-met met_forcing.nc -o anomalies.csv
```

[Back to README](../README.md)
