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
