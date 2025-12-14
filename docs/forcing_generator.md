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

### CLI

```bash
xfvcom-make-met-nc grid.nc \
  --start 2025-01-01T00:00Z \
  --end 2025-01-07T00:00Z \
  --utm-zone 54 \
  --ts wind.csv:uwind,vwind \
  --air-temperature 20.0 \
  --humidity 0.7
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
xf-check-met met_forcing.nc
xf-check-met met_forcing.nc -o anomalies.csv
```

[Back to README](../README.md)
