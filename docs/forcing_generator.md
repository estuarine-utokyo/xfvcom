# Forcing File Generation

[Back to README](../README.md)

## Overview

| Generator | CLI Command | Output |
|-----------|-------------|--------|
| River | `xfvcom-make-river-nc` | River discharge, temperature, salinity |
| Meteorological | `xfvcom-make-met-nc` | Wind, radiation, pressure, humidity |
| Groundwater | `xfvcom-make-groundwater-nc` | Groundwater flux, temperature, salinity, dye |

---

## Meteorological Forcing (GWO-AMD)

```bash
#!/bin/bash
GRID=grid.dat
YEAR=2020
UTM_ZONE=54
OUTPUT=met_forcing.nc
# GWO-AMD options
GWO_DIR=/path/to/GWO/Hourly
STATION_MAP="slht:Tokyo,kous:Tokyo,clod:Tokyo,*:Chiba"
WIND_FACTOR=1.8
MAX_GAP_HOURS=6
# Gap filling options
FILL_GAPS=true
FALLBACK_STATIONS="Chiba:Tokyo,Yokohama,Tateyama"
SOLAR_MODEL=empirical

xfvcom-make-met-nc "$GRID" --start "$YEAR" --utm-zone "$UTM_ZONE" \
    --gwo-dir "$GWO_DIR" --station-map "$STATION_MAP" \
    --wind-factor "$WIND_FACTOR" --max-gap-hours "$MAX_GAP_HOURS" \
    $($FILL_GAPS && echo "--fill-gaps") \
    --fallback-stations "$FALLBACK_STATIONS" \
    --solar-model "$SOLAR_MODEL" \
    -o "$OUTPUT"
```

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
