# Forcing File Generation

[Back to README](../README.md)

## Overview

| Generator | CLI Command | Output |
|-----------|-------------|--------|
| Meteorological | `xfvcom-make-met-nc` | Wind, radiation, pressure, humidity |
| River | `xfvcom-make-river-nc` | River discharge, temperature, salinity |
| River (from river_dl) | `xfvcom-make-river-nc-from-river-dl` | Same as above, sourced from per-river `discharge_hourly.nc` files |
| Rivers NML | `xfvcom-make-rivers-namelist` | `RIVERS_NAMELIST*.nml` (companion to the river_dl adapter) |
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

## Metforce 2D-OI Gridded Source

`xfvcom-make-met-nc-from-metforce` bilinearly interpolates the unified
metforce analysis `fvcom_forcing_<YEAR>.nc` (regular ~5 km MSM-S / ERA5
grid, eight variables on `(time, lat, lon)`) onto the FVCOM mesh and
writes a FVCOM-compliant atmospheric forcing NetCDF
(`uwind_speed`, `vwind_speed`, `air_temperature`, `relative_humidity`,
`air_pressure`, `short_wave`, `long_wave`, `Precipitation`, and a
constant `cloud_cover`).

### Filename convention (new_bc baseline)

The legacy `tb18_*.nc` files carried the `18` as an F04-era
**wind-scale × 1.8** suffix (Optuna-tuned). The 2026-05-16 new_bc
baseline runs at scale = 1.0 (default), so the FVCOM-mesh forcing
file emitted by this CLI drops the `18`:

| Era | Wind scale | File pattern |
|---|---|---|
| F04 (legacy production) | 1.8 | `tb18_wnd.nc` (single-point) |
| 2026-05-16 new_bc (post-incident, scale = 1.0) | 1.0 | **`tb_wnd_metforce_<year>.nc`** |

The transitional `tb18_wnd_metforce_2020.nc` (Job 5831777 input)
inherited the legacy prefix by accident and is the file that crashed
the 2026-05-16 new_bc baseline at simulation time
`2020-02-21T13:58:20` because it carried propagated `int16` sentinel
NaN from the MSM-S daily archive. Rebuilds **must** be emitted as
`tb_wnd_metforce_<year>.nc`; do not overwrite the legacy file.

### Metforce Path-B input

Source `~/Github/metforce/analysis/fvcom_forcing_<YEAR>.nc` is the
sentinel-clean output of metforce's Bayesian uninformative-prior OI
("Path B", shipped 2026-05-16). The six core OI variables
(`U10`, `V10`, `T2`, `SH`, `SLP`, `DSWRF`) plus the derived `DLWRF`
are NaN-free; `PRECIP` retains ~28 scattered NaN hours per year that
metforce does not gap-fill (it comes from the background field, not
OI). For new_bc the residual is acceptable because
`PRECIPITATION_ON = F` in the run NML. See
`~/Github/metforce/docs/msm_s_archive_validation.md` for the upstream
incident report and Path-B specification.

### 2020 rebuild wrapper

`scripts/job_build_tb_wnd_metforce_2020.pjsub` is the canonical
pjsub wrapper for the 2020 rebuild on GENKAI. It pins the source
NetCDF, the goto2023 grid, the UTM zone, and clamps the timeline to
the metforce source range (`--end 2020-12-31T23:00:00`, 8784 hours)
so the rebuild has no trailing-hour NaN. Output:

```
~/Github/TB-FVCOM/input/goto2023/forcing-fvcom-grid/2020/tb_wnd_metforce_2020.nc
```

### Known issue: time-axis `dt` drift

The generator currently writes the MJD `time` variable as `float32`.
At the 2020 epoch (MJD ≈ 58849) the `float32` ULP is ≈ 339 s, so a
nominally hourly cadence decodes as `dt ∈ {3375, 3712, 3713} s`
(mean 3600 s) — a Julian-day round-trip artefact rather than a real
timeline drift. The legacy file inherited the same drift. Until the
`time` variable is widened to `float64`, downstream consumers should
treat the cadence as nominally hourly.

### Direct CLI Usage

```bash
xfvcom-make-met-nc-from-metforce TokyoBay_grd.dat \
    --metforce-file ~/Github/metforce/analysis/fvcom_forcing_2020.nc \
    --start 2020-01-01T00:00:00 --end 2020-12-31T23:00:00 \
    --utm-zone 54 \
    -o tb_wnd_metforce_2020.nc
```

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

### From river_dl archives (`xfvcom-make-river-nc-from-river-dl`)

For projects that already maintain per-river `discharge_hourly.nc`
files under a `river_dl` archive (e.g. MLIT observation feeds), the
sibling CLI `xfvcom-make-river-nc-from-river-dl` consumes a
`--river-map` YAML that pins each FVCOM river name to one
`discharge_hourly.nc`:

```yaml
defaults:
  temp: 15.0
  salt: 0.0
rivers:
  - name: EastArakawa
    source: ${DATA_DIR}/river/discharge/Arakawa/Iwabuchi/discharge_hourly.nc
    scale: 0.25
  - name: Shibaura
    source: ${DATA_DIR}/wastewater/Shibaura/discharge_hourly.nc
    temp: 20.0
```

#### Constant sources

A YAML entry may declare `kind: constant` in place of `source:` to
represent a flux with no upstream NetCDF (e.g. a sewer plant whose
discharge is approximated by a fixed annual mean while real
observation data is unavailable). The `(flux, temp, salt)` tuple is
broadcast across the requested time axis:

```yaml
rivers:
  - name: Kisarazu
    kind: constant
    flux: 0.2902       # m^3/s, constant
    temp: 10.0
    salt: 0.0          # optional; default from defaults: block
```

Constant entries are rejected if they also carry `source:` (must be
one or the other) or `scale:` (the value is already in physical
units), or omit `flux:` (the only mandatory field).

#### River temperature (`temp_source`)

Each entry's water temperature is set by a required `temp_source`
mapping (schema v3 — a bare `temp:` scalar is rejected for annual
runs). Two kinds are supported:

- `monthly_climatology`: a 12-element `monthly_means` list (Jan..Dec),
  broadcast as a month-of-year step function.
- `air_regression`: `T_water = slope * T_air + intercept`, where
  `T_air` is read from a metforce NC (`air_nc_template`, `air_var`) at
  the nearest cell to (`air_lat`, `air_lon`). Optional `min_temp` /
  `max_temp` clip the result.

`air_regression` accepts two optional keys that smooth the air driver
so the river responds to an **antecedent** (time-integrated) air
temperature rather than the instantaneous value — rivers damp the
diurnal/synoptic air-temperature swings because of their thermal
inertia:

- `smoothing_days`: width of a trailing (causal) moving average applied
  to the air series before the regression. A 7-day window is the
  recommended default (5–10 days depending on river depth).
- `smoothing_method`: `simple` (default) or `exponential` (an EWMA
  whose e-folding time equals `smoothing_days`). Only valid together
  with `smoothing_days`.

```yaml
rivers:
  - name: EastArakawa
    source: ${DATA_DIR}/river/discharge/Arakawa/Iwabuchi/discharge_hourly.nc
    temp_source:
      kind: air_regression
      air_nc_template: ${DATA_DIR}/metforce/fvcom_forcing_{year}.nc
      air_var: T2
      air_lat: 35.79
      air_lon: 139.78
      slope: 0.8073
      intercept: 3.9063
      smoothing_days: 7        # trailing moving average (days)
      smoothing_method: simple # or: exponential
```

Absent `smoothing_days`, the regression uses the instantaneous air
temperature (unchanged behaviour). The methodology and literature
basis are recorded in the river_dl findings doc
`docs/water_temperature_observed_2026_05_17.md` (river_dl repo).

### Companion NML generator (`xfvcom-make-rivers-namelist`)

`xfvcom-make-rivers-namelist` consumes the **same** `--river-map`
YAML and emits the matching `RIVERS_NAMELIST*.nml` with one
`&NML_RIVER` block per entry. Pairing the two CLIs lets the
combined NC + NML pair be regenerated end-to-end from a single
YAML, which is the pattern used by `TB-FVCOM/hydro/jobs/build_riv_sewer.sh`:

```bash
xfvcom-make-rivers-namelist \
    --river-map  hydro/input/river_dl_map_goto2023.yaml \
    --river-file tb18_riv_sewer_riverdl_2020.nc \
    --output     input/goto2023/river/RIVERS_NAMELIST_sewer_new_bc.nml

xfvcom-make-river-nc-from-river-dl \
    --nml        input/goto2023/river/RIVERS_NAMELIST_sewer_new_bc.nml \
    --river-map  hydro/input/river_dl_map_goto2023.yaml \
    --start      2020-01-01T00:00:00 \
    --end        2021-01-01T00:00:00 \
    --output     input/goto2023/river/2020/tb18_riv_sewer_riverdl_2020.nc
```

The NML generator reads `node:` (RIVER_GRID_LOCATION) and `vertical:`
(RIVER_VERTICAL_DISTRIBUTION) from each YAML row. For a partially
migrated YAML that lacks `node:`, pass `--nml-fallback <existing.nml>`
to source node ids from a reference NML keyed by `RIVER_NAME`.
`RIVER_FLUX_SCALE_LOCAL` is hard-coded to `1.0`; per-river runtime
calibration belongs in the run script, not in this generated file.

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
