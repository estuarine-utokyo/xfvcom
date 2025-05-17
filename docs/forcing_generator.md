# FVCOM Forcing File Generator ― Quick Guide  
*Create river & atmospheric forcing NetCDF/NML files with `xfvcom`*

> **Target version:** `xfvcom ≥ 0.6`  
> **Last update:** 2025-05-17

---

## 1  Overview — What you can generate

| Generator | Output | Typical use |
|-----------|--------|-------------|
| **`RiverNetCDFGenerator`** | `river_forcing.nc` (NetCDF-4) | River discharge (`river_flux`), temperature (`river_temp`), salinity (`river_salt`) |
| **`RiverNmlGenerator`** | `rivers.nml` (FVCOM namelist) | Geometric / static info for each river source |

The same **time-series CSV / TSV** can be reused for both.

---

## 2  Input files

| Path | Description | Sample |
|------|-------------|--------|
| `tests/data/arakawa_flux.csv` | Discharge time series (UTC) | `time,flux` |
| `tests/data/global_temp.tsv` | Water temperature time series (UTF-8/TSV) | `time\ttemp` |
| `tests/data/rivers_minimal.nml` | Base namelist shipped with the grid | — |
| `river_cfg.yaml` (optional) | YAML overrides (defaults / const values) | see below |

### 2.1  Time-series table rules

* Must contain a **`time` column** or already be indexed by `DatetimeIndex`.
* **UTC only**. If timestamps are tz-aware, they are converted to UTC then tz-info is stripped.
* Missing values: `NaN`, `""`, `"NA"`, `"nan"` … (`pandas` default plus YAML-configurable).
* **Linear interpolation** inside the data span.  
  Forward/backward fill, nearest neighbour and **extrapolation are all forbidden**.

### 2.2  Example YAML

```yaml
defaults:
  flux: 5         # fallback if no ts/const provided
  temp: 14.0
  salt: 0.1

interp:
  method: linear      # only 'linear' is allowed
rivers:
  - name: Arakawa
    ts: tests/data/arakawa_flux.csv:flux
    const: {salt: 0.05}
