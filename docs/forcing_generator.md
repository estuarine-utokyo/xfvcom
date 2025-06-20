# FVCOM Forcing‐file Generator
*Create river & atmospheric forcing NetCDF/NML files with `xfvcom`*


This document explains how to create river forcing NetCDF files and the accompanying `river.nml` namelist with **xfvcom**.
Functions for creating atmospheric forcing files will come soon.

---

## 1. Overview

| Generator | Output | Typical use |
|-----------|--------|-------------|
| **`RiverNetCDFGenerator`** | `river_forcing.nc` (NetCDF-4) | River discharge (`river_flux`), temperature (`river_temp`), salinity (`river_salt`) |
| **`RiverNmlGenerator`** | `rivers.nml` (FVCOM namelist) | Geometric / static info for each river source |

The same **time-series CSV / TSV** can be reused for both.


The generator takes one or more *time‑series* tables plus optional constant values and converts them into a single NetCDF‑4 file that FVCOM can read.  A helper class automatically writes the matching `&RIVER_INFO` namelist.

```
xfvcom/io/river_nc_generator.py   ← NetCDF writer
xfvcom/io/river_nml_generator.py  ← namelist writer
```

---

## 2. Usage patterns


| Path | Description | Sample |
|------|-------------|--------|
| `tests/data/arakawa_flux.csv` | Discharge time series (UTC) | `time,flux` |
| `tests/data/global_temp.tsv` | Water temperature time series (UTF-8/TSV) | `time\ttemp` |
| `tests/data/rivers_minimal.nml` | Base namelist shipped with the grid | — |
| `river_cfg.yaml` (optional) | YAML overrides (defaults / const values) | see below |

**The NML file is mandatory and must exist.** A ``FileNotFoundError`` is raised
if the path does not point to a real file.


### 2.1  One‑liner CLI

```bash
make_river_nc.py   --nml rivers_minimal.nml   --start 2025‑01‑01T00:00Z   --end   2025‑01‑02T00:00Z   --dt    3600   --ts  Arakawa=arakawa_flux.csv:flux   --ts  global_temp.tsv:temp   --const Arakawa.salt=0.05   --const flux=30        # global fallback for other rivers
```

* `--ts RIVER=path:column` &nbsp;adds a *per‑river* time‑series.
* `--ts path:column` (without a river name) defines a global file shared by all rivers.
* `--const RIVER.var=value` sets a river‑specific constant.
* `--const var=value` sets a global fallback.

### 2.2  Example **YAML**

```yaml
# river_cfg.yaml
defaults:
  flux: 5                 # global fallback (m³ s⁻¹)
  temp: 20                #   ″     (°C)
  salt: 0.1               #   ″     (PSU)

rivers:
  - name: Arakawa
    ts:
      - arakawa_flux.csv:flux
    const:
      salt: 0.05          # override only for Arakawa

interp:
  method: linear          # allowed: linear (only)
```

Run with:

```bash
make_river_nc.py   --nml rivers_minimal.nml   --start 2025‑01‑01T00:00Z   --end   2025‑01‑01T12:00Z   --dt    21600   --config river_cfg.yaml
```

---

## 3. Time‑series format

* **Delimiter**: auto‑detected (CSV / TSV).
* **Encoding**: auto‑detected via *chardet*; UTF‑8 recommended.
* Required column: `time` (ISO 8601 or any pandas‑supported date).
* Additional numeric columns hold variables (e.g. `flux`, `temp`).

```text
time,flux
2024‑12‑31T15:00Z,100
2024‑12‑31T21:00Z,105
2025‑01‑01T03:00Z,110
```

### 3.1  Interpolation rules

* Only **linear** interpolation is implemented.
* All requested timestamps must be **inside** the source span.  
  Outside → raises `ValueError`.
* NaNs remaining *inside* the span also raise an error.
* No extrapolation, ffill or bfill is ever applied.

---

## 4. Programmatic API

```python
from pathlib import Path
from xfvcom.io.river_nc_generator import RiverNetCDFGenerator

gen = RiverNetCDFGenerator(
    nml_path       = Path("rivers_minimal.nml"),
    start          = "2025‑01‑01T00:00Z",
    end            = "2025‑01‑02T00:00Z",
    dt_seconds     = 3600,
    ts_specs       = ["Arakawa=arakawa_flux.csv:flux"],
    const_specs    = ["Arakawa.salt=0.05", "flux=30"],
)

nc_bytes = gen.render()          # → bytes; write() or open with xarray
(Path("river.nc")).write_bytes(nc_bytes)
```

---

## 5. Tips & gotchas

| Problem                                   | Remedy |
|-------------------------------------------|--------|
| “outside the available data range” error  | extend source table to cover *start…end* period |
| “Column 'temp' not found”                 | add the column to the CSV/TSV or correct `:column` spec |
| multi‑river run                           | give `--const flux=<val>` as a global fallback or provide per‑river time‑series |

---

## 6. File locations

| Path | Purpose |
|------|---------|
| `tests/data/arakawa_flux.csv` | sample discharge table |
| `tests/data/global_temp.tsv` | sample temperature table |
| `tests/data/river_cfg.yaml`  | YAML used in unit tests |

---

*Last updated: 2025‑05‑18*
