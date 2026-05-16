# Append the trailing 2021-01-01 00:00 endpoint to `tb_wnd_metforce_2020.nc`

You are working in `~/Github/xfvcom`. Read `CLAUDE.md` first
(language policy, the `xfvcom` conda env, the GENKAI batch-job
rule, the Sphinx-based docs layout, the package-management
policy, and the Direction-files convention at the bottom of
`CLAUDE.md`). Then read the previous, partially-completed
direction at
`docs/directions/20260516_tb_wnd_metforce_2020_rebuild.md` —
**this file is its follow-up, not a replacement**: the existing
NaN gap-fill is correct and stays in; only the time-axis end
condition needs fixing.

## Background (one paragraph)

`scripts/job_build_tb_wnd_metforce_2020.pjsub` (commit `e960358`,
output committed at `c8c4a64`) regenerated the FVCOM-mesh
atmospheric forcing from the gap-filled
`~/Github/metforce/analysis/fvcom_forcing_2020.nc`, producing
`~/Github/TB-FVCOM/input/goto2023/forcing-fvcom-grid/2020/tb_wnd_metforce_2020.nc`.
The metforce source covers the half-open hourly range
`2020-01-01T00:00 → 2020-12-31T23:00` (8784 hours). The wrapper
pinned `--end 2020-12-31T23:00:00` to dodge a trailing-hour NaN
that the inclusive default `2021-01-01T00:00:00` would have
produced. Result: the rebuilt file has **8784 hourly steps,
ending at 2020-12-31 23:00** (stored as MJD `59214.957031` —
intended `59214.958333`, drifted by float32 precision).

Downstream, the TB-FVCOM `new_bc` baseline uses
`NML_CASE.END_DATE = '2021-01-01 00:00:00'`. FVCOM iterates up
to and *including* the END_DATE timestep and demands a forcing
sample at exactly that moment. Because the rebuilt file ends one
hour short, FVCOM aborts in 9 seconds at startup with:

```
FATAL_ERROR on rank N : IN SURFACE WIND BOUNDARY CONDITION FILE OBJECT
   FILE NAME: tb_wnd_metforce_2020.nc
   THE MODEL RUN ENDS AFTER THE FORCING TIME SERIES
```

(reproduced in TB-FVCOM job 5832920 on 2026-05-16; see
`~/Github/TB-FVCOM/hydro/tuning/new_bc/logs/new_bc_5832920.log`).

The previous (legacy) `tb18_wnd_metforce_2020.nc` had **8785
steps**, with the trailing step at MJD `59215.0` (= exactly
`2021-01-01 00:00:00`), so it satisfied FVCOM's inclusive-end
requirement. The hydro-side `new_bc` workflow (a) cannot fall
back to the legacy file (it still carries the 21-hour MSM-S
NaN), and (b) per the directives, must not work around an
unfinished upstream by trimming `END_DATE`. Hence this fix lives
in xfvcom.

## What needs to change

The rebuilt file must contain **one extra hourly step at exactly
`2021-01-01T00:00:00 UTC`** (MJD = `59215.0`), bringing the
length from 8784 to 8785, with field values copied verbatim from
the last source step (`2020-12-31T23:00:00`, idx 8783). FVCOM
linearly interpolates between adjacent timesteps; with the last
two timesteps holding identical fields, the wind/T/SLP/SW/LW are
held constant for the closing hour of the simulation — a benign
choice for a 1-hour endpoint extrapolation.

Two collateral fixes that are cheap to do in the same rebuild
and would have caught (or prevented) downstream surprises:

1. **Switch the `time` variable to `float64`.** The legacy file
   was also `float32` MJD but its 8785-step axis happens to land
   on integer MJD at both ends; the rebuild's 8784-step axis
   accumulates float32 error and shows `min dt = 3375 s` /
   `max dt = 3712.5 s` despite a uniform intended 3600 s. Storing
   MJD-since-epoch in `float64` (or `int32` seconds since
   `2020-01-01T00:00:00`) eliminates the drift and makes
   `dt = 3600 s` exactly for every interval.

2. **Self-validate that the emitted last timestep equals the
   inclusive endpoint requested.** If a future caller passes
   `--end 2020-12-31T23:00:00` (no inclusive bookend) the file
   length should be 8784; if a caller passes
   `--end 2021-01-01T00:00:00` (inclusive bookend) the file
   length should be 8785 with the last hour duplicated from the
   source. Today the CLI produces 8784 either way because the
   metforce source has no 8785th hour. Make the bookend behaviour
   opt-in and explicit (see "Implementation choice" below); never
   silently truncate.

## Implementation choice (you decide; just be consistent)

Either of the two patterns below is acceptable; pick one and
justify it in the commit body.

* **Option A — CLI flag** (preferred, leaves the existing CLI
  default unchanged). Add a flag like
  `--pad-trailing-bookend / --no-pad-trailing-bookend` (default:
  off, to preserve the current behaviour for callers that did not
  ask for it). When set and the requested `--end` is exactly one
  source-hour past the source's last timestamp, append one extra
  step with the source's last-hour fields. Update the rebuild
  pjsub wrapper to pass `--pad-trailing-bookend --end
  2021-01-01T00:00:00`.

* **Option B — wrapper-side post-processing**. Leave the CLI
  alone; have the pjsub wrapper open the emitted NC, append one
  trailing-hour record (duplicating idx `-1`), and rewrite the
  `time` axis as `float64` MJD with the appended `59215.0`. This
  keeps the change small and localised to the wrapper but means
  any other caller of the CLI needs to repeat the post-step.

Either way, the float64 time-axis fix should be applied
uniformly (it costs ~70 KB extra in the file vs. saving hours of
debugging downstream).

## Concrete steps

1. **Re-read** the previous direction at
   `docs/directions/20260516_tb_wnd_metforce_2020_rebuild.md` and
   the existing wrapper at
   `scripts/job_build_tb_wnd_metforce_2020.pjsub`. Note the
   metforce source path
   (`${HOME}/Github/metforce/analysis/fvcom_forcing_2020.nc`),
   the grid path
   (`${HOME}/Github/TB-FVCOM/input/goto2023/grid/TokyoBay_grd.dat`),
   and the output path
   (`${HOME}/Github/TB-FVCOM/input/goto2023/forcing-fvcom-grid/2020/tb_wnd_metforce_2020.nc`).
   The output path stays the same; we are *replacing in place*
   the 8784-step file with an 8785-step file under the same
   filename.

2. **Implement** Option A or Option B per the section above. The
   user-visible behaviour (in either pattern):
   - When the wrapper requests an inclusive `2021-01-01T00:00:00`
     end, the emitted file has `nt = 8785` with the last
     timestep at MJD `59215.0` exactly.
   - All seven core OI/derived variables (`uwind_speed`,
     `vwind_speed`, `air_temperature`, `relative_humidity`,
     `air_pressure`, `short_wave`, `long_wave`) and
     `cloud_cover` are NaN-free at the appended step (since they
     are copied from the previous step).
   - `Precipitation` keeps its existing residual NaN budget as
     documented in the prior rebuild's commit `c8c4a64`. The
     duplicated trailing hour is allowed to inherit whatever the
     8783rd step had for `Precipitation` — downstream
     `PRECIPITATION_ON = F` makes it irrelevant.
   - The `time` variable is `float64 days since 1858-11-17
     00:00:00` (or equivalent int32-seconds basis), with
     `dt = 3600 s` exactly for every interval.

3. **Update the pjsub wrapper**
   `scripts/job_build_tb_wnd_metforce_2020.pjsub` to pass
   `--end 2021-01-01T00:00:00` (and the new flag if Option A).
   Drop or rewrite the comment block that previously justified
   pinning to `2020-12-31T23:00:00`; replace with a one-line
   note explaining the trailing-hour bookend.

4. **Validate** in the wrapper's existing inline Python block:
   ```python
   import numpy as np, xarray as xr
   ds = xr.open_dataset(out, decode_cf=False)
   assert ds.dims["time"] == 8785, f"want 8785, got {ds.dims['time']}"
   assert float(ds["time"].values[-1]) == 59215.0, ds["time"].values[-1]
   dt = np.diff(ds["time"].values) * 86400.0
   assert dt.min() == 3600.0 and dt.max() == 3600.0, (dt.min(), dt.max())
   for v in ("uwind_speed","vwind_speed","air_temperature",
             "relative_humidity","air_pressure","short_wave","long_wave"):
       assert int(np.isnan(ds[v].values).sum()) == 0, v
   ```
   If any assertion fails, the wrapper must exit non-zero so the
   bad NC does not persist.

5. **Submit** the wrapper:
   `pjsub scripts/job_build_tb_wnd_metforce_2020.pjsub`. Capture
   the new pjsub job ID. Confirm the resulting file matches the
   acceptance criteria below.

6. **Commit**, English commit messages. Suggested order:
   1. `xfvcom: append trailing bookend step to FVCOM-mesh forcing
      output` — the CLI / library change (Option A) or the
      wrapper-side post-processing (Option B), plus the float64
      time-axis fix. Reference the prior rebuild commit
      (`c8c4a64`) and explain why the bookend is needed
      (FVCOM-inclusive END_DATE semantics).
   2. `scripts: pass inclusive 2021-01-01 endpoint to tb_wnd_metforce
      rebuild` — the wrapper update only.
   3. `Regenerate tb_wnd_metforce_2020.nc with 8785-step time axis`
      — the rebuilt NC. Body must include the new pjsub job ID
      and the validation output (`nt`, last MJD, dt min/max,
      NaN counts).

   Each commit body must include
   `Implements docs/directions/20260516_tb_wnd_metforce_2020_trailing_step.md`
   so the audit trail is intact. Do not push to origin without
   explicit user instruction.

7. **Notify the hydro side** by leaving a one-line note in your
   final summary saying the file is regenerated and the
   downstream re-submission can proceed. The hydro side will
   re-run `pjsub run_new_bc.sh` (no script changes needed there
   beyond the existing preflight gate).

## Acceptance criteria (recap)

- `~/Github/TB-FVCOM/input/goto2023/forcing-fvcom-grid/2020/tb_wnd_metforce_2020.nc`
  has `nt = 8785`, last `time` value `= 59215.0` MJD (=
  `2021-01-01T00:00:00 UTC`).
- `time` dtype is `float64` (or equivalent integer basis), with
  `dt = 3600 s` uniformly.
- Seven core OI/derived variables and `cloud_cover` remain
  NaN-free; `Precipitation` residual NaN count is unchanged
  modulo the appended hour (and is irrelevant downstream).
- Wrapper validation block exits non-zero on any failed
  assertion.
- Commits as listed above, with direction-file reference in
  each body. Do not push without explicit user instruction.

## Constraints (re-read `CLAUDE.md` if unsure)

- **Login-node policy**: build the NC via `pjsub` per the
  existing wrapper. The validation Python is small (a few
  seconds) and may run inside the same job.
- **Conda env**: `xfvcom`.
- **Package management**: any new dependency must come from
  `conda-forge` via `environment.yml`. The fix described here
  uses only `numpy` and `xarray`, both already present.
- **Filename / location** unchanged from the prior rebuild — the
  existing `tb_wnd_metforce_2020.nc` is overwritten in place.
- **Language**: code, docs, commits in English.
- **Sphinx**: `docs/directions/` files are not Sphinx-indexed;
  do not add to `index.rst`.
- **CI**: keep `black`, `isort`, `mypy`, `pytest -m "not png"`
  green (CLI changes touch type signatures; mypy will complain
  if `Optional` / `bool` annotations are missing).

## When you finish

Print a one-paragraph final summary covering:
- Which option (A or B) you chose and why,
- The pjsub job ID for the regenerated NC,
- Validation output (`nt`, last `time` value, `dt min/max`,
  NaN counts for the eight non-PRECIP variables and the
  PRECIP residual),
- Confirmation that the hydro side can re-submit
  `run_new_bc.sh` without any further upstream changes.
