# Rebuild `tb_wnd_metforce_2020.nc` from the gap-filled metforce 2020 OI

You are working in `~/Github/xfvcom`. Read `CLAUDE.md` first (language
policy, the `xfvcom` conda env, the GENKAI batch-job rule, the
Sphinx-based docs layout, and the Direction-files convention at the
bottom). Then read the related upstream incident document at
`~/Github/metforce/docs/msm_s_archive_validation.md` end-to-end. This
file lives in the `metforce` repo but is the canonical reference for
*why* the rebuild is needed and *what* changed upstream — your task
is the downstream FVCOM-mesh side of that fix.

## Background (one paragraph)

The 2026-05-16 TB-FVCOM `tb-new-bc` baseline rebuild (Job 5831777)
crashed at simulation time `2020-02-21T13:58:20` because its
FVCOM-mesh forcing file `tb18_wnd_metforce_2020.nc` carried NaN at
21 hourly steps that originated from `int16` missing-value
sentinels in the MSM-S daily archive. The metforce sibling repo
fixed this with a Bayesian uninformative-prior treatment in
`metforce.bayes.oi.oi_analyse` (commits `c204793 / 8158e53 / 4d2eae9`)
and now ships a sentinel-clean
`~/Github/metforce/analysis/fvcom_forcing_2020.nc` for the six core
OI variables (`U10`, `V10`, `T2`, `SH`, `SLP`, `DSWRF`) plus the
derived `DLWRF`. Your job is to regenerate the FVCOM-mesh forcing
NetCDF that TB-FVCOM consumes, **with the new filename**
`tb_wnd_metforce_2020.nc` (drop the legacy `18` prefix — see
"Filename convention" below).

## Filename convention

The `18` in `tb18_*.nc` was an F04-era **wind-scale ×1.8** suffix
(Optuna-tuned). The 2026-05-16 new_bc baseline runs at scale = 1.0
(default), so the rebuilt FVCOM-mesh forcing file **must** be named

```
tb_wnd_metforce_<year>.nc
```

(no `18`). The legacy `tb18_wnd_metforce_2020.nc` is the inherited-
prefix file that crashed the 5831777 baseline; do not overwrite it.
Emit the new file in the same directory:

```
/home/pj24001722/ku40000343/Github/TB-FVCOM/input/goto2023/forcing-fvcom-grid/2020/tb_wnd_metforce_2020.nc
```

If that directory is not the right home for new-naming output (e.g.
the team prefers a `forcing-fvcom-grid/2020/new_bc/` subdirectory),
flag it in your final summary — but err on the side of co-location
unless you find a strong reason otherwise.

## Concrete steps

1. **Survey the existing toolchain.** Start at
   `xfvcom/cli/make_met_nc_from_metforce.py` (this is the recently
   added CLI for the `MetforceGriddedSource` → FVCOM-mesh path; see
   commit `5bb1315`). Run `xfvcom-make-met-nc-from-metforce --help`
   under the `xfvcom` conda env to confirm the flag surface. Trace
   how the source path, output path, mesh path, and year are wired
   through. If `pyproject.toml` registers the CLI under a different
   name, use the actual entry point shown there.

2. **Identify the legacy build invocation.** Find any documented or
   committed invocation that produced `tb18_wnd_metforce_2020.nc`
   (search `~/Github/xfvcom` and `~/Github/TB-FVCOM` for shell or
   `pjsub` wrappers that call `xfvcom-make-met-nc-from-metforce`,
   `xfvcom-make-met-nc`, or `make_met_nc_from_metforce.py`). The
   rebuild should match its mesh choice, time-axis spec, and
   variable subset — diverging silently from the legacy output
   shape would defeat the point.

3. **Build the rebuild wrapper** as a `pjsub` script under
   `~/Github/xfvcom/scripts/job_*.pjsub` (xfvcom convention is to
   colocate batch wrappers with the python they call, mirroring
   metforce; see `~/Github/metforce/scripts/job_*.pjsub` for shape).
   Required behaviour:
   - Single `vnode=1`, modest `vnode-core` (the mesh interpolation
     is light vs. the OI itself).
   - `#PJM -X` so `$DATA_DIR` and friends inherit.
   - Pin source path to
     `${HOME}/Github/metforce/analysis/fvcom_forcing_2020.nc` (the
     post-Path-B clean file).
   - Pin output path to
     `${HOME}/Github/TB-FVCOM/input/goto2023/forcing-fvcom-grid/2020/tb_wnd_metforce_2020.nc`.
   - Pin mesh / element / node config to whatever the legacy
     invocation used.
   - Do *not* call `#PJM -N` (default jobname keeps the output
     log inside the `.gitignore` pattern).

4. **Validate the output before committing**. Confirm:
   ```python
   import xarray as xr, numpy as np
   ds = xr.open_dataset(
       "input/goto2023/forcing-fvcom-grid/2020/tb_wnd_metforce_2020.nc"
   )
   # The six core OI variables PLUS DLWRF / DSWRF must be NaN-free
   # across the full year on the FVCOM mesh:
   core_mesh_vars = (
       "uwind_speed", "vwind_speed",        # per-element (nele)
       "air_temperature", "relative_humidity",
       "air_pressure", "short_wave", "long_wave",   # per-node (node)
   )
   for v in core_mesh_vars:
       n = int(np.isnan(ds[v].values).sum())
       assert n == 0, f"{v} still carries {n} NaN cells"
   ```
   (The exact variable names depend on the legacy NC schema —
   confirm them by `xr.open_dataset("…/tb18_wnd_metforce_2020.nc")`
   first, then assert the same set is NaN-free in the rebuilt
   file.)

   The `Precipitation` variable is **expected to retain ~28 NaN
   hours** because metforce Path B does not gap-fill PRECIP (it
   comes from the background field directly, not from OI; see the
   `msm_s_archive_validation.md` "PRECIP" residual note). For the
   new_bc baseline this is acceptable because `PRECIPITATION_ON=F`
   in the run NML; document the residual in your final summary
   and in the commit message. Do not block the rebuild on PRECIP.

5. **Make sure the time axis is exactly hourly.** The legacy file
   showed irregular `dt` (`3375 s / 3712 s / 3713 s` averaging to
   3600 s — a float-rounding artefact from a Julian-day round-trip
   somewhere in the build). If the new build path produces the
   same drift, raise the issue in your final summary; ideally the
   rebuilt file has `dt = 3600 s` for every step.

6. **Commit**, English commit messages, in this order:
   1. `scripts: add tb_wnd_metforce 2020 rebuild wrapper` — the
      pjsub wrapper plus any helper that uniquely supports the
      rebuild.
   2. `Regenerate tb_wnd_metforce_2020.nc from gap-filled
      fvcom_forcing_2020.nc` — the rebuilt NC. Reference
      `~/Github/metforce/docs/msm_s_archive_validation.md` and
      include the pjsub job ID(s).

   Each commit body should include
   `Implements docs/directions/20260516_tb_wnd_metforce_2020_rebuild.md`
   so the audit trail from intent to delivery is intact.

7. **Update `docs/forcing_generator.md`** (already exists in the
   xfvcom docs) with one paragraph documenting the new filename
   convention (`tb_wnd_metforce_<year>.nc`, drop `18` for scale =
   1.0 baselines) and the metforce-PathB-aware build. Commit
   separately as `docs: record tb_wnd_metforce naming for new_bc
   baseline`.

## Acceptance criteria (recap)

- Output file `tb_wnd_metforce_2020.nc` exists at the location
  above, produced by a committed pjsub wrapper.
- Validation script / assert passes: `np.isnan(ds[v]).sum() == 0`
  for every variable except `Precipitation`.
- `Precipitation` residual NaN count is **documented** (commit
  message + final summary) rather than swept under the rug.
- Time axis ideally `dt = 3600 s` uniformly; if not, drift is
  flagged.
- Commits as listed above, with direction-file reference in each
  body. Do not push to origin without explicit user instruction.

## Constraints (re-read `CLAUDE.md` if unsure)

- **Login-node policy**: full-year mesh interpolation runs via
  `pjsub`. Login-node only for the `--help` smoke test and the
  final validation (a `python -c '...isnan...'` snippet is OK).
- **Conda env**: `xfvcom` (see CLAUDE.md for batch-job init).
- **Language**: code, docs, commits in English.
- **Filename**: `tb_wnd_metforce_<year>.nc` (no `18`).
- **Sphinx**: `docs/directions/` files are not Sphinx-indexed; do
  not register them in `index.rst`.

## When you finish

Print a one-paragraph final summary covering:
- Which CLI invocation / pjsub wrapper produced the rebuild,
- The pjsub job ID(s),
- The pre- and post-rebuild NaN counts for the core mesh
  variables,
- The PRECIP residual count (if any) and confirmation that PRECIP
  forcing is off in the downstream new_bc NML,
- Time-axis check result (`dt` uniformity),
- Anything else the next session (TB-FVCOM new_bc rerun, see
  `~/Github/TB-FVCOM/hydro/docs/directions/20260516_new_bc_baseline_rerun_with_preflight.md`)
  needs to know.
