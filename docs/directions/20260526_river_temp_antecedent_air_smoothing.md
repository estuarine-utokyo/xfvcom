# Add antecedent-air-temperature smoothing to the `air_regression` river-T forcing

You are working in `~/Github/xfvcom`. Read `CLAUDE.md` first
(language policy, the `xfvcom` conda env, the GENKAI batch-job rule,
the Sphinx-based docs layout, the package-management policy, and the
Direction-files convention at the bottom of `CLAUDE.md`). Then read,
in order:

1. `xfvcom/cli/make_river_nc_from_river_dl.py` — the adapter CLI;
   specifically `_validate_temp_source()` (the `air_regression`
   branch, currently keys
   `air_nc_template, air_var, air_lat, air_lon, slope, intercept`
   plus optional `min_temp` / `max_temp`).
2. `xfvcom/io/river_nc_generator.py` — the generator;
   specifically `_evaluate_air_regression()` (builds the hourly
   `air_on_timeline` series and computes
   `water = slope * air_on_timeline + intercept`, then clips).
3. `~/Github/river_dl/docs/water_temperature_observed_2026_05_17.md`
   — the upstream findings doc. Its **Antecedent-air-temperature
   smoothing (2026-05-26)** section is the methodology rationale and
   literature basis for this directive; read it for the *why* and
   the recommended window.
4. `docs/directions/20260517_river_adapter_constant_source.md` — the
   prior river-adapter directive, for house style and the schema-v3
   `temp_source` design it built on.

## Background (one paragraph)

The `air_regression` temp_source maps air temperature to river
temperature at the **same hourly timestep**:
`T_water(t) = slope·T_air(t) + intercept`, evaluated on the metforce
`T2` timeline (`_evaluate_air_regression` in
`xfvcom/io/river_nc_generator.py`). Because water has far greater
thermal inertia than air, this transfers the full diurnal + synoptic
air-temperature variance into the river boundary forcing, producing
water-temperature swings that are physically too large. The empirical
air→water literature (Stefan & Preud'homme 1993 weekly; Mohseni et al.
1998 weekly logistic; a "7-day universal integration period" in recent
work; air2stream as the heat-budget end-member) drives the regression
with an **antecedent** (backward moving-average) air temperature
instead. The upstream `river_dl` findings doc records the consensus:
a **7-day backward simple moving average** is the recommended default,
5–10 d depending on river depth. Crucially, because the model is
linear, `MA[slope·Ta + intercept] = slope·MA[Ta] + intercept`, so
smoothing the air driver needs **no refit** of the existing per-river
coefficients and preserves the seasonal cycle.

## What needs to change

A single, fully backward-compatible addition to the `air_regression`
temp_source: an optional `smoothing_days` field (plus an optional
`smoothing_method`). When absent, output is byte-identical to today.

### Change 1 — schema validation

In `xfvcom/cli/make_river_nc_from_river_dl.py::_validate_temp_source`,
inside the `kind == "air_regression"` branch, accept two new optional
keys and normalise them onto the returned dict:

```yaml
temp_source:
  kind: air_regression
  air_nc_template: ${DATA_DIR}/metforce/fvcom_forcing_{year}.nc
  air_var: T2
  air_lat: 35.79
  air_lon: 139.78
  slope: 0.8073
  intercept: 3.9063
  min_temp: 0.0            # existing, optional
  smoothing_days: 7.0      # NEW, optional — backward moving-average window (days)
  smoothing_method: simple # NEW, optional — "simple" (default) | "exponential"
```

Validation rules (fail-loud, single-line message naming the entry):

- `smoothing_days`: optional `float`. If present it must be `> 0`.
  Reject `<= 0` and non-numeric. Absent ⇒ no smoothing.
- `smoothing_method`: optional `str`, one of `{"simple", "exponential"}`,
  default `"simple"`. Reject anything else. It is only meaningful when
  `smoothing_days` is present; if `smoothing_method` is given without
  `smoothing_days`, reject with a clear message (no window to apply it
  to).
- For `smoothing_method: exponential`, interpret `smoothing_days` as the
  **e-folding time constant** (not a hard window); see Change 2.
- These keys are only valid under `kind: air_regression`. They have no
  meaning for `monthly_climatology`; if seen there, reject.

### Change 2 — apply the smoothing in the generator

In `xfvcom/io/river_nc_generator.py::_evaluate_air_regression`, apply
the smoothing to `air_on_timeline` **before** the regression line
`water = slope * air_on_timeline.to_numpy(...) + intercept` and
**before** the `min_temp`/`max_temp` clip (clip must remain the last
operation, so the physical bound is enforced on the final water-T).

Semantics:

- **Window in timesteps**: the timeline cadence is `self.dt` seconds.
  `n_steps = round(smoothing_days * 86400 / self.dt)` (e.g. 7 days at
  `dt=3600` ⇒ 168 steps). A trailing window of whole days inherently
  removes the diurnal cycle and damps synoptic variability in one step.
- **Causal / trailing**: use a backward (right-aligned, `center=False`)
  window — this is the physically-correct "antecedent air temperature"
  and the user's stated intent ("reflect past history"). It also delays
  the seasonal peak by ≈ window/2, which is correct (the river's annual
  max lags the air's).
- `smoothing_method: simple` ⇒
  `air_smoothed = air_on_timeline.rolling(window=n_steps, min_periods=...).mean()`.
- `smoothing_method: exponential` ⇒ pandas
  `air_on_timeline.ewm(halflife=<smoothing_days in timesteps>, times=...).mean()`
  (or `span`/`alpha` — document the exact mapping you choose; e-folding
  time τ days ↔ `alpha = 1 - exp(-dt_days/τ)`).

**Edge / lead-in handling (important — do not introduce a ramp-up
bias).** A trailing window needs `smoothing_days` of air data *before*
`self.timeline[0]`. The current code only opens the metforce NC for
`years = sorted(set(self.timeline.year))`, so the first `smoothing_days`
of the output would otherwise be averaged over a short window. Fix by
**extending the air read** to start `smoothing_days` (plus a small
margin) before `self.timeline[0]` — open the prior year's NC too when
the lead-in crosses a year boundary — compute the rolling/ewm mean on
the **extended** air series, and only then reindex onto `self.timeline`.
If the prior year's NC genuinely does not exist (start of archive),
fall back to `min_periods=1` and emit a one-line `[WARN]` naming the
river and the truncated lead-in; do **not** silently produce NaN (the
existing `isna().any()` guard at the end must still hold).

**Linearity note for the implementer**: smoothing the air input is
mathematically identical to smoothing the `slope·air+intercept` output,
so no coefficient refit is involved. Smoothing the input is preferred
because it keeps the `min_temp`/`max_temp` clip as the final, physically
meaningful operation.

## Implementation choices (you decide; be consistent)

- **Default window**: do **not** bake a default `smoothing_days` into
  the generator. Absent ⇒ no smoothing (backward compat). The *project*
  default of 7 days is expressed in the YAML map (hydro side), not in
  code.
- **Per-group windows**: out of scope for this directive — it is just a
  YAML value per river. The hydro-side YAML may set 5 d for urban rivers
  and 7–10 d for mountainous ones (see the river_dl findings doc); the
  generator only needs to honour whatever `smoothing_days` it is given.
- **`smoothing_method: exponential`**: implement it, but treat `simple`
  as the primary, well-tested path. If `ewm` with a lead-in is awkward,
  it is acceptable to land `simple` first and `exponential` in a
  follow-up commit — say so in the commit body.

## Concrete steps

1. **Schema validation + tests** in
   `tests/cli/test_make_river_nc_from_river_dl*.py`:
   - `smoothing_days: 7` under `air_regression` validates and round-trips.
   - `smoothing_days: 0` / `-1` / `"abc"` → ValueError.
   - `smoothing_method: bogus` → ValueError; `exponential` accepted.
   - `smoothing_method` without `smoothing_days` → ValueError.
   - `smoothing_days` under `monthly_climatology` → ValueError.

2. **Generator behaviour + tests** in the river-generator test module:
   - **Backward-compat**: an `air_regression` entry *without*
     `smoothing_days` produces a column **byte-identical** to the
     current output (guard the "invisible default" promise).
   - **Smoothing correctness**: with a synthetic air series (e.g. a
     diurnal sinusoid on a constant seasonal mean), the 7-day-smoothed
     water column has (a) the same time-mean as the unsmoothed column to
     tolerance (MA preserves the mean away from edges), and (b) strictly
     smaller variance.
   - **Commutativity**: smoothing the air input then applying the
     regression equals applying the regression then smoothing the
     output, to floating-point tolerance.
   - **Lead-in**: a run starting mid-archive (prior year NC available)
     has a *fully* averaged first day (no ramp-up dip); a run starting at
     the archive's first year falls back to `min_periods=1` with the
     documented `[WARN]` and no NaN.

3. **Docs (Sphinx)**: in the river-adapter chapter (locate via
   `grep -rl xfvcom-make-river-nc-from-river-dl docs/`), document the two
   new optional keys with the 7-day default recommendation and a one-line
   pointer to the methodology in the river_dl findings doc. Keep this
   `docs/directions/` file out of Sphinx (already governed by the
   directory README).

4. **CI**: `black`, `isort`, `mypy`, `pytest -m "not png"` must stay
   green. Run them locally (use the pre-commit guard) before committing.

5. **Commit** (English messages). Suggested:
   1. `xfvcom: add smoothing_days to air_regression river-T forcing` —
      schema + generator + tests.
   2. `docs: document antecedent-air-T smoothing in river-adapter chapter`
      — Sphinx (if not folded into 1).

   Each commit body must include
   `Implements docs/directions/20260526_river_temp_antecedent_air_smoothing.md`.
   **Do not push to origin** without explicit user instruction.

## Acceptance criteria (recap)

- `air_regression` accepts optional `smoothing_days` (float > 0) and
  `smoothing_method` ∈ {simple, exponential}; absent ⇒ byte-identical
  output to today.
- The smoothing is a trailing (causal) filter applied to the air series
  before the regression and before the temperature clip, with a correct
  lead-in (no start-of-series ramp-up bias) and no silent NaN fill.
- Unit tests cover the validation matrix, backward-compat byte-identity,
  variance reduction, mean preservation, commutativity, and lead-in
  handling, and run under `pytest -m "not png"`.
- Sphinx river-adapter chapter documents the new keys and the 7-day
  default; cross-links the river_dl findings doc.
- Commits cite this direction file; no push without explicit instruction.

## Constraints (re-read `CLAUDE.md` if unsure)

- **Login-node policy**: unit tests are tiny and may run on the login
  node. Any end-to-end NC regeneration over a full year goes through
  `pjsub`.
- **Conda env**: `xfvcom`. **Package management**: scope uses only
  `numpy` / `pandas` / `xarray` / `netCDF4`, all already present — no new
  dependency.
- **Language**: code, docstrings, comments, commits all in English.
- **NaN / fail-loud**: validation errors abort with a clear single-line
  message naming the offending entry; never silently fall back to a
  default window or silently fill NaN.

## When you finish

Print a one-paragraph summary covering: whether you landed
`exponential` in this directive or deferred it, the test counts
(`pytest -m "not png"` green; number of new cases), confirmation of the
byte-identical backward-compat result for a no-`smoothing_days` run, and
a note that the hydro-side YAML (`river_dl_map_goto2023.yaml`) can now
add `smoothing_days: 7` (or per-group 5/7–10 d) per river to activate it.
