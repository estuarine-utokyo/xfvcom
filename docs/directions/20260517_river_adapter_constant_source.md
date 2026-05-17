# Extend river adapter with a `kind: constant` source + add a YAML→NML generator CLI

You are working in `~/Github/xfvcom`. Read `CLAUDE.md` first
(language policy, the `xfvcom` conda env, the GENKAI batch-job
rule, the Sphinx-based docs layout, the package-management
policy, and the Direction-files convention at the bottom of
`CLAUDE.md`). Then read, in order:

1. `xfvcom/cli/make_river_nc_from_river_dl.py` (the existing
   adapter CLI) and `xfvcom/io/river_nc_generator.py` (the
   underlying generator class).
2. `~/Github/TB-FVCOM/hydro/input/river_dl_map_goto2023.yaml`
   (the production YAML map for the goto2023 mesh — 29 entries
   today, will gain a 30th `kind: constant` Kisarazu entry once
   this directive lands).
3. `~/Github/TB-FVCOM/hydro/docs/bc_construction_protocol.md`
   §5.2 (the protocol spec the adapter targets).
4. `~/Github/TB-FVCOM/hydro/docs/new_bc_baseline.md` §2.1, §6
   row 5839748, and §8 — the new_bc baseline, the Kisarazu
   in-place workaround already on disk, and the campaign log
   that motivates this directive.
5. `~/Github/TB-FVCOM/hydro/docs/directions/20260517_new_bc_input_pipeline_normalization.md`
   — the paired hydro-side directive that **waits on this one**.

## Background (one paragraph)

The TB-FVCOM `new_bc` baseline (post-F04, see
`~/Github/TB-FVCOM/hydro/docs/new_bc_baseline.md`) generates its
22-river + 8-sewer combined NetCDF from `$DATA_DIR/river_dl/*` via
`xfvcom-make-river-nc-from-river-dl`. The 8th sewer (Kisarazu,
木更津下水処理放流水, node 1440) has no `wasterwater_dl` upstream
source yet, so the 2026-05-17 workaround
(`hydro/scripts/append_kisarazu_to_combined_nc.py`, commit
`ac8547c`) post-processes the combined NC in place to add a 30th
entry with constants `Q = 0.2902` m³/s, `T = 10.0` °C, `S = 0`
PSU broadcast across the 8785 hourly steps. The hydro-side NML
(`RIVERS_NAMELIST_sewer_new_bc.nml`) was hand-edited to match.
This split — adapter produces 29, post-process adds 1, NML is
hand-written — is fragile: any re-run of the adapter (e.g. when a
river_dl source updates) drops Kisarazu, and the NML must be kept
manually in sync with the NC. The user wants the entire 30-entry
output regenerable from a single `--river-map` YAML in one shot.

## What needs to change

Two changes in xfvcom, used together by the paired hydro-side
directive:

### Change 1 — extend the adapter to consume `kind: constant`

`xfvcom-make-river-nc-from-river-dl` today demands every YAML
entry have a `source: <NetCDF path>`. Add a second source
flavour, gated by a new `kind` field:

```yaml
defaults:
  flux: 0.0
  temp: 15.0
  salt: 0.0
rivers:
  # Existing flavour (default; kind: river_dl when omitted).
  - name: EastArakawa
    source: ${DATA_DIR}/river_dl/discharge/Arakawa/Iwabuchi/discharge_hourly.nc
    scale: 0.25
  # New flavour — constant broadcast across the time axis.
  - name: Kisarazu
    kind: constant
    flux: 0.2902              # m^3/s, constant
    temp: 10.0                # degC
    salt: 0.0                 # PSU (optional; default from defaults: block)
    node: 1440                # RIVER_GRID_LOCATION (used by Change 2)
    vertical: uniform         # RIVER_VERTICAL_DISTRIBUTION (used by Change 2)
    provenance: "add_kisarazu_to_sewer.py FY2021 mean"  # free text, copied to NC global attr
```

Schema rules:

- `kind` defaults to `river_dl` if omitted (full backward compat
  with the existing 29-entry production YAML).
- `kind: river_dl` requires `source:`; `kind: constant` forbids
  `source:` and requires `flux:` (`temp` and `salt` may fall
  back to the `defaults:` block).
- `node:` is **optional** for `kind: river_dl` (today's NML is
  the source of truth for ordering / node id); **required** for
  `kind: constant` because the new NML generator (Change 2)
  cannot deduce it elsewhere.
- `vertical:` is optional (default `uniform`).
- `provenance:` is optional free-text; if present, it is
  concatenated into the output NC's global attribute
  `kisarazu_provenance` / `constant_source_provenance` (one
  attribute per `kind: constant` entry, keyed by name) for
  audit.

Runtime behaviour for `kind: constant`:

- The adapter must not attempt to open any NetCDF for the entry.
- For the constant entry's column in `river_flux`, write
  `flux` broadcast across the requested `(start, end, dt)` time
  axis (same length as the rest of the matrix).
- Same for `river_temp` (column = `temp`) and `river_salt`
  (column = `salt`).
- The river's position in the output NC's `rivers` dimension is
  determined by either the `--nml` ordering (existing behaviour)
  or — if `--rivers-from yaml` is passed (see optional fold-in
  below) — by YAML order.

Optional fold-in (judgement call — adopt only if it does not
break the existing CI):

- Accept `--rivers-from {nml,yaml}` (default `nml` for backward
  compat). When `--rivers-from yaml`, the adapter takes the
  river list and ordering from the YAML directly and `--nml`
  becomes optional. This is the simpler downstream UX once
  Change 2 lands (single source of truth), but the existing
  callers can keep passing `--nml` indefinitely.

### Change 2 — add `xfvcom-make-rivers-namelist` CLI

Add a sibling CLI that reads the same `--river-map` YAML and
emits a `RIVERS_NAMELIST*.nml`:

```bash
xfvcom-make-rivers-namelist \
    --river-map ~/Github/TB-FVCOM/hydro/input/river_dl_map_goto2023.yaml \
    --river-file tb18_riv_sewer_riverdl_2020.nc \
    --output     ~/Github/TB-FVCOM/input/goto2023/river/RIVERS_NAMELIST_sewer_new_bc.nml
```

Per YAML entry, emit one `&NML_RIVER` block in YAML order:

```fortran
&NML_RIVER
  RIVER_NAME             = '<name>',
  RIVER_FILE             = '<--river-file value>',
  RIVER_GRID_LOCATION    = <node from YAML>,
  RIVER_VERTICAL_DISTRIBUTION = '<vertical, default uniform>',
  RIVER_FLUX_SCALE_LOCAL = 1.0,
/
```

Rules:

- `RIVER_FILE` is the **same value** for every block (the
  combined NC filename, passed via `--river-file`). It is a
  filename only; FVCOM prepends `INPUT_DIR` at run time.
- `RIVER_FLUX_SCALE_LOCAL` is **always `1.0`** in the generated
  NML (new_bc baseline rule, see `new_bc_baseline.md` §2.1).
  Per-river runtime calibration is the caller's job in the run
  script (today: none); this file is mechanical.
- `RIVER_NAME` must be quoted with single quotes, padded with
  no trailing spaces. Match the formatting of the existing
  hand-written `RIVERS_NAMELIST_sewer_new_bc.nml` (in
  `~/Github/TB-FVCOM/input/goto2023/river/`) so the rendered
  output is diffable against the hand-written version for the
  current 30-entry case.
- `node` for `kind: river_dl` entries: if YAML lacks `node:`,
  read it from `--nml-fallback <path>` (optional CLI arg; used
  during the goto2023 backfill where the existing NML carries
  authoritative node ids). Once the YAML is fully populated
  with `node:` for every entry, `--nml-fallback` becomes
  unnecessary.

The CLI must be NaN-free / fail-loud:

- Refuse to emit if any YAML entry lacks a resolvable `node`
  (neither in YAML nor in the fallback NML). Print which
  entries are unresolved and exit non-zero.
- Refuse to emit if two entries share the same `node` (two
  rivers cannot be pinned to the same mesh node). Print both
  names.

Register the new CLI under `pyproject.toml` `[project.scripts]`
alongside the existing `xfvcom-make-river-nc-from-river-dl`:

```toml
xfvcom-make-rivers-namelist = "xfvcom.cli.make_rivers_namelist:main"
```

## Implementation choice (you decide; just be consistent)

* **Schema location for `node` / `vertical`**: either keep them
  flat on each river entry (`node: 1440`) or nest under a
  `geometry:` sub-mapping. The flat form matches the existing
  YAML style; prefer it unless you have a strong reason
  otherwise.

* **Where to compute time axis for `kind: constant`**: the
  cleanest place is inside `RiverNetCDFGenerator` (so both
  `kind` paths produce a final matrix the same way and write a
  single combined NC). If that requires too much refactor of
  the generator, you may handle constants in a thin wrapper in
  the adapter CLI and call into the generator only for the
  river_dl entries; document the choice in the commit body.

* **`scale:` semantics for `kind: constant`**: forbid `scale:`
  on constant entries (it is meaningless because the constant
  is already in physical units). Validate and reject with a
  clear error if seen.

## Concrete steps

1. **Add schema validation + `kind: constant` path to the
   adapter**. Existing tests must continue to pass. Add unit
   tests covering:
   - Default `kind` resolves to `river_dl` (regression).
   - `kind: constant` entry with missing `flux` → ValueError.
   - `kind: constant` with `source:` → ValueError.
   - `kind: constant` end-to-end: a 24-hour mini run produces an
     output NC whose constant-entry column equals the YAML
     `flux` / `temp` / `salt` to bitwise tolerance, and whose
     `river_dl` entries are byte-identical to a control run
     without the constant entry (i.e. adding a `kind: constant`
     entry must not perturb any other entry's values).

2. **Implement `xfvcom-make-rivers-namelist`** as
   `xfvcom/cli/make_rivers_namelist.py`. Add a thin doctest /
   pytest that round-trips the existing
   `~/Github/TB-FVCOM/input/goto2023/river/RIVERS_NAMELIST_sewer_new_bc.nml`
   (30 entries, the hand-written file) from the soon-to-be
   updated YAML (paired hydro directive) — for the body of this
   directive, you can stop at "regenerates the 29-entry
   `RIVERS_NAMELIST_new_bc.nml` byte-identical to the current
   hand-written file modulo whitespace" since the 30th entry
   does not exist in YAML yet at this point.

3. **Regression check**: run the adapter on the **unchanged**
   29-entry YAML (no `kind` field anywhere) for
   `start=2020-01-01, end=2021-01-01, dt=3600` and confirm the
   output NetCDF is byte-identical to the on-disk
   `~/Github/TB-FVCOM/input/goto2023/river/2020/tb18_riv_sewer_riverdl_2020.nc.bak_pre_kisarazu`.
   This guards the "default-`kind` is invisible" backward-compat
   promise.

4. **Update Sphinx docs**: in the existing river-adapter chapter
   (locate via `grep -l xfvcom-make-river-nc-from-river-dl
   docs/`), add a "Constant sources" subsection summarising the
   schema and a small example. Also add a new `cli` page or
   section for `xfvcom-make-rivers-namelist`. Keep the
   `docs/directions/` file **out** of Sphinx (already governed
   by the README in that directory).

5. **CI**: `black`, `isort`, `mypy`, `pytest -m "not png"` must
   stay green. Run them locally before committing.

6. **Commit**, English commit messages. Suggested order:
   1. `xfvcom: support kind: constant in river_dl YAML map` —
      schema parser + generator changes + unit tests.
   2. `xfvcom: add make-rivers-namelist CLI` — new CLI + unit
      tests + pyproject entry + docs section.
   3. `docs: regenerate river-adapter chapter` — Sphinx updates
      only (if not folded into the prior two).

   Each commit body must include
   `Implements docs/directions/20260517_river_adapter_constant_source.md`
   so the audit trail is intact. **Do not push to origin**
   without explicit user instruction.

7. **Notify the hydro side**: leave a one-line note in the
   final summary saying the xfvcom side is landed and the
   paired hydro directive
   (`~/Github/TB-FVCOM/hydro/docs/directions/20260517_new_bc_input_pipeline_normalization.md`)
   can proceed.

## Acceptance criteria (recap)

- `xfvcom-make-river-nc-from-river-dl` accepts `kind: constant`
  entries in the YAML map; backward-compatible default for
  `kind: river_dl` produces byte-identical output for the
  existing 29-entry goto2023 YAML.
- `xfvcom-make-rivers-namelist` is published as a console script
  and regenerates the 29-entry `RIVERS_NAMELIST_new_bc.nml`
  byte-identical to the hand-written file modulo whitespace
  (full 30-entry regeneration is exercised by the paired hydro
  directive, not this one).
- Unit tests cover the schema validation matrix described above
  and run as part of the standard `pytest -m "not png"` suite.
- Sphinx docs updated for both the adapter's new `kind` field
  and the new NML generator CLI.
- Commits in the order above, each body citing this direction
  file. No push without explicit user instruction.

## Constraints (re-read `CLAUDE.md` if unsure)

- **Login-node policy**: unit tests are tiny and may run on
  the login node via `pytest`. Any end-to-end NC regeneration
  goes through `pjsub` if it takes more than a few seconds.
- **Conda env**: `xfvcom`.
- **Package management**: any new dependency from
  `conda-forge` via `environment.yml`. The scope here uses
  only `numpy`, `xarray`, `pyyaml` — all already present.
- **Filename / location**: new CLI module lives at
  `xfvcom/cli/make_rivers_namelist.py`; tests at
  `tests/cli/test_make_rivers_namelist.py` and
  `tests/cli/test_make_river_nc_from_river_dl_constant.py`.
- **Language**: code, docstrings, comments, commits all in
  English.
- **Sphinx**: `docs/directions/` files stay out of Sphinx
  (already excluded). Sphinx CLI pages live under `docs/`.
- **NaN / fail-loud**: validation errors must abort with a
  clear, single-line message naming the offending entry; never
  silently fall back to a default.
- **Hydro-side YAML stays untouched** in this directive — the
  paired hydro directive owns the schema migration of
  `river_dl_map_goto2023.yaml` (adding `node:` to every entry
  and the new Kisarazu `kind: constant` entry).

## When you finish

Print a one-paragraph final summary covering:
- Which optional fold-ins you adopted (`--rivers-from yaml`?
  Schema-location choice?) and why,
- Test counts (`pytest -m "not png"` green; number of new
  cases),
- The byte-identical regression result against
  `tb18_riv_sewer_riverdl_2020.nc.bak_pre_kisarazu`,
- Confirmation that the paired hydro directive can proceed.
