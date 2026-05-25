# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

xfvcom is a Python package for preprocessing and postprocessing FVCOM (Finite Volume Community Ocean Model) data. Built on xarray, it handles ocean model I/O, analysis, ensemble processing, and visualization.

## Environment Setup

```bash
# First-time setup (creates "xfvcom" environment with Python 3.13)
bash setup.sh              # or: bash setup.sh --impi  (for Intel MPI)

# Activate existing environment
conda activate xfvcom

# Verify
python -c "import xfvcom; print(xfvcom.__version__)"
```

**Note**: Use `mamba` (via `setup.sh` / `environment.yml`) for installation to avoid conflicts with Intel oneAPI Python on supercomputers. `pip install -e .[dev]` is supported but not recommended.

## Package management policy

All Python dependencies in this project are sourced from **conda-forge**
via `mamba` (or `conda`). `pip install <package>` is **not** used to add
runtime dependencies — pip and conda environments interact poorly and
silently break compiled stacks (`numpy` / `netCDF4` / `h5py` / `hdf5` /
`eccodes`, etc.).

Workflow when adding a dependency:

1. Add the package to `environment.yml` (it must resolve on `conda-forge`;
   confirm with `mamba search -c conda-forge <pkg>` if uncertain).
2. Apply with `mamba env update -n xfvcom -f environment.yml`.
3. Commit `environment.yml` so the env is reproducible.

Two narrow exceptions are allowed and already encoded in
`environment.yml`:

- **The local editable install** of this project itself: `pip install -e .`
  inside the env (the `pip:` block at the bottom of `environment.yml`).
- **Packages genuinely unavailable on conda-forge.** Verify absence first,
  then fall back to `pip install <pkg>` and document the reason in
  `environment.yml`.

This rule mirrors the global policy in `~/.claude/CLAUDE.md`; do not
deviate from it without an explicit exception.

## Essential Commands

### CI-equivalent checks (run before committing)
```bash
black --check .
isort --check-only .
mypy .
pytest -m "not png"
```

**Quick one-liner to run all checks:**
```bash
python -m black --check . && python -m isort --check-only . && python -m mypy .
```

**Quick one-liner to auto-fix formatting (then manually fix mypy):**
```bash
python -m black . && python -m isort . && python -m mypy .
```

### Pre-commit guard (MANDATORY — prevents recurring CI format failures)

CI (`.github/workflows/ci.yml`) runs `black --check .`, `isort --check-only .`
and `mypy .`. Committing un-formatted code therefore fails CI every time.
This used to recur, so the repo ships a **commit-time guard** that runs those
exact checks before each commit. **Enable it once per clone:**

```bash
conda activate xfvcom    # the tools must be on PATH
pre-commit install       # writes .git/hooks/pre-commit
```

After that, every `git commit` auto-runs black + isort (auto-fixing staged
files) and `mypy .`. If black/isort reformat anything the commit aborts —
re-`git add` the changed files and commit again.

- **Never bypass with `git commit --no-verify`** to dodge a formatting/type
  failure; that is exactly what pushes the error to GitHub.
- **Non-interactive / agent commits** (env not activated): prepend the env
  to `PATH` so the hook can find the tools, e.g.
  `PATH="$HOME/mambaforge/envs/xfvcom/bin:$PATH" git commit -m "…"`.
- **No version pinning here on purpose.** `.pre-commit-config.yaml` uses
  `language: system`, so it runs the *same* black/isort/mypy that CI and the
  conda env install (all unpinned → latest). A prior pinned config drifted
  ~2 years behind CI and silently let mis-formatted commits through; do not
  reintroduce hard-pinned `rev:` versions.

### Common mypy fixes
- Add type annotations to variables: `count: NDArray = np.zeros(...)`
- Use `| None` for optional types: `self.input_dir: Path | None = None`
- Use distinct variable names to avoid type redefinition within same scope
- Add assertions for properties that build internal state: `assert self._data is not None`
- Use `list()` copy to satisfy list invariance: `layers: list[int | None] = list(sigmas)`
- Cast pandas values explicitly: `node_id = int(row["column"])`
- Handle `None` from `importlib.util.spec_from_file_location()`: check `if spec is None`
- Use `ffill()`/`bfill()` instead of deprecated `fillna(method=...)` (pandas-stubs ≥ 3.0)

### Development
```bash
# Format code
black . && isort .

# Run all tests (including PNG regression)
pytest

# Regenerate baseline images after intentional visual changes
pytest --regenerate-baseline -q

# Run specific test
pytest tests/test_grid.py::TestGridCalculations -v
```

**Note**: Ruff is NOT run in CI, so ruff errors are informational only.

## Architecture

### Data Flow
```
Input Files → FvcomDataLoader → xarray Dataset → FvcomAnalyzer → Results
                                      ↓
                                FvcomPlotter + FvcomPlotConfig → Plots
                                      ↓
                                CoastMask (xcoast pkg) → Land masking overlay

Grid File → FvcomInputLoader → Area Calculations / Mesh Connectivity

GWO CSV → GWOReader → GapFiller → MetNetCDFGenerator → met.nc
```

### Core Modules

| Module | Key Classes | Purpose |
|--------|-------------|---------|
| `xfvcom/io/` | `FvcomDataLoader`, `FvcomInputLoader` | Load FVCOM NetCDF and grid files |
| `xfvcom/io/*_generator.py` | `MetNetCDFGenerator`, `RiverNetCDFGenerator` | Generate FVCOM forcing files |
| `xfvcom/io/gwo_*.py` | `GWOReader`, `GapFiller`, `StationCorrelations` | JMA meteorological data processing |
| `xfvcom/io/mpos.py` | `MposLoader`, `MposMeteoLoader` | Tokyo Bay MPOS station data (water-quality + wind / air-temperature) |
| `xfvcom/io/sources/mpos_wind.py` | `MposWindSource` | Spatial wind / air-temperature source for FVCOM (IDW from 3 MPOS stations; 02kawasaki excluded by default due to wind-tower shielding) |
| `xfvcom/io/spatial_interp.py` | `idw_weights`, `idw_apply`, `haversine_km` | Geographic IDW for sparse-station -> mesh projection |
| `xfvcom/analysis.py` | `FvcomAnalyzer` | KDTree search, layer averages, tidal decomposition |
| `xfvcom/plot/` | `FvcomPlotter`, `FvcomPlotConfig` | Visualization engine and styling |
| `xfvcom/coastmask/` | (deprecation shim) | Re-exports `CoastMask`, `CoastmaskConfig` from the standalone `xcoast` package — see https://github.com/estuarine-utokyo/xcoast |
| `xfvcom/dye_timeseries.py` | `DyeCase`, `Selection`, `aggregate()` | Multi-member ensemble aggregation |
| `xfvcom/grid/` | `FvcomGrid` | Area calculations (median-dual, triangle sum) |
| `xfvcom/cli/` | CLI entry points | Command-line tools (`xfvcom-*`) |

### Force File Generators

**Critical**: Use `netCDF4` directly (not xarray) for FVCOM compatibility. All generators inherit from `BaseForceGenerator`.

## Key Conventions

### Index Conventions
- **FVCOM**: 1-based indexing
- **Python/xarray**: 0-based indexing
- Most functions accept `index_base` parameter (default varies by context)

### Timezone Handling
- **Default**: Asia/Tokyo (JST) input → UTC output
- **GWO-AMD mode exception**: JST values stored directly but labeled as UTC (Tokyo Bay convention)

### Coordinate Systems
- **UTM**: For area calculations (meters) — specify `utm_zone` (e.g., 54 for Tokyo Bay)
- **Geographic**: For visualization (degrees)

### Area Calculations
- `calculate_node_area_median_dual()`: FVCOM-standard control volumes (preferred)
- `calculate_node_area()`: Legacy triangle sum method

### NaN Handling
- **Ensemble plots**: Hard-fail on NaN (design choice — user must clean data first)
- **Other plots**: Graceful handling

## Testing

### Key Fixtures (from `tests/conftest.py`)
- `fvcom_ds`: Minimal 3-D FVCOM-like xarray Dataset
- `plotter`: FvcomPlotter instance bound to fvcom_ds
- `regen_baseline`: True when `--regenerate-baseline` flag set

### PNG Regression Tests
- Marked with `@pytest.mark.png`
- Skipped in CI (baseline images in `tests/baseline/`)
- Regenerate after matplotlib/cartopy updates or intentional visual changes

## CLI Tools

Registered in `pyproject.toml` under `[project.scripts]`:
- `xfvcom-make-met-nc`: Meteorological forcing (supports GWO-AMD with gap filling)
- `xfvcom-make-river-nc`, `xfvcom-make-river-nml`: River forcing
- `xfvcom-make-groundwater-nc`: Groundwater forcing
- `xfvcom-check-met`: Validate forcing files
- `xfvcom-calc-gwo-corr`: Compute station correlations
- `xfvcom-dye-ts`: Dye time series extraction

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors after updating | `pip install -e .` |
| PNG tests failing after matplotlib update | `pytest --regenerate-baseline -q` |
| Force file rejected by FVCOM | Ensure netCDF4 (not xarray) was used; check time format |
| Cartopy text clipping | Use `make_node_marker_post(..., text_clip_buffer=-0.001)` |
| Slow nearest neighbor search | Reuse `FvcomAnalyzer` instance (KDTree built once) |

## Language

- **Chat (conversation)**: Always respond in Japanese (日本語で応答すること)
- **Documentation files** (README.md, etc.): English
- **Code comments**: English
- **Commit messages**: English

## Direction files (`docs/directions/`)

Task-specific delegation instructions for automated Claude Code
sessions (interactive or `claude -p` headless) are committed under
`docs/directions/` with the filename convention

```
docs/directions/YYYYMMDD_<topic-slug>.md
```

Each file captures the human's intent at the moment of delegation
and pairs with the resulting implementation commit(s).

Rules:

* **Immutable** once committed. They capture intent at the
  delegation moment; the *result* lives in the implementation
  commits and any updates to durable docs (e.g. `docs/*.md`,
  the Sphinx-built user manual).
* **Self-contained**. A fresh-session or headless Claude reading
  the file alongside this CLAUDE.md should have enough context
  to start work without conversation history. Reference durable
  docs by path.
* **Date prefix** sorts chronologically.
  `<topic-slug>` is lowercase-with-underscores and describes the
  deliverable (e.g. `tb_wnd_metforce_2020_rebuild`, not `task_c`).
* **Pair with implementation commits**: in the body of each commit
  that implements the directive, say
  `Implements docs/directions/YYYYMMDD_<topic>.md`. That gives
  `git log` a clean trail from intent to delivery.

These files live under `docs/` for proximity to the package's
other documentation, but they are **not** indexed by Sphinx —
they are delegation artefacts, not part of the published xfvcom
user manual.

Sibling repos with the same convention:
`~/Github/metforce/docs/directions/`,
`~/Github/TB-FVCOM/{hydro,ersem}/docs/directions/`,
`~/Github/jcopetda/docs/directions/`.

## Campaign logs (`docs/campaigns/` if needed)

A "campaign" is a multi-repo investigation or fix arc whose state
outlives any single direction file. The orchestrator session
driving the campaign produces a **campaign log** that captures
*why* each direction was issued, what cross-repo decisions were
taken (and which alternatives were rejected, with reasons), and
the open state of the dependency graph at session end — so a
future orchestrator (or a future Claude resuming the campaign)
can reconstruct the meta-state, not just the per-direction
artefacts. Without this layer, the rationale that ties multiple
directions together evaporates when the orchestrator session
ends.

Convention:

* When a campaign has a clear root document — typically an
  incident or validation reference in the upstream repo — the
  campaign log lives as a trailing section of that document. The
  2026-05-16 MSM-S sentinel incident's campaign log lives at the
  end of `~/Github/metforce/docs/msm_s_archive_validation.md`.
  When you execute the
  `docs/directions/20260516_tb_wnd_metforce_2020_rebuild.md`
  direction (the xfvcom-side downstream task in that campaign),
  that campaign log is the meta-context document.
* If a future campaign is xfvcom-rooted with no upstream root doc,
  fall back to `docs/campaigns/YYYYMMDD_<topic-slug>.md`. These
  files are markdown but, like `docs/directions/`, are **not**
  indexed by Sphinx — they are delegation / orchestration
  artefacts, not part of the published xfvcom user manual.
* **Campaign logs are mutable.** Unlike direction files
  (immutable once committed), a campaign log is a living record
  of an ongoing or recently-closed arc. Date-stamp each material
  update inside the log itself.
* Sibling-repo direction files cross-link back to the campaign
  log so a delegated session reading just its direction can find
  the meta-context.
