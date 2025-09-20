# Repository Guidelines

## Project Structure & Module Organization
- `xfvcom/`: Core library modules covering analysis utilities, CLI entry points, grid helpers, plotting, and shared decorators; import from here when adding features.
- `tests/`: Pytest suite with regression helpers, fixture data under `tests/data`, and image baselines in `tests/baseline`—mirror new features with matching test modules.
- `docs/`: Task-focused guides (e.g., plotting, forcing generators) that should be updated when workflows change.
- `examples/`: Ready-to-run notebooks and scripts that demonstrate common processing patterns; keep them lightweight and reproducible.
- `environment.yml`, `setup.sh`: Provision reproducible development environments; refresh versions here when dependencies shift.

## Build, Test & Development Commands
- `./setup.sh`: Create the default Conda environment and install the package in editable mode.
- `pip install -e .[dev]`: Manual setup that installs the library plus developer extras (`pytest`, `mypy`, `black`, `isort`, `ruff`).
- `pytest`: Run the full automated suite; add `-m "not png"` when skipping image-based regression tests.
- `ruff check .`: Static analysis aligned with the repo lint rules.
- `black . && isort .`: Auto-format code with an 88-character line length and Black-compatible import ordering.
- `mypy --config-file mypy.ini xfvcom`: Type-check new code paths using the shared configuration (Python 3.12 target, relaxed import checks).
- `pytest && black --check . && isort --check-only . && mypy xfvcom`: Mirror the GitHub CI gate locally before committing.

## Coding Style & Naming Conventions
- Follow Python 3.10+ idioms; prefer type annotations and dataclasses where appropriate.
- Use 4-space indentation, `snake_case` for functions and modules, and `PascalCase` for classes and configuration objects.
- Keep CLI commands and script names lowercase with hyphens (`xfvcom-make-river-nc`).
- Document complex routines with concise docstrings; cross-link to docs pages when behavior mirrors published guides.

## Testing Guidelines
- Mirror production scenarios with fixtures stored in `tests/data`; keep large NetCDF assets compressed and reference them via fixtures.
- Prefer focused unit tests next to related modules and add regression coverage for plotting or IO changes.
- Use the `png` marker sparingly for image comparisons and skip them in CI when they require system-specific backends.
- Validate new CLI options with pytest parameterization so each code path is exercised.

## Commit & Pull Request Guidelines
- Match the existing conventional style: `<type>: <short imperative>` (e.g., `feat: Add KDTree helper`).
- Scope commits narrowly, referencing FVCOM ticket IDs or GitHub issues in the body when applicable.
- Open PRs with a clear summary, testing evidence (`pytest`, `ruff`, `mypy` runs), and screenshots or sample plots when visual output changes.
- Highlight any dependency updates or migration steps so downstream users can adjust their environments safely.
