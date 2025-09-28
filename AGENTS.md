# Repository Guidelines

## Project Structure & Module Organization
- `xfvcom/`: Core Python package with analysis utilities, CLI entrypoints, grid helpers, plotting adapters; extend here.
- `tests/`: Pytest suite with fixtures in `tests/data` and image baselines under `tests/baseline`; mirror new code paths.
- `docs/`: Task guides and workflow notes; update when CLI, IO, or configuration behavior shifts.
- `examples/`: Lightweight notebooks and scripts showing common pipelines; keep runs deterministic and fast.
- `tools/`: Standalone maintenance scripts such as `extract_fvcom_boundary.py` for mesh prep.
- `environment.yml`, `setup.sh`: Reference builds for the Conda environment; refresh when dependencies move.

## Build, Test, and Development Commands
- `./setup.sh`: Provision the standard Conda env and install the package editable.
- `pip install -e .[dev]`: Alternate setup with developer extras (`pytest`, `mypy`, `ruff`, `black`, `isort`).
- `pytest`: Execute the suite; append `-m "not png"` to skip image regressions.
- `ruff check .`: Run lint rules mirrored in CI.
- `black . && isort .`: Apply formatting (88-char lines) and canonical imports.
- `mypy --config-file mypy.ini xfvcom`: Type-check against the shared Python 3.12 profile.

## Coding Style & Naming Conventions
- Use Python 3.10+ idioms, four-space indentation, and `snake_case` for modules and functions.
- Reserve `PascalCase` for classes and configuration objects.
- Keep CLI entrypoints lowercase with hyphenation, e.g., `xfvcom-make-river-nc`.
- Document complex routines with concise docstrings and reference related `docs/` pages.

## Testing Guidelines
- Co-locate focused unit tests with their modules and add regression coverage for IO or plotting changes.
- Reuse fixtures in `tests/data`; keep large NetCDF assets compressed.
- Mark image comparisons with `@pytest.mark.png` and skip them in CI when backends diverge.
- Parameterize CLI tests so each option executes at least once.

## Commit & Pull Request Guidelines
- Follow `<type>: <imperative>` commits (e.g., `feat: Add KDTree helper`) and keep scope tight.
- Reference FVCOM tickets or GitHub issues when relevant.
- PRs need a concise summary, verification evidence (`pytest`, `ruff`, `mypy`), and screenshots when outputs change.
- Highlight dependency updates or migration steps for downstream consumers.

## Environment & Security Tips
- Prefer the managed Conda env; document deliberate deviations.
- Never commit credentials or API keys; rely on ignored `.env` files.
- Update `environment.yml` and `setup.sh` together after dependency changes.
