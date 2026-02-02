# Repository Guidelines

## Project Structure & Module Organization
- `xfvcom/`: Core Python package with analysis utilities, CLI entrypoints, grid helpers, plotting adapters; extend here.
- `tests/`: Pytest suite with fixtures in `tests/data` and image baselines under `tests/baseline`; mirror new code paths.
- `docs/`: Task guides and workflow notes; update when CLI, IO, or configuration behavior shifts.
- `examples/`: Lightweight notebooks and scripts showing common pipelines; keep runs deterministic and fast.
- `tools/`: Standalone maintenance scripts such as `extract_fvcom_boundary.py` for mesh prep.
- `environment.yml`, `setup.sh`: Reference builds for the Conda environment; refresh when dependencies move.

## Build, Test, and Development Commands
- `bash setup.sh`: Provision the `xfvcom` Conda env (Python 3.13) and install the package editable. Use `--impi` for Intel MPI.
- `pip install -e .[dev]`: Alternate setup with developer extras (not recommended; use mamba).
- `pytest`: Execute the suite; append `-m "not png"` to skip image regressions.
- `black . && isort .`: Apply formatting (88-char lines) and canonical imports.
- `mypy .`: Type-check against the shared Python 3.13 profile.
- `ruff check .`: Run lint rules (informational only, not enforced in CI).

## Coding Style & Naming Conventions
- Use Python 3.13+ idioms, four-space indentation, and `snake_case` for modules and functions.
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
- Update `environment.yml`, `setup.sh`, and `pyproject.toml` together after dependency changes.
