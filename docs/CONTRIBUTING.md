# Contributing to xfvcom

Thanks for your interest in improving **xfvcom**!
Below are the typical steps for setting up the development environment and contributing code.

---

## 1. Environment Setup

### Prerequisites

- [Miniforge](https://github.com/conda-forge/miniforge) (provides `mamba`)

### Create the development environment

```bash
git clone https://github.com/jsasaki-utokyo/xfvcom.git
cd xfvcom
bash setup.sh              # Creates "xfvcom" environment (Python 3.13)
conda activate xfvcom
```

For Intel MPI on supercomputers:

```bash
bash setup.sh --impi
```

This installs all dependencies (including dev tools) and xfvcom in editable mode.

---

## 2. Code Quality Checks

Run all CI-equivalent checks before committing:

```bash
black --check . && isort --check-only . && mypy . && pytest -m "not png"
```

Auto-fix formatting:

```bash
black . && isort .
```

> **Note**: Ruff is installed but NOT run in CI. Ruff warnings are informational only.

---

## 3. Running Tests

```bash
# Run tests (excluding PNG regression)
pytest -m "not png"

# Run all tests including image comparison
pytest

# Run a specific test
pytest tests/test_grid.py::TestGridCalculations -v
```

---

## 4. Update PNG Baselines

Only needed when plot appearance changes intentionally (e.g., after matplotlib/cartopy updates):

```bash
pytest --regenerate-baseline -q     # Rebuild baseline images
git add tests/baseline/*.png
git commit -m "Update image baselines"
```

---

## 5. Add Examples / Screenshots

1. Place images in `docs/images/`
2. Reference them from the relevant docs page (e.g., `docs/plot_2d.md`)

---

## 6. Commit & Pull Request Guidelines

- Follow `<type>: <imperative>` commits (e.g., `feat: Add KDTree helper`)
- Keep scope tight and PRs focused
- PRs need a concise summary, verification evidence (`pytest`, `mypy`), and screenshots when outputs change
- Reference GitHub issues when relevant

---

[Back to README](../README.md)
