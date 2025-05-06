# xfvcom

**xfvcom** is a Python package designed to streamline preprocessing and postprocessing for the Finite Volume Community Ocean Model ([FVCOM](https://github.com/FVCOM-GitHub/FVCOM)). Built on top of [xarray](https://docs.xarray.dev/en/stable/), this package simplifies large-scale ocean model data analysis and visualization. This package is under active development.

---

## Features

* **Load FVCOM data**
  Easily load NetCDF-format FVCOM input and output files.
* **Coordinate transformation**
  Convert between UTM coordinates and geographic coordinates (longitude/latitude).
* **Depth calculation**
  Add depth variables based on water levels, sigma layers, and bathymetry.
* **Analysis tools**

  * Find nearest nodes
  * Filter variables based on specific dimensions
* **Visualization**

  * Create time-series plots
  * Generate 2D contour and vector plots
  * Produce animated GIFs and MP4s for 2D spatial data

---

## Documentation

* [2-D Horizontal Plotting](docs/plot_2d.md)
* [Time-series & Depth-profiles](docs/plot_ts.md)
* [Vertical Sections](docs/plot_section.md)

## Installation

This package is currently intended for **local development only** and is **not** published on PyPI.

### Prerequisites

* Miniforge (or compatible conda distribution)
* Python ≥ 3.10

### Create and activate conda environment

```bash
conda create -n xfvcom python=3.10 -c conda-forge \
    numpy xarray pandas matplotlib cartopy pyproj \
    scipy scikit-learn imageio moviepy tqdm \
    pytest mypy black isort
conda activate xfvcom
```

### Install xfvcom in editable mode

```bash
pip install -e .
```

---

## Usage

### Load Data

```python
from xfvcom import FvcomDataLoader

fvcom = FvcomDataLoader(base_path="/path/to/data", ncfile="sample.nc")
ds = fvcom.ds
```

### Analyze Data

```python
from xfvcom import FvcomAnalyzer

analyzer = FvcomAnalyzer(ds)
idx = analyzer.nearest_neighbor(lon=140.0, lat=35.0)
print(f"Nearest node index: {idx}")
```

### Time-Series Plot

```python
from xfvcom import FvcomPlotConfig, FvcomPlotter

cfg = FvcomPlotConfig(figsize=(8, 2), dpi=300)
plotter = FvcomPlotter(ds, cfg)
fig = plotter.ts_contourf("zeta", index=idx, start="2020-01-01", end="2020-12-31")
```

### 2D GIF/MP4 Animation

```python
from xfvcom.plot_utils import create_anim_2d_plot

create_anim_2d_plot(
    plotter=plotter,
    var_name="salinity",
    siglay=0,
    fps=10,
    post_process_func=None,
    plot_kwargs={"vmin": 28, "vmax": 34, "cmap": "jet"}
)
```

## Dependencies

All runtime dependencies are listed in **pyproject.toml** under `[project.dependencies]`. To add new packages, use:

```bash
conda install -c conda-forge <package>
```

---

## Developer Information

### Run Tests & Quality Checks

```bash
# 1) Unit tests
pytest tests/

# 2) Static type checking
mypy .

# 3) Code formatting and import order
black --check .
isort --check-only .
```

### Contributing

Contributions are welcome! Please fork the repo, create a feature branch, and submit a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## File Relationships

* **xfvcom.py**
  Core module for loading, analyzing, and plotting FVCOM data.
* **helpers.py**
  Provides helper functions for GIF/MP4 generation, frame creation, and batch plotting.
* **helpers\_utils.py**
  Utility functions for cleaning and unpacking keyword arguments.
* **plot\_utils.py**
  Creates 2D plot animations (GIF/MP4) by integrating `helpers.py` and `xfvcom.py`.
* **plot\_options.py**
  Configuration dataclass for plot styling and options.
* **core.py**
  Main plotting routines (contour, vector, section).
* **io.py**
  Data loader for FVCOM NetCDF files.
