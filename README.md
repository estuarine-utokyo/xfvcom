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
* [Forcing File Generator](docs/forcing_generator.md)

## Installation

This package is currently intended for **local development only** and is **not** published on PyPI.

### Prerequisites

* Miniforge (or compatible conda distribution)
* Python ≥ 3.10

### Create and activate conda environment

```bash
conda create -n xfvcom python=3.12 -c conda-forge \
    numpy xarray pandas matplotlib cartopy pyproj \
    scipy scikit-learn imageio moviepy tqdm \
    pytest mypy black isort jinja2 pyyaml types-pyyaml
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

### Extended forcing inputs

#### CLI one-liner

```bash
make_river_nc.py rivers_minimal.nml \
  --start 2025-01-01T00:00Z \
  --end   2025-01-02T00:00Z \
  --dt    3600 \
```

The NML file given above must exist on disk; otherwise
``RiverNetCDFGenerator`` will raise ``FileNotFoundError``.

Input CSV/TSV values are assumed to be in ``Asia/Tokyo`` by default and
converted to UTC.  Use ``--data-tz`` to override this.  For example a
timestamp ``2025-01-01T00:00`` with ``--data-tz Asia/Tokyo`` becomes
``2024-12-31T15:00Z`` in the output NetCDF file.

## Dependencies

All runtime dependencies are listed in **pyproject.toml** under `[project.dependencies]`. To add new packages, use:

```bash
conda install -c conda-forge <package>
```

---

## Developer Information

### Contributing

Contributions are welcome! Please fork the repo, create a feature branch, and submit a pull request.
For development workflow and test-suite details, please see
[CONTRIBUTING.md](docs/CONTRIBUTING.md).

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## File Relationships

* **helpers.py**  
  High-level API for GIF/MP4 creation, frame generation, and batch plotting
  (e.g., `FrameGenerator`).

* **helpers_utils.py**  
  Utility helpers for cleaning and merging option dictionaries used across
  the code base.

* **utils.py**  
  Wrapper functions that combine `helpers.py` and `core.py` to create
  sequences of 2-D plots and export them as animations (GIF/MP4).

* **plot_options.py**  
  `FvcomPlotOptions` dataclass holding all styling and plotting options.

* **core.py**  
  Main drawing routines implemented in `FvcomPlotter` (scalar contours,
  vector fields, vertical sections, etc.).

* **io.py**  
  `FvcomDataLoader` for reading FVCOM NetCDF files, completing the mesh, and
  performing light preprocessing.

* **config.py**  
  `FvcomPlotConfig` with default fonts, projections, and other constants that
  `core.py` refers to.

* **analysis.py**  
  `FvcomAnalyzer` providing physics-oriented calculations and reductions
  (layer averages, tidal decomposition, etc.).
