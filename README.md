# xfvcom

**xfvcom** is a Python package designed to streamline preprocessing and postprocessing for the Finite Volume Community Ocean Model ([FVCOM](https://github.com/FVCOM-GitHub/FVCOM)). Built on top of [xarray](https://docs.xarray.dev/en/stable/), this package simplifies large-scale ocean model data analysis and visualization. This package is under active development.

---

## Features

- **Load FVCOM data**: Easily load NetCDF-format FVCOM input and output files.
- **Coordinate transformation**: Convert between UTM coordinates and geographic coordinates (longitude/latitude).
- **Depth calculation**: Add depth variables based on water levels, sigma layers, and bathymetry.
- **Analysis tools**:
  - Find nearest nodes.
  - Filter variables based on specific dimensions.
- **Visualization**:
  - Create time-series plots.
  - Generate 2D contour and vector plots.
  - Produce animated GIFs for 2D spatial data.

---

## Installation

### Install in Development Mode

Follow these steps to install the package in development mode:

1. Clone the repository:
   ```bash
   git clone git@github.com:estuarine-utokyo/xfvcom.git
   cd xfvcom
   ```

2. Install required dependencies:
   - **Using conda**:
     ```bash
     conda install numpy xarray matplotlib pyproj scikit-learn
     conda install jupyterlab pandas hvplot
     ```
   - **Using pip**:
     ```bash
     pip install -r requirements.txt
     ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

---

## Usage

### Load Data
```python
from xfvcom import FvcomDataLoader

# Load FVCOM data
loader = FvcomDataLoader(base_path="/path/to/data", ncfile="sample.nc")
dataset = loader.ds
```

### Analyze Data
```python
from xfvcom import FvcomAnalyzer

# Initialize analyzer
analyzer = FvcomAnalyzer(dataset)
nearest_node = analyzer.nearest_neighbor(lon=140.0, lat=35.0)
print(f"Nearest node index: {nearest_node}")
```

### Plot Data

#### Time-Series Plot
```python
from xfvcom import FvcomPlotConfig, FvcomPlotter

# Configure plot settings
plot_config = FvcomPlotConfig(figsize=(8, 2), dpi=300)
plotter = FvcomPlotter(dataset, plot_config)

# Plot time series
time_series_plot = plotter.plot_timeseries(
    var_name="zeta", index=nearest_node, start="2020-01-01", end="2020-12-31", rolling_window=25
)
```

#### Generate 2D GIF Animation
```python
from xfvcom.plot_utils import create_gif_anim_2d_plot

# Generate a GIF animation
create_gif_anim_2d_plot(
    plotter=plotter,
    var_name="salinity",
    siglay=0,
    fps=10,
    post_process_func=None,
    plot_kwargs={"vmin": 28, "vmax": 34, "cmap": "jet"}
)
```

---

## Dependencies

This package depends on the following libraries:

- `numpy`
- `xarray`
- `matplotlib`
- `pyproj`
- `scikit-learn`
- `imageio`

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

---

## Developer Information

### Run Tests
Run tests using `pytest`:
```bash
pytest tests/
```

### Contributing
Contributions are welcome! Follow these steps to submit a pull request:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Submit a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## File Relationships

- `xfvcom.py`: Core module for loading, analyzing, and plotting FVCOM data.
- `helpers.py`: Provides helper functions for GIF generation, frame creation, and batch plotting.
- `helpers_utils.py`: Utility functions for cleaning and unpacking keyword arguments.
- `plot_utils.py`: Focuses on creating 2D plot animations (GIFs) by integrating `helpers.py` and `xfvcom.py`.
