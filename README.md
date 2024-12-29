# xfvcom

**xfvcom** is a Python package designed to streamline preprocessing and postprocessing for the Finite Volume Community Ocean Model ([FVCOM](https://github.com/FVCOM-GitHub/FVCOM)). Built on top of [xarray](https://docs.xarray.dev/en/stable/), this package simplifies large-scale ocean model data analysis and visualization. This package is under construction.

---

## Features

- **Load FVCOM data**: Easily load NetCDF-format FVCOM input and output files.
- **Coordinate transformation**: Convert between UTM coordinates and geographic coordinates (longitude/latitude).
- **Depth calculation**: Add depth variables based on water levels, sigma layers, and bathymetry.
- **Analysis tools**: Find nearest nodes and filter variables based on specific dimensions.
- **Visualization**: Create time-series plots, vector plots, and more.

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
     # Recommended below as well
     conda install jupyterlab pandas hvplot
     ```
   - **Using pip**:
     ```bash
     pip install -r requirements.txt
     ```

3. Install the package in development mode:
   ```bash
   pip install -e .           # Most common
   pip install --no-deps -e . # Disable pip dependency resolution and install only the development package
   ```

---

## Usage

### Load Data
```python
from xfvcom import FvcomDataLoader

# Load FVCOM data
loader = FvcomDataLoader(dirpath="/path/to/data", ncfile="sample.nc")
dataset = loader.ds
```

### Find Nearest Node
```python
from xfvcom import FvcomAnalyzer

# Initialize analyzer
analyzer = FvcomAnalyzer(dataset)
nearest_node = analyzer.nearest_neighbor(lon=140.0, lat=35.0)
print(f"Nearest node index: {nearest_node}")
```

### Plot Time-Series Data
```python
from xfvcom import FvcomPlotConfig, FvcomPlotter

# Configure plot settings
plot_config = FvcomPlotConfig(figsize=(8, 2), dpi=300)
plotter = FvcomPlotter(dataset, plot_config)

# Plot time series
time_series_plot = plotter.plot_time_series(var_name="zeta", index=nearest_node, start="2020-01-01", end="2020-12-31", rolling_window=25)
```

---

## Dependencies

This package depends on the following libraries:

- `numpy`
- `xarray`
- `matplotlib`
- `pyproj`
- `scikit-learn`

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



