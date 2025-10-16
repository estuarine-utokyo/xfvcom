# xfvcom Examples

This directory contains example scripts, notebooks, and tutorials demonstrating how to use xfvcom for FVCOM preprocessing and postprocessing tasks.

---

## 📂 Directory Structure

```
examples/
├── configs/          # Configuration files (YAML, NML)
├── data/             # Sample data files (CSV, TSV)
├── input/            # FVCOM grid/mesh files (DAT)
├── notebooks/        # Jupyter notebook tutorials
├── output/           # Example output files and visualizations
├── *.py              # Python example scripts
└── README*.md        # Documentation and guides
```

---

## 🚀 Quick Start

### 1. Python Scripts

Run example scripts directly from the examples directory:

```bash
cd examples/

# Create groundwater forcing
python create_groundwater_netcdf.py

# Extract member-node mapping from dye data
python extract_member_node_mapping.py

# Create 2D animations
python create_anim_2d.py

# Test dye time series extraction
python test_dye_timeseries.py
```

### 2. Jupyter Notebooks

Launch Jupyter and explore interactive tutorials:

```bash
jupyter notebook notebooks/
```

### 3. Command-Line Tools

Use xfvcom CLI tools (see [CLI Examples](#-command-line-tools)):

```bash
# Generate groundwater forcing
xfvcom-make-groundwater-nc input/chn_grd.dat --start 2024-01-01 --end 2024-12-31 --flux 1e-6 -o groundwater.nc

# Extract dye time series
xfvcom-dye-ts --years 2021 --members 1,2,3 --nodes 100,200 --save dye_series.nc
```

---

## 📚 Documentation Guides

### Specialized Guides

- **[Dye Time Series Extraction](README_DYE_TIMESERIES.md)** - Multi-year, multi-member dye concentration analysis
  - Time alignment strategies (intersection, same_calendar, climatology)
  - Negative value handling
  - Linearity verification
  - Python API and CLI usage

- **[Groundwater Forcing](README_groundwater.md)** - Creating groundwater discharge forcing files
  - Flux units and conversion (m/s vs m³/s)
  - Node-specific and time-varying forcing
  - FVCOM namelist configuration
  - Dye tracer integration

- **[CLI Examples](groundwater_cli_examples.md)** - Command-line usage patterns
  - Constant and time-varying data
  - CSV formats (wide and long)
  - Mixed constant/time series

---

## 📓 Jupyter Notebooks

### Dye Time Series and Ensemble Analysis

| Notebook | Description |
|----------|-------------|
| `demo_dye_timeseries.ipynb` | **Dye time series extraction and visualization** <br> Multi-member aggregation, ensemble statistics, stacked plots |
| `demo_member_node_mapping.ipynb` | **Identify dye release locations** <br> Extract member-node mapping, export to CSV with coordinates |

### Data Validation and QC

| Notebook | Description |
|----------|-------------|
| `demo_node_checker.ipynb` | **Node data validation** <br> Check node values, identify issues |
| `demo_river_input_checker.ipynb` | **River input validation** <br> Verify river discharge and temperature data |
| `demo_river_ts_extender.ipynb` | **Time series extension** <br> Extend river data using seasonal/linear patterns |

### Visualization

| Notebook | Description |
|----------|-------------|
| `Ex1_plot_2d.ipynb` | **Basic 2D plotting** <br> Surface plots, mesh visualization |
| `create_2d.ipynb` | **Advanced 2D plots** <br> Contours, vectors, map tiles |
| `create_anim_2d.ipynb` | **Animation creation** <br> Generate GIF/MP4 from time series |
| `plot_chn.ipynb` | **Channel test case** <br> Visualize FVCOM channel example |
| `plot_inout.ipynb` | **Inflow/outflow analysis** <br> Flux calculations and visualization |

### Forcing File Generation

| Notebook | Description |
|----------|-------------|
| `groundwater_generator.ipynb` | **Interactive groundwater forcing** <br> GUI-based groundwater file creation |
| `forcing_generator.ipynb` | **General forcing generation** <br> River, meteorological, and other forcing types |
| `input_river.ipynb` | **River forcing setup** <br> Configure river inputs |

### Development and Advanced

| Notebook | Description |
|----------|-------------|
| `dev_fvcom2d.ipynb` | **FVCOM 2D mode** <br> Depth-averaged model examples |
| `dev_fvcom_grid.ipynb` | **Grid manipulation** <br> Mesh editing and analysis |
| `obc_dye.ipynb` | **Open boundary conditions** <br> Dye tracer at boundaries |
| `fvcom_mpos.ipynb` | **MPOS integration** <br> Marine pollution observation system |

---

## 🐍 Python Scripts

### Ensemble Analysis

- **`extract_member_node_mapping.py`** - Extract dye release locations from ensemble members
- **`plot_dye_timeseries.py`** - Plot ensemble dye time series
- **`test_dye_timeseries.py`** - Test dye time series extraction

### Groundwater Forcing

- **`create_groundwater_netcdf.py`** - Create groundwater forcing NetCDF files
- **`add_dye_to_groundwater.py`** - Add dye tracer to existing groundwater files
- **`create_groundwater_timeseries.py`** - Generate time-varying groundwater forcing

### Visualization

- **`create_anim_2d.py`** - Create 2D animations from FVCOM output
- **`create_gif_anim_2d_from_frames.py`** - Convert animation frames to GIF
- **`seasonal_extension_visualizer.py`** - Visualize time series extension

### Utilities

- **`node_checker.py`** - Validate node data
- **`test_marker_buffer_demo.py`** - Demonstrate marker buffer control
- **`test_text_clipping_demo.py`** - Demonstrate text clipping fixes

---

## 🔨 Command-Line Tools

### Groundwater Forcing

```bash
# Constant values for all nodes
xfvcom-make-groundwater-nc grid.dat \
  --utm-zone 54 \
  --start 2024-01-01T00:00Z \
  --end 2024-12-31T23:00Z \
  --flux 1e-6 \
  --temperature 10.0 \
  --salinity 0.0 \
  -o groundwater.nc

# Time-varying from CSV
xfvcom-make-groundwater-nc grid.nc \
  --start 2024-01-01T00:00Z \
  --end 2024-12-31T23:00Z \
  --flux flux_timeseries.csv:datetime \
  --temperature temperature_timeseries.csv:datetime \
  --salinity 0.0
```

See [groundwater_cli_examples.md](groundwater_cli_examples.md) for comprehensive CLI usage.

### Dye Time Series Extraction

```bash
# Basic extraction
xfvcom-dye-ts \
  --years 2021 \
  --members 0,1,2 \
  --nodes 100,200,300 \
  --sigmas 0,1 \
  --save output/dye_series.nc

# With linearity verification
xfvcom-dye-ts \
  --years 2020,2021 \
  --members 0,1,2 \
  --nodes 100 \
  --sigmas 0 \
  --verify-linearity \
  --ref-member 0 \
  --parts 1,2 \
  --print-neg-stats
```

See [README_DYE_TIMESERIES.md](README_DYE_TIMESERIES.md) for detailed documentation.

### River and Meteorological Forcing

```bash
# River namelist
xfvcom-make-river-nml river_data.csv --output rivers.nml

# River forcing
xfvcom-make-river-nc rivers.nml \
  --start 2025-01-01T00:00Z \
  --end 2025-12-31T23:00Z \
  --dt 3600

# Meteorological forcing
xfvcom-make-met-nc grid.nc \
  --start 2025-01-01T00:00Z \
  --end 2025-01-07T00:00Z \
  --ts wind.csv:uwind,vwind \
  --air-temperature 20.0 \
  --humidity 0.7
```

---

## 📁 Configuration Files

### Groundwater (`configs/`)

- `groundwater_config.yaml` - Basic constant groundwater forcing
- `groundwater_timevar.yaml` - Time-varying groundwater forcing
- `groundwater_advanced_config.yaml` - Advanced configuration options
- `groundwater_namelist.nml` - FVCOM namelist snippet

### River (`configs/`)

- `rivers_minimal.nml` - Minimal river forcing namelist

---

## 📊 Sample Data

### CSV Files (`data/`)

- `groundwater_data.csv` - Example groundwater time series
- `arakawa_flux.csv` - River flux data
- `short_wave.csv` - Meteorological data
- `sample_sjis.tsv` - Japanese encoding example (Shift-JIS)

### Grid Files (`input/`)

- `chn_*.dat` - Channel test case grid files
  - `chn_grd.dat` - Grid coordinates
  - `chn_dep.dat` - Bathymetry
  - `chn_sigma.dat` - Sigma layers
  - `chn_cor.dat` - Coriolis parameters
  - `chn_obc.dat` - Open boundary conditions

---

## 📖 Common Workflows

### 1. Create Groundwater Forcing with Dye Tracer

```python
from xfvcom.io.groundwater_nc_generator import GroundwaterNetCDFGenerator

# Create generator
gen = GroundwaterNetCDFGenerator(
    grid_nc="input/chn_grd.dat",
    start="2024-01-01T00:00:00Z",
    end="2024-01-03T00:00:00Z",
    dt_seconds=3600,
    flux=1e-6,
    temperature=10.0,
    salinity=0.0,
    dye_concentration=100.0  # Optional tracer
)

# Write NetCDF
gen.write("groundwater_dye.nc")
```

### 2. Extract and Analyze Dye Ensemble

```python
from xfvcom.dye_timeseries import (
    DyeCase, Selection, Paths, NegPolicy, AlignPolicy,
    collect_member_files, aggregate, negative_stats
)

# Configure
paths = Paths(tb_fvcom_dir="/path/to/TB-FVCOM")
case = DyeCase(basename="tb_w18_r16", years=[2021], members=[1,2,3])
sel = Selection(nodes=[100, 200], sigmas=[0])

# Aggregate
member_map = collect_member_files(paths, case)
ds = aggregate(member_map, case, sel, NegPolicy("keep"), AlignPolicy("native_intersection"))

# Analyze
stats = negative_stats(ds)
print(f"Negative values: {stats['global']['count_neg']} / {stats['global']['total_samples']}")
```

### 3. Visualize Ensemble Time Series

```python
from xfvcom import FvcomPlotConfig
from xfvcom.plot import plot_ensemble_timeseries, plot_dye_timeseries_stacked

# Create config
cfg = FvcomPlotConfig(
    figsize=(14, 6),
    fontsize={"xlabel": 14, "ylabel": 14, "title": 15}
)

# Line plot
fig, ax = plot_ensemble_timeseries(
    ds,
    var_name="dye",
    cfg=cfg,
    title="Ensemble Dye Concentration"
)

# Stacked area plot
result = plot_dye_timeseries_stacked(
    ds,
    cfg=cfg,
    member_ids=[1, 2, 3, 4, 5],
    output="dye_stacked.png"
)
```

### 4. Identify Dye Release Locations

```python
from xfvcom.ensemble_analysis import (
    extract_member_node_mapping,
    export_member_mapping
)

# Extract mapping
mapping = extract_member_node_mapping(
    dye_files=["dye_member1.nc", "dye_member2.nc", ...],
    threshold=1e-6,
    grid_file="grid.dat"
)

# Export to CSV
export_member_mapping(
    mapping,
    "member_mapping.csv",
    grid_file="grid.dat"
)
```

---

## 🔍 Quick Reference

### Notebook Quick Reference

The `notebooks/QUICK_REFERENCE.md` file provides a quick reference for the `demo_dye_timeseries.ipynb` notebook, including:
- Configuration settings
- Key changes and fixes
- Testing commands
- Troubleshooting tips

### Important Notes

1. **Node Indexing**: FVCOM uses 1-based indexing. Python uses 0-based. Use `index_base` parameter.

2. **Time Zones**: Default input is Asia/Tokyo (JST), output is UTC. Configure with `data_tz` parameter.

3. **Groundwater Flux Units**: FVCOM expects velocity (m/s), NOT volumetric flux (m³/s)!
   ```python
   velocity = volumetric_flux / node_area
   ```

4. **Area Calculations**: Two methods available:
   - `calculate_node_area_median_dual()` - FVCOM-standard (preferred)
   - `calculate_node_area()` - Triangle sum (legacy)

5. **File Paths**: Examples assume they are run from the `examples/` directory

---

## 🧪 Testing Examples

### Run All Examples

```bash
cd examples/

# Test dye time series
python test_dye_timeseries.py

# Test member mapping
python extract_member_node_mapping.py

# Run notebook (non-interactive)
jupyter nbconvert --to notebook --execute notebooks/demo_dye_timeseries.ipynb
```

### Verify Outputs

```bash
# Check NetCDF file
ncdump -h output/dye_series.nc

# Inspect groundwater forcing
ncdump -v groundwater_flux groundwater.nc | head -50

# View member mapping
cat output/member_mapping.csv
```

---

## 🛠️ Troubleshooting

### Common Issues

**Issue**: Import errors when running scripts
```bash
# Solution: Install xfvcom in editable mode
cd /path/to/xfvcom
pip install -e .
```

**Issue**: Missing data files
```bash
# Solution: Check TB-FVCOM path
export TB_FVCOM_DIR=/path/to/TB-FVCOM
# Or use --tb-fvcom-dir flag
```

**Issue**: Groundwater file rejected by FVCOM
```bash
# Check: Flux units should be m/s (velocity), not m³/s
# Check: Time format should be Modified Julian Day
ncdump -h groundwater.nc
```

**Issue**: Empty time intersection in dye aggregation
```bash
# Solution: Use alternative alignment mode
xfvcom-dye-ts --align same_calendar  # or --align climatology
```

---

## 📝 Getting Help

1. **Main README**: See `/README.md` for package overview
2. **Documentation**: Check `/docs/` for detailed guides
3. **API Reference**: Use Python help:
   ```python
   from xfvcom import dye_timeseries
   help(dye_timeseries)
   ```
4. **Issues**: Report at https://github.com/estuarine-utokyo/xfvcom/issues

---

## 📄 License

MIT License - see main package LICENSE file

---

## 🔗 Related Documentation

- [Main README](../README.md) - Package overview
- [Contributing Guide](../docs/CONTRIBUTING.md) - Development guidelines
- [CLAUDE.md](../CLAUDE.md) - Developer guide for AI assistants
- [Forcing Generator](../docs/forcing_generator.md) - Detailed forcing documentation
- [2D Plotting](../docs/plot_2d.md) - 2D visualization guide
- [Time Series Plotting](../docs/plot_ts.md) - Time series visualization
