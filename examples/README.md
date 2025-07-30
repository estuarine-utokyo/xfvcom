# xfvcom Examples

This directory contains example scripts and notebooks demonstrating how to use xfvcom for FVCOM pre- and post-processing tasks.

## Directory Structure

```
examples/
├── configs/          # Configuration files (YAML, NML)
├── data/            # Sample data files (CSV, TSV)
├── input/           # FVCOM grid/mesh files (DAT)
├── notebooks/       # Jupyter notebook tutorials
└── *.py            # Python example scripts
```

## Quick Start

### Python Scripts

The Python scripts in the root examples directory demonstrate specific tasks:

- **`create_groundwater_netcdf.py`** - Create FVCOM groundwater forcing NetCDF files
- **`add_dye_to_groundwater.py`** - Add dye tracer to existing groundwater NetCDF files
- **`create_groundwater_timeseries.py`** - Generate time-varying groundwater forcing
- **`create_anim_2d.py`** - Create 2D animations from FVCOM output
- **`create_gif_anim_2d_from_frames.py`** - Convert animation frames to GIF

Run any script with:
```bash
python script_name.py
```

### Configuration Files

The `configs/` directory contains example configuration files:

- **Groundwater forcing configs**:
  - `groundwater_config.yaml` - Basic constant groundwater forcing
  - `groundwater_timevar.yaml` - Time-varying groundwater forcing
  - `groundwater_advanced_config.yaml` - Advanced configuration options
  - `groundwater_namelist.nml` - FVCOM namelist snippet for groundwater

- **River forcing configs**:
  - `rivers_minimal.nml` - Minimal river forcing namelist

### Sample Data

The `data/` directory contains sample data files:

- `groundwater_data.csv` - Example groundwater time series data
- `arakawa_flux.csv` - River flux data example
- `short_wave.csv` - Meteorological data example
- `sample_sjis.tsv` - Example of handling different encodings

### Grid Files

The `input/` directory contains FVCOM grid files:

- `chn_*.dat` - Channel test case grid files (coordinates, depth, sigma levels, etc.)

### Jupyter Notebooks

The `notebooks/` directory contains interactive tutorials:

- **Forcing Generation**:
  - `groundwater_generator.ipynb` - Interactive groundwater forcing generation
  - `forcing_generator.ipynb` - General forcing file creation
  - `input_river.ipynb` - River forcing setup

- **Visualization**:
  - `Ex1_plot_2d.ipynb` - Basic 2D plotting
  - `plot_chn.ipynb` - Channel test case visualization
  - `create_anim_2d.ipynb` - Animation creation tutorial

- **Development/Advanced**:
  - `dev_fvcom2d.ipynb` - FVCOM 2D mode examples
  - `dev_fvcom_grid.ipynb` - Grid manipulation
  - `obc_dye.ipynb` - Open boundary condition with dye

## Command Line Examples

See `groundwater_cli_example.sh` for comprehensive command-line usage examples.

## Getting Started

1. **Install xfvcom**: Follow the installation instructions in the main README

2. **Run a simple example**:
   ```bash
   # Create a basic groundwater forcing file
   python create_groundwater_netcdf.py
   ```

3. **Explore notebooks**:
   ```bash
   jupyter notebook notebooks/groundwater_generator.ipynb
   ```

## Common Tasks

### Creating Groundwater Forcing

```bash
# Using CLI
xfvcom-make-groundwater-nc input/chn_grd.dat \
  --utm-zone 33 \
  --start 2024-01-01T00:00Z \
  --end 2024-01-10T00:00Z \
  --flux 1e-6 \
  --temperature 10.0 \
  --salinity 0.0 \
  -o groundwater.nc

# Using Python API
python create_groundwater_netcdf.py
```

### Adding Dye to Groundwater

```bash
python add_dye_to_groundwater.py groundwater.nc
```

### Creating Animations

```bash
python create_anim_2d.py fvcom_output.nc temperature
```

## Notes

- Most examples use the channel test case (`chn`) as a simple demonstration grid
- File paths in scripts assume they are run from the examples directory
- For production use, adapt the examples to your specific FVCOM setup