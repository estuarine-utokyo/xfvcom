# FVCOM Groundwater NetCDF Generation Examples

This directory contains examples for creating groundwater forcing NetCDF files for FVCOM using xfvcom.

## Overview

FVCOM's groundwater module simulates submarine groundwater discharge (SGD) into coastal waters. The module requires NetCDF files containing:
- Groundwater flux velocity (m/s) - NOT volumetric flux!
- Temperature (degC)
- Salinity (PSU)
- Optional: Dye concentration

**Important**: FVCOM expects groundwater flux as a velocity (m/s), which it multiplies by the node's bottom area internally. Do NOT provide volumetric flux (m³/s).

## Files in this Directory

### 1. `create_groundwater_netcdf.py`
Standalone script that creates groundwater NetCDF files using the low-level netCDF4 package. This is useful for understanding the exact NetCDF format required by FVCOM.

**Features:**
- Creates FVCOM-compatible NetCDF files
- Supports dye concentration (optional)
- Uses Modified Julian Day (MJD) time format
- Handles both Cartesian and Geographic coordinates

**Usage:**
```python
from create_groundwater_netcdf import create_groundwater_netcdf

# Create groundwater forcing with specific nodes
create_groundwater_netcdf(
    grid_file="path/to/grid.dat",
    output_file="groundwater.nc",
    start_datetime="2024-01-01 00:00:00",
    end_datetime="2024-01-03 00:00:00",
    time_interval_seconds=3600,
    active_nodes=[637, 638, 639, 662],  # 1-based Fortran indices
    flux_value=1e-6,  # m/s (velocity, not volumetric flux!)
    temperature_value=10.0,  # degC
    salinity_value=0.0,  # PSU
    dye_value=100.0,  # optional
    coordinate_system="Cartesian"
)
```

### 2. `add_dye_to_groundwater.py`
Script to add dye tracer to existing groundwater NetCDF files. Useful for tracer studies.

**Usage:**
```bash
python add_dye_to_groundwater.py groundwater.nc --dye-concentration 100.0
```

### 3. `create_groundwater_timeseries.py`
Generate time-varying groundwater forcing data. Creates CSV files that can be used with the CLI tools.

**Usage:**
```bash
python create_groundwater_timeseries.py
```

### 4. `groundwater_cli_example.sh`
Shell script showing how to use xfvcom CLI tools and create configuration files.

## Using xfvcom's Integrated Generator

The xfvcom package includes a groundwater NetCDF generator that integrates with the package's infrastructure:

```python
from xfvcom.io.groundwater_nc_generator import GroundwaterNetCDFGenerator
import numpy as np

# Create generator
gen = GroundwaterNetCDFGenerator(
    grid_nc="grid.nc",  # or .dat file
    start="2024-01-01T00:00:00Z",
    end="2024-01-03T00:00:00Z",
    dt_seconds=3600,
    flux=1e-6,  # constant flux for all nodes
    temperature=10.0,
    salinity=0.0
)

# Write to file
gen.write("groundwater.nc")
```

## FVCOM Namelist Configuration

Add this to your FVCOM namelist to use groundwater forcing:

```fortran
&NML_GROUNDWATER
GROUNDWATER_ON       = T,
GROUNDWATER_TEMP_ON  = T,
GROUNDWATER_SALT_ON  = T,
GROUNDWATER_DYE_ON   = T,  ! If using dye
GROUNDWATER_KIND     = 'variable',
GROUNDWATER_FILE     = 'groundwater.nc',
GROUNDWATER_FLOW     = 0.0,  ! Default values
GROUNDWATER_TEMP     = 10.0,
GROUNDWATER_SALT     = 0.0,
GROUNDWATER_DYE      = 100.0
/

&NML_DYE_RELEASE  ! Required if using dye
DYE_ON = T,
DYE_RELEASE_START = '2024-01-01 00:00:00',
DYE_RELEASE_STOP  = '2024-01-03 00:00:00',
KSPE_DYE = 0,
MSPE_DYE = 0
/
```

## Key Points

1. **Node Indices**: FVCOM uses 1-based (Fortran) indexing. When specifying active nodes, use 1-based indices.

2. **Time Format**: FVCOM expects Modified Julian Day (MJD) format with reference date 1858-11-17.

3. **Flux Units**: Groundwater flux is in m/s (velocity), NOT m³/s! FVCOM multiplies by node area internally.

4. **Active Nodes**: Only specified nodes have non-zero flux. All other nodes have zero flux.

5. **Grid Files**: Can use either `.dat` (ASCII) or `.nc` (NetCDF) grid files.

## Converting Volumetric Flux to Velocity

Since FVCOM expects velocity (m/s) but you may have volumetric flux data (m³/s), you need to convert:

```python
# If you have volumetric flux Q (m³/s) and need velocity:
# velocity = Q / node_area

# Example: Extract node areas from FVCOM output
import netCDF4 as nc
ds = nc.Dataset('fvcom_output.nc')
art1 = ds.variables['art1'][:]  # Node areas in m²

# Convert 0.1 m³/s discharge at node 637 to velocity
node_idx = 636  # Python 0-based index
Q = 0.1  # m³/s
velocity = Q / art1[node_idx]  # m/s
print(f"Velocity: {velocity:.6f} m/s for area {art1[node_idx]:.1f} m²")
```

## Example: Submarine Groundwater Discharge

To simulate SGD at specific coastal nodes:

```python
# Nodes where groundwater enters (1-based)
sgd_nodes = [637, 638, 639, 662]

# Create forcing file
create_groundwater_netcdf(
    grid_file="coastal_grid.nc",
    output_file="sgd_forcing.nc",
    start_datetime="2024-01-01 00:00:00",
    end_datetime="2024-12-31 23:00:00",
    time_interval_seconds=3600,  # Hourly
    active_nodes=sgd_nodes,
    flux_value=1e-5,  # m/s velocity (to get volumetric flux, multiply by node area)
    temperature_value=15.0,  # Groundwater temperature
    salinity_value=0.5,  # Low salinity
    dye_value=100.0  # Tracer concentration
)
```

## Troubleshooting

1. **File Format**: Ensure the NetCDF file uses NETCDF4_CLASSIC format for FVCOM compatibility.

2. **Coordinate System**: Match the coordinate system (Cartesian/Geographic) with your grid file.

3. **Time Steps**: FVCOM interpolates between time steps, so hourly data is usually sufficient.

4. **Memory**: For long simulations with many nodes, consider creating files in chunks.

## References

- FVCOM User Manual - Groundwater Module
- `FVCOM/docs/notes/groundwater_dye_user_guide.md`
- `FVCOM/docs/FVCOM_Groundwater_Discharge_Modeling.md`