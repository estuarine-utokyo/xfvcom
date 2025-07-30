# xfvcom Groundwater CLI Examples

The `xfvcom-make-groundwater-nc` command generates groundwater forcing NetCDF files for FVCOM. Unlike the YAML configuration approach, this uses command-line arguments directly.

## Grid File Formats

FVCOM grid files are typically ASCII text files with `.dat` extension. However, xfvcom can also work with NetCDF grid files that have been preprocessed. Most users will use `.dat` files.

## Basic Usage

### 1. Constant Values for All Nodes (with .dat grid file)

```bash
xfvcom-make-groundwater-nc chn_grd.dat \
  --utm-zone 33 \
  --start 2024-01-01T00:00Z \
  --end 2024-01-10T00:00Z \
  --flux 0.0 \
  --temperature 10.0 \
  --salinity 0.0 \
  -o groundwater_constant.nc
```

This creates a file where ALL nodes have the same (zero) flux. To specify non-zero flux only at certain nodes, you need to use CSV files.

### 2. Using NetCDF Grid Files (preprocessed)

If you have a NetCDF grid file (e.g., preprocessed from a .dat file), you don't need to specify UTM zone:

```bash
xfvcom-make-groundwater-nc grid_preprocessed.nc \
  --start 2024-01-01T00:00Z \
  --end 2024-01-10T00:00Z \
  --flux 0.0 \
  --temperature 10.0 \
  --salinity 0.0
```

Note: NetCDF grid files must contain `x`, `y`, and `nv` variables.

## Time Series Data

### 3. CSV Files for Node-Specific Constant Values

Create a CSV file with flux values for specific nodes (all others will be zero):

**flux_by_node.csv:**
```csv
node_id,flux
637,1.0e-6
638,1.0e-6
639,1.0e-6
662,1.0e-6
```

**temperature_by_node.csv:**
```csv
node_id,temperature
637,10.0
638,10.5
639,11.0
662,9.5
```

Use them:
```bash
xfvcom-make-groundwater-nc chn_grd.dat \
  --utm-zone 54 \
  --start 2024-01-01T00:00Z \
  --end 2024-12-31T23:00Z \
  --flux flux_by_node.csv \
  --temperature temperature_by_node.csv \
  --salinity 0.0
```

### 4. Time-Varying Data (Wide Format)

For time series data, use CSV files with datetime column and one column per node:

**flux_timeseries.csv:**
```csv
datetime,637,638,639,662
2024-01-01 00:00:00,1.0e-6,1.2e-6,1.4e-6,1.6e-6
2024-01-01 01:00:00,1.1e-6,1.3e-6,1.5e-6,1.7e-6
2024-01-01 02:00:00,1.2e-6,1.4e-6,1.6e-6,1.8e-6
...
```

Use with `:datetime` suffix to indicate time series:
```bash
xfvcom-make-groundwater-nc chn_grd.dat \
  --utm-zone 33 \
  --start 2024-01-01T00:00Z \
  --end 2024-01-10T00:00Z \
  --flux flux_timeseries.csv:datetime \
  --temperature temperature_timeseries.csv:datetime \
  --salinity salinity_timeseries.csv:datetime \
  -o groundwater_timeseries.nc
```

### 5. Time-Varying Data (Long Format)

Alternative format with all data in one file:

**groundwater_data.csv:**
```csv
datetime,node_id,flux,temperature,salinity
2024-01-01 00:00:00,637,1.0e-6,10.0,0.0
2024-01-01 00:00:00,638,1.2e-6,10.5,0.5
2024-01-01 00:00:00,639,1.4e-6,11.0,0.3
2024-01-01 00:00:00,662,1.6e-6,9.5,1.0
2024-01-01 01:00:00,637,1.1e-6,10.1,0.0
...
```

Use with column specifications:
```bash
xfvcom-make-groundwater-nc chn_grd.dat \
  --utm-zone 33 \
  --start 2024-01-01T00:00Z \
  --end 2024-01-10T00:00Z \
  --flux groundwater_data.csv:datetime,node_id,flux \
  --temperature groundwater_data.csv:datetime,node_id,temperature \
  --salinity groundwater_data.csv:datetime,node_id,salinity
```

### 6. Mixed Constant and Time-Varying

You can mix constant values with time series:
```bash
xfvcom-make-groundwater-nc chn_grd.dat \
  --utm-zone 33 \
  --start 2024-01-01T00:00Z \
  --end 2024-01-10T00:00Z \
  --flux flux_timeseries.csv:datetime \
  --temperature 10.0 \
  --salinity 0.0
```

## Time Zone Handling

By default, times are interpreted as UTC. To use a different timezone:

```bash
xfvcom-make-groundwater-nc chn_grd.dat \
  --utm-zone 33 \
  --start 2024-01-01 \
  --end 2024-01-10 \
  --start-tz Asia/Tokyo \
  --flux flux_timeseries.csv:datetime \
  --temperature 10.0 \
  --salinity 0.0
```

Note: The output NetCDF file always uses UTC times.

## Complete Example

Here's a complete example using the test files we created:

```bash
# First, generate time series CSV files
python create_groundwater_timeseries.py

# Then use them with the CLI
xfvcom-make-groundwater-nc ../../FVCOM/Tests/GroundwaterDye/input/chn_grd.dat \
  --utm-zone 33 \
  --start 2024-01-01T00:00Z \
  --end 2024-01-10T00:00Z \
  --dt 3600 \
  --flux flux_timeseries.csv:datetime \
  --temperature temperature_timeseries.csv:datetime \
  --salinity salinity_timeseries.csv:datetime \
  -o groundwater_fvcom.nc
```

## Notes

1. **Node Indexing**: Node IDs in CSV files use 1-based (Fortran) indexing
2. **Missing Nodes**: Nodes not specified in CSV files will have zero flux
3. **Time Interpolation**: FVCOM interpolates between time steps
4. **File Size**: For long simulations, consider using coarser time steps (e.g., 6-hourly instead of hourly)
5. **Dye Support**: The current CLI doesn't support dye concentration; use the Python API for that

## Verifying Output

Check the generated NetCDF file:
```bash
ncdump -h groundwater_fvcom.nc  # Show header
ncdump -v groundwater_flux groundwater_fvcom.nc | head -50  # Show flux data
```

Or with Python:
```python
import netCDF4 as nc
import numpy as np

with nc.Dataset('groundwater_fvcom.nc', 'r') as ds:
    print(f"Dimensions: {dict([(d, len(ds.dimensions[d])) for d in ds.dimensions])}")
    print(f"Variables: {list(ds.variables.keys())}")
    
    # Check flux at specific nodes
    flux = ds.variables['groundwater_flux'][:]
    nodes = [637, 638, 639, 662]
    for node_id in nodes:
        node_flux = flux[node_id - 1, :]  # Convert to 0-based
        print(f"Node {node_id}: mean={np.mean(node_flux):.3e}, max={np.max(node_flux):.3e}")
```