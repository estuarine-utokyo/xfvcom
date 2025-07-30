#!/bin/bash
# Example: Using xfvcom CLI to generate groundwater NetCDF files

echo "=== FVCOM Groundwater NetCDF Generation Example ==="
echo
echo "This example demonstrates how to generate groundwater forcing files"
echo "for FVCOM using the xfvcom command-line tools."
echo

# Example 1: Basic constant groundwater forcing
echo "Example 1: Constant groundwater forcing"
echo "--------------------------------------"
cat > groundwater_config.yaml << EOF
# Groundwater forcing configuration
grid_file: ../FVCOM/Tests/GroundwaterDye/input/chn_grd.dat
output_file: groundwater_constant.nc
start_time: "2024-01-01T00:00:00Z"
end_time: "2024-01-03T00:00:00Z"
time_interval: 3600  # seconds

# Active nodes (1-based Fortran indices)
active_nodes: [637, 638, 639, 662]

# Constant values
groundwater:
  flux: 1.0e-6      # m/s (velocity, not volumetric flux!)
  temperature: 10.0  # degC
  salinity: 0.0     # PSU
  dye: 100.0        # concentration (optional)

# Coordinate system
coordinate_system: "Cartesian"
utm_zone: 33
northern: true
EOF

echo "Configuration saved to groundwater_config.yaml"
echo
echo "To generate the NetCDF file, run:"
echo "  xfvcom-make-groundwater-nc groundwater_config.yaml"
echo

# Example 2: Time-varying groundwater
echo "Example 2: Time-varying groundwater forcing"
echo "------------------------------------------"
cat > groundwater_timevar.yaml << EOF
# Time-varying groundwater forcing
grid_file: ../FVCOM/Tests/GroundwaterDye/input/chn_grd.dat
output_file: groundwater_timevar.nc
start_time: "2024-01-01T00:00:00Z"
end_time: "2024-01-10T00:00:00Z"
time_interval: 1800  # 30 minutes

# This would read from a data file with time series
groundwater_kind: "variable"
groundwater_file: "groundwater_timeseries.csv"

# OR specify a function (tidal variation example)
groundwater:
  flux_function: "1e-6 * (1 + 0.5 * sin(2*pi*t/24))"  # Diurnal variation
  temperature: 10.0
  salinity: 0.0
EOF

echo "Time-varying configuration saved to groundwater_timevar.yaml"
echo

# Example 3: Using Python API directly
echo "Example 3: Using Python API"
echo "--------------------------"
cat > generate_groundwater.py << 'EOF'
#!/usr/bin/env python3
"""Generate groundwater forcing using xfvcom Python API."""

from pathlib import Path
from xfvcom.io.groundwater_nc_generator import GroundwaterNetCDFGenerator
import numpy as np

# Configuration
grid_file = Path("../FVCOM/Tests/GroundwaterDye/input/chn_grd.dat")
if not grid_file.exists():
    grid_file = Path("tests/data/taiya_grd.dat")  # Fallback

# Active nodes for groundwater discharge (1-based)
active_nodes = [637, 638, 639, 662]

# Create generator
gen = GroundwaterNetCDFGenerator(
    grid_nc=grid_file,
    start="2024-01-01T00:00:00Z",
    end="2024-01-03T00:00:00Z",
    dt_seconds=3600,
    utm_zone=33,
    flux=0.0,  # Will set per-node values
    temperature=10.0,
    salinity=0.0
)

# Load grid
gen.load()
node_count = gen.mesh_ds.sizes["node"]
time_count = len(gen.timeline)

# Create flux array with non-zero values only at active nodes
flux = np.zeros((node_count, time_count))
for node_id in active_nodes:
    if node_id <= node_count:
        flux[node_id - 1, :] = 1e-6  # Convert to 0-based index

gen.flux_data = flux

# Generate and save
content = gen.render()
with open("groundwater_api.nc", "wb") as f:
    f.write(content)

print(f"Generated groundwater_api.nc")
print(f"Active nodes: {active_nodes}")
print(f"Total discharge: {1e-6 * len(active_nodes):.2e} m³/s")
EOF

echo "Python script saved to generate_groundwater.py"
echo "Run with: python generate_groundwater.py"
echo

# Example 4: Namelist snippet for FVCOM
echo "Example 4: FVCOM Namelist Configuration"
echo "--------------------------------------"
cat > groundwater_namelist.nml << EOF
! Add this section to your FVCOM namelist file

&NML_GROUNDWATER
GROUNDWATER_ON       = T,               ! Enable groundwater
GROUNDWATER_TEMP_ON  = T,               ! Include temperature
GROUNDWATER_SALT_ON  = T,               ! Include salinity  
GROUNDWATER_DYE_ON   = T,               ! Include dye tracer
GROUNDWATER_KIND     = 'variable',      ! Read from NetCDF file
GROUNDWATER_FILE     = 'groundwater_constant.nc',
! These are used as defaults/fallbacks:
GROUNDWATER_FLOW     = 0.0,
GROUNDWATER_TEMP     = 10.0,
GROUNDWATER_SALT     = 0.0,
GROUNDWATER_DYE      = 100.0
/

! Also ensure dye module is enabled:
&NML_DYE_RELEASE  
DYE_ON = T,
DYE_RELEASE_START = '2024-01-01 00:00:00',
DYE_RELEASE_STOP  = '2024-01-03 00:00:00',
KSPE_DYE = 0,  ! No point sources
MSPE_DYE = 0   ! No point sources
/
EOF

echo "FVCOM namelist snippet saved to groundwater_namelist.nml"
echo

echo "=== Summary ==="
echo "The groundwater forcing for FVCOM requires:"
echo "1. A NetCDF file with groundwater flux, temperature, and salinity"
echo "2. Time dimension and coordinate variables (MJD format)"
echo "3. Data at all nodes (non-zero only at discharge locations)"
echo "4. Proper namelist configuration in FVCOM"
echo
echo "Active nodes for this example: 637, 638, 639, 662 (1-based)"
echo "These represent submarine groundwater discharge locations"