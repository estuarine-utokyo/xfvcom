#!/usr/bin/env python3
"""Generate groundwater forcing using xfvcom Python API."""

from pathlib import Path

import numpy as np

from xfvcom.io.groundwater_nc_generator import GroundwaterNetCDFGenerator

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
    salinity=0.0,
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
print(f"Flux velocity: {1e-6:.2e} m/s at {len(active_nodes)} nodes")
