#!/usr/bin/env python3
"""
Test script for FVCOM Groundwater NetCDF generation.

This script demonstrates creating a groundwater forcing file for FVCOM
using the specific nodes and parameters from the FVCOM test case.
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path to import xfvcom
sys.path.insert(0, str(Path(__file__).parent.parent))

import netCDF4 as nc
import numpy as np

from xfvcom.io.groundwater_nc_generator import GroundwaterNetCDFGenerator


def create_test_grid_nc(node_count=700):
    """Create a simple test grid NetCDF file."""
    # Create a simple rectangular grid
    nx, ny = 25, 28  # 25x28 = 700 nodes

    # Create node coordinates
    x = np.repeat(np.arange(nx) * 1000.0, ny)
    y = np.tile(np.arange(ny) * 1000.0, nx)

    # Simple triangulation (creates 2*(nx-1)*(ny-1) elements)
    elements = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            # Node indices for this quad
            n1 = i * ny + j
            n2 = (i + 1) * ny + j
            n3 = (i + 1) * ny + (j + 1)
            n4 = i * ny + (j + 1)

            # Lower triangle
            elements.append([n1, n2, n3])
            # Upper triangle
            elements.append([n1, n3, n4])

    nv = np.array(elements).T + 1  # Convert to 1-based and transpose

    # Create NetCDF file
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        grid_file = Path(tmp.name)

    with nc.Dataset(grid_file, "w") as ds:
        # Dimensions
        ds.createDimension("node", node_count)
        ds.createDimension("nele", nv.shape[1])
        ds.createDimension("three", 3)

        # Variables
        v_x = ds.createVariable("x", "f8", ("node",))
        v_x.units = "meter"
        v_x[:] = x

        v_y = ds.createVariable("y", "f8", ("node",))
        v_y.units = "meter"
        v_y[:] = y

        v_nv = ds.createVariable("nv", "i4", ("three", "nele"))
        v_nv[:] = nv

        # Add coordinate system info
        ds.CoordinateSystem = "Cartesian"

    return grid_file


def test_groundwater_generator():
    """Test the groundwater NetCDF generator with FVCOM test case parameters."""

    # Define active nodes (1-based Fortran indices)
    active_nodes = [637, 638, 639, 662]

    # Create a test grid NetCDF file
    print("Creating test grid NetCDF file...")
    grid_file = create_test_grid_nc(node_count=700)
    print(f"Created test grid: {grid_file}")
    print(f"Grid has 700 nodes (created to include nodes {active_nodes})")

    # Test 1: Using the integrated generator with constant values
    print("\n=== Test 1: Integrated Generator with Constant Values ===")

    # Create generator with constant values
    generator = GroundwaterNetCDFGenerator(
        grid_nc=grid_file,
        start="2024-01-01T00:00:00Z",
        end="2024-01-03T00:00:00Z",
        dt_seconds=3600,  # Hourly
        utm_zone=(
            33 if "gwdye" in str(grid_file) else None
        ),  # UTM zone 33 for FVCOM test
        flux=0.0,  # Initialize all nodes to zero flux
        temperature=10.0,  # Constant 10°C
        salinity=0.0,  # Fresh water
        ideal=False,  # Use MJD format like FVCOM
    )

    # Load and validate
    generator.load()
    generator.validate()

    # Now we need to set non-zero flux only at active nodes
    # Create flux array with proper shape
    nt = len(generator.timeline)
    node = generator.mesh_ds.sizes["node"]

    flux_array = np.zeros((nt, node), dtype=np.float64)
    for node_idx in active_nodes:
        if node_idx <= node:  # Check if node is in range
            flux_array[:, node_idx - 1] = 1e-6  # Convert to 0-based and set flux

    generator.flux_data = flux_array

    # Generate the file
    content = generator.render()

    # Write to file
    output_file = "groundwater_test_integrated.nc"
    with open(output_file, "wb") as f:
        f.write(content)

    print(f"Created {output_file} using integrated generator")
    print(f"Active nodes (1-based): {active_nodes}")
    print(
        f"Flux velocity: {1e-6:.3e} m/s at {len([n for n in active_nodes if n <= node])} nodes"
    )

    # Test 2: With dye concentration using the standalone script
    print("\n=== Test 2: Standalone Script with Dye ===")

    # Check if standalone script exists
    standalone_script = Path("create_groundwater_netcdf.py")
    if standalone_script.exists():
        # Import and use the standalone script
        from create_groundwater_netcdf import create_groundwater_netcdf

        create_groundwater_netcdf(
            grid_file=grid_file,
            output_file="groundwater_test_with_dye.nc",
            start_datetime="2024-01-01 00:00:00",
            end_datetime="2024-01-03 00:00:00",
            time_interval_seconds=3600,
            active_nodes=active_nodes,
            flux_value=1e-6,  # m/s (velocity)
            temperature_value=10.0,  # °C
            salinity_value=0.0,  # PSU (fresh water)
            dye_value=100.0,  # concentration units
            coordinate_system="Cartesian",
        )
    else:
        print("Standalone script not found. Skipping dye test.")
        print("Note: The integrated generator doesn't support dye yet.")

    # Test 3: Time-varying example
    print("\n=== Test 3: Time-Varying Flux ===")

    # Create time-varying flux (e.g., tidal variation)
    flux_varying = np.zeros((nt, node), dtype=np.float64)
    time_hours = np.arange(nt)

    for i, node_idx in enumerate(active_nodes):
        if node_idx <= node:
            # Sinusoidal variation with different phase for each node
            phase = i * np.pi / 4  # Different phase for each node
            flux_varying[:, node_idx - 1] = 1e-6 * (
                1 + 0.5 * np.sin(2 * np.pi * time_hours / 24 + phase)
            )

    generator.flux_data = flux_varying

    # Generate the file
    content = generator.render()

    # Write to file
    output_file = "groundwater_test_timevarying.nc"
    with open(output_file, "wb") as f:
        f.write(content)

    print(f"Created {output_file} with time-varying flux")

    # Verify the files
    print("\n=== Verification ===")
    try:
        import netCDF4 as nc

        for fname in [
            "groundwater_test_integrated.nc",
            "groundwater_test_with_dye.nc",
            "groundwater_test_timevarying.nc",
        ]:
            if Path(fname).exists():
                print(f"\nChecking {fname}:")
                with nc.Dataset(fname, "r") as ds:
                    print(
                        f"  Dimensions: node={ds.dimensions['node'].size}, time={ds.dimensions['time'].size}"
                    )
                    print(f"  Variables: {list(ds.variables.keys())}")

                    # Check flux at active nodes
                    flux = ds.variables["groundwater_flux"][:]
                    # Note: flux dimensions are (time, node)
                    for node_idx in active_nodes[:3]:  # Check first 3 nodes
                        if node_idx <= ds.dimensions["node"].size:
                            flux_at_node = flux[
                                0, node_idx - 1
                            ]  # First time step, correct node
                            print(
                                f"  Flux velocity at node {node_idx}: {flux_at_node:.3e} m/s"
                            )
    except ImportError:
        print("netCDF4 not available for verification")

    # Clean up temporary grid file
    if grid_file.exists():
        grid_file.unlink()
        print(f"\nCleaned up temporary grid file: {grid_file}")


if __name__ == "__main__":
    test_groundwater_generator()
