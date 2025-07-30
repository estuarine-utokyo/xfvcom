#!/usr/bin/env python3
"""
Create FVCOM Groundwater NetCDF file using low-level netCDF4 package.

This script creates a properly formatted NetCDF file for FVCOM groundwater forcing,
containing time series of flux, temperature, salinity, and optionally dye concentration
at all nodes.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Union

import netCDF4 as nc
import numpy as np

# Add parent directory to use xfvcom
sys.path.insert(0, str(Path(__file__).parent.parent))

from xfvcom.grid import FvcomGrid


def create_groundwater_netcdf(
    grid_file: Union[str, Path],
    output_file: Union[str, Path],
    start_datetime: str,
    end_datetime: str,
    time_interval_seconds: int,
    active_nodes: List[int],  # 1-based Fortran indices
    flux_value: float = 1e-5,  # m/s (velocity, not volumetric flux!)
    temperature_value: float = 10.0,  # degC
    salinity_value: float = 0.0,  # PSU
    dye_value: Optional[float] = None,  # concentration units
    coordinate_system: str = "Cartesian",  # or "Geographic"
) -> None:
    """
    Create a groundwater forcing NetCDF file for FVCOM.

    Parameters
    ----------
    grid_file : str or Path
        Path to FVCOM grid file (.dat or .nc) to get node coordinates
    output_file : str or Path
        Path for output NetCDF file
    start_datetime : str
        Start datetime string, e.g., "2020-01-01 00:00:00"
    end_datetime : str
        End datetime string, e.g., "2020-01-10 00:00:00"
    time_interval_seconds : int
        Time interval in seconds between data points
    active_nodes : List[int]
        List of node indices (1-based, Fortran style) with non-zero flux
    flux_value : float
        Groundwater flux velocity for active nodes (m/s) - NOT volumetric flux!
    temperature_value : float
        Groundwater temperature (°C)
    salinity_value : float
        Groundwater salinity (PSU)
    dye_value : float, optional
        Groundwater dye concentration (if None, dye variable not created)
    coordinate_system : str
        Either "Cartesian" or "Geographic"
    """

    # Convert paths
    grid_file = Path(grid_file)
    output_file = Path(output_file)

    # Load grid information using xfvcom
    print(f"Loading grid from {grid_file}")
    if grid_file.suffix == ".dat":
        # Read ASCII grid file using xfvcom
        try:
            # Try with UTM zone 33 (common for European grids)
            grid = FvcomGrid.from_dat(grid_file, utm_zone=33, northern=True)
            grid_ds = grid.to_xarray()
        except Exception as e:
            print(f"Error reading grid file: {e}")
            print("Creating a simple test grid instead...")
            # Create dummy grid for testing
            grid_file = create_dummy_grid()
            grid = FvcomGrid.from_dat(grid_file, utm_zone=33, northern=True)
            grid_ds = grid.to_xarray()

        node_count = len(grid_ds.node)
        x_coords = grid_ds.x.values
        y_coords = grid_ds.y.values

        # Check if geographic coordinates exist
        if "lon" in grid_ds:
            lon_coords = grid_ds.lon.values
            lat_coords = grid_ds.lat.values
        else:
            # Dummy conversion for demonstration
            lon_coords = x_coords * 1e-5
            lat_coords = y_coords * 1e-5
    else:
        # Read NetCDF grid file
        import xarray as xr

        grid_ds = xr.open_dataset(grid_file)
        node_count = len(grid_ds.node)

        if "x" in grid_ds:
            x_coords = grid_ds.x.values
            y_coords = grid_ds.y.values
        else:
            x_coords = np.zeros(node_count)
            y_coords = np.zeros(node_count)

        if "lon" in grid_ds:
            lon_coords = grid_ds.lon.values
            lat_coords = grid_ds.lat.values
        else:
            # If no geographic coords, create dummy ones
            lon_coords = x_coords * 1e-5
            lat_coords = y_coords * 1e-5

    # Create time array
    start_dt = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_datetime, "%Y-%m-%d %H:%M:%S")

    time_points = []
    current_dt = start_dt
    while current_dt <= end_dt:
        time_points.append(current_dt)
        current_dt += timedelta(seconds=time_interval_seconds)

    n_times = len(time_points)
    print(f"Creating {n_times} time steps from {start_datetime} to {end_datetime}")

    # Convert active nodes from 1-based to 0-based
    active_nodes_0based = [n - 1 for n in active_nodes]

    # Prepare data arrays
    # Initialize all fluxes to zero
    flux_array: np.ndarray = np.zeros((n_times, node_count), dtype=np.float32)
    temp_array: np.ndarray = np.full(
        (n_times, node_count), temperature_value, dtype=np.float32
    )
    salt_array: np.ndarray = np.full(
        (n_times, node_count), salinity_value, dtype=np.float32
    )

    # Set non-zero flux only at active nodes
    for node_idx in active_nodes_0based:
        if 0 <= node_idx < node_count:
            flux_array[:, node_idx] = flux_value
        else:
            print(f"Warning: Node {node_idx + 1} is out of range (max {node_count})")

    if dye_value is not None:
        dye_array: np.ndarray = np.zeros((n_times, node_count), dtype=np.float32)
        # Set dye concentration at active nodes
        for node_idx in active_nodes_0based:
            if 0 <= node_idx < node_count:
                dye_array[:, node_idx] = dye_value

    # Calculate time variables for FVCOM
    # Modified Julian Day reference: November 17, 1858
    mjd_ref = datetime(1858, 11, 17)

    time_mjd = []
    itime = []
    itime2 = []
    times_str = []

    for dt in time_points:
        # Calculate days since MJD reference
        delta = dt - mjd_ref
        days = delta.total_seconds() / 86400.0
        time_mjd.append(days)

        # Itime: integer days
        itime.append(int(days))

        # Itime2: milliseconds since midnight of that day
        midnight = datetime(dt.year, dt.month, dt.day)
        msec_since_midnight = int((dt - midnight).total_seconds() * 1000)
        itime2.append(msec_since_midnight)

        # Times string: "YYYY-MM-DD HH:MM:SS.000000"
        times_str.append(dt.strftime("%Y-%m-%d %H:%M:%S.000000"))

    # Create NetCDF file
    print(f"Writing NetCDF file: {output_file}")
    with nc.Dataset(output_file, "w", format="NETCDF4_CLASSIC") as ds:

        # Create dimensions
        node_dim = ds.createDimension("node", node_count)
        time_dim = ds.createDimension("time", None)  # unlimited
        datestrlen_dim = ds.createDimension("DateStrLen", 26)

        # Create coordinate variables
        if coordinate_system == "Geographic":
            # Geographic coordinates
            lon_var = ds.createVariable("lon", "f4", ("node",))
            lon_var.long_name = "nodal longitude"
            lon_var.units = "degrees_east"
            lon_var[:] = lon_coords

            lat_var = ds.createVariable("lat", "f4", ("node",))
            lat_var.long_name = "nodal latitude"
            lat_var.units = "degrees_north"
            lat_var[:] = lat_coords
        else:
            # Cartesian coordinates
            x_var = ds.createVariable("x", "f4", ("node",))
            x_var.long_name = "nodal x"
            x_var.units = "meter"
            x_var[:] = x_coords

            y_var = ds.createVariable("y", "f4", ("node",))
            y_var.long_name = "nodal y"
            y_var.units = "meter"
            y_var[:] = y_coords

        # Create time variables
        time_var = ds.createVariable("time", "f4", ("time",))
        time_var.long_name = "time"
        time_var.units = "days since 1858-11-17 00:00:00"
        time_var.format = "modified julian day (MJD)"
        time_var.time_zone = "UTC"
        time_var[:] = time_mjd

        itime_var = ds.createVariable("Itime", "i4", ("time",))
        itime_var.units = "days since 1858-11-17 00:00:00"
        itime_var.format = "modified julian day (MJD)"
        itime_var.time_zone = "UTC"
        itime_var[:] = itime

        itime2_var = ds.createVariable("Itime2", "i4", ("time",))
        itime2_var.units = "msec since 00:00:00"
        itime2_var.time_zone = "UTC"
        itime2_var[:] = itime2

        times_var = ds.createVariable("Times", "S1", ("time", "DateStrLen"))
        times_var.time_zone = "UTC"
        # Convert strings to character array
        times_char: np.ndarray = np.zeros((n_times, 26), dtype="S1")
        for i, time_str in enumerate(times_str):
            for j, char in enumerate(time_str.ljust(26)):
                times_char[i, j] = char.encode("ascii")
        times_var[:] = times_char

        # Create groundwater forcing variables
        flux_var = ds.createVariable("groundwater_flux", "f4", ("time", "node"))
        flux_var.long_name = "Ground Water Flux"
        flux_var.units = "m s-1"
        flux_var[:] = flux_array

        temp_var = ds.createVariable("groundwater_temp", "f4", ("time", "node"))
        temp_var.long_name = "Ground Water Temperature"
        temp_var.units = "degree C"
        temp_var[:] = temp_array

        salt_var = ds.createVariable("groundwater_salt", "f4", ("time", "node"))
        salt_var.long_name = "Ground Water Salinity"
        salt_var.units = "psu"
        salt_var[:] = salt_array

        # Create dye variable if requested
        if dye_value is not None:
            dye_var = ds.createVariable("groundwater_dye", "f4", ("time", "node"))
            dye_var.long_name = "Ground Water Dye Concentration"
            dye_var.units = "concentration units"
            dye_var[:] = dye_array

        # Global attributes
        ds.source = "fvcom grid (unstructured) surface forcing"
        ds.history = f"Created by create_groundwater_netcdf.py on {datetime.now()}"

        # Summary information
        ds.active_nodes = ",".join(map(str, active_nodes))
        ds.flux_value = flux_value
        ds.temperature_value = temperature_value
        ds.salinity_value = salinity_value
        if dye_value is not None:
            ds.dye_value = dye_value

    print(f"NetCDF file created successfully!")
    print(f"Active nodes (1-based): {active_nodes}")
    print(f"Flux velocity: {flux_value:.3e} m/s at {len(active_nodes)} nodes")


def example_usage():
    """Example usage of the groundwater NetCDF creator."""

    # Example 1: Basic usage with channel grid
    grid_file = Path("../../FVCOM/Tests/GroundwaterDye/input/chn_grd.dat")

    if not grid_file.exists():
        print(f"Grid file not found: {grid_file}")
        # print("Using dummy grid for demonstration...")
        # Create a minimal dummy grid file for testing
        # grid_file = create_dummy_grid()

    # Create groundwater forcing with specified nodes
    create_groundwater_netcdf(
        grid_file=grid_file,
        output_file="groundwater_test.nc",
        start_datetime="2020-01-01 00:00:00",
        end_datetime="2020-01-10 00:00:00",
        time_interval_seconds=3600,  # hourly
        active_nodes=[637, 638, 639, 662],  # 1-based node indices
        flux_value=1e-5,  # m/s (velocity)
        temperature_value=10.0,  # degC
        salinity_value=0.0,  # PSU (fresh water)
        dye_value=100.0,  # concentration
        coordinate_system="Cartesian",
    )

    # Example 2: Without dye (only if using real grid)
    if grid_file.exists() and "dummy" not in str(grid_file):
        create_groundwater_netcdf(
            grid_file=grid_file,
            output_file="groundwater_no_dye.nc",
            start_datetime="2020-01-01 00:00:00",
            end_datetime="2020-01-05 00:00:00",
            time_interval_seconds=1800,  # 30 minutes
            active_nodes=[100, 200, 300],  # Different nodes
            flux_value=5e-6,
            temperature_value=15.0,
            salinity_value=5.0,
            dye_value=None,  # No dye
            coordinate_system="Cartesian",
        )


def create_dummy_grid() -> Path:
    """Create a minimal dummy grid for testing."""
    # Use NetCDF format for simplicity
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        dummy_file = Path(tmp.name)

    # Create a simple rectangular grid with at least 700 nodes
    nx, ny = 25, 28  # 25x28 = 700 nodes

    # Create node coordinates
    x: np.ndarray = np.repeat(np.arange(nx) * 1000.0, ny)
    y: np.ndarray = np.tile(np.arange(ny) * 1000.0, nx)

    # Simple triangulation
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
    with nc.Dataset(dummy_file, "w") as ds:
        # Dimensions
        ds.createDimension("node", len(x))
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

    return dummy_file


if __name__ == "__main__":
    example_usage()
