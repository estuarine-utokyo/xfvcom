#!/usr/bin/env python3
"""
Add dye concentration to an existing groundwater NetCDF file.
"""

import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np


def add_dye_to_groundwater(
    nc_file: str,
    dye_value: float = 100.0,
    active_nodes: list[int] = None,
    dye_csv: str = None,
):
    """
    Add dye variable to existing groundwater NetCDF file.

    Parameters
    ----------
    nc_file : str
        Path to existing groundwater NetCDF file
    dye_value : float
        Constant dye concentration for active nodes
    active_nodes : list[int]
        List of nodes with dye (1-based). If None, read from file attribute.
    dye_csv : str
        Optional CSV file with time-varying dye data
    """
    # Open file in append mode
    with nc.Dataset(nc_file, "r+") as ds:
        # Get dimensions
        node_dim = ds.dimensions["node"].size
        time_dim = ds.dimensions["time"].size

        # Get active nodes from file or parameter
        if active_nodes is None:
            if hasattr(ds, "active_nodes"):
                active_nodes = [int(n) for n in ds.active_nodes.split(",")]
            else:
                # If not specified, use nodes with non-zero flux
                flux = ds.variables["groundwater_flux"][:]
                active_nodes = []
                for i in range(node_dim):
                    if np.any(flux[i, :] > 0):
                        active_nodes.append(i + 1)  # Convert to 1-based

        print(f"Adding dye to nodes: {active_nodes}")

        # Create dye variable if it doesn't exist
        if "groundwater_dye" not in ds.variables:
            dye_var = ds.createVariable(
                "groundwater_dye", "f4", ("node", "time"), zlib=True, complevel=4
            )
            dye_var.long_name = "groundwater dye concentration"
            dye_var.units = "tracer units"
            dye_var.grid = "fvcom_grid"
            dye_var.type = "data"
        else:
            dye_var = ds.variables["groundwater_dye"]

        # Initialize with zeros
        dye_data: np.ndarray = np.zeros((node_dim, time_dim), dtype=np.float32)

        if dye_csv:
            # Read time-varying dye from CSV
            import pandas as pd

            df = pd.read_csv(dye_csv, parse_dates=["datetime"])

            # Check format
            if "node_id" in df.columns:
                # Long format
                for _, row in df.iterrows():
                    node_idx = int(row["node_id"]) - 1  # Convert to 0-based
                    time_idx = 0  # Would need to match datetime
                    dye_data[node_idx, time_idx] = row["dye"]
            else:
                # Wide format - columns are node IDs
                for node_id in active_nodes:
                    if str(node_id) in df.columns:
                        dye_data[node_id - 1, :] = df[str(node_id)].values
        else:
            # Constant dye value for active nodes
            for node_id in active_nodes:
                if 1 <= node_id <= node_dim:
                    dye_data[node_id - 1, :] = dye_value

        # Write data
        dye_var[:] = dye_data

        # Update global attribute
        if hasattr(ds, "variables_list"):
            var_list = ds.variables_list
            if "groundwater_dye" not in var_list:
                ds.variables_list = var_list + " groundwater_dye"
        else:
            # Add a note about the dye variable
            ds.setncattr("groundwater_dye_added", "true")

        print(f"Dye variable added to {nc_file}")


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Add dye to groundwater NetCDF")
    parser.add_argument("nc_file", help="Groundwater NetCDF file")
    parser.add_argument(
        "--dye-value",
        type=float,
        default=100.0,
        help="Constant dye concentration (default: 100.0)",
    )
    parser.add_argument(
        "--active-nodes", nargs="+", type=int, help="Nodes with dye (1-based)"
    )
    parser.add_argument("--dye-csv", help="CSV file with time-varying dye data")

    args = parser.parse_args()

    add_dye_to_groundwater(
        args.nc_file,
        dye_value=args.dye_value,
        active_nodes=args.active_nodes,
        dye_csv=args.dye_csv,
    )


if __name__ == "__main__":
    main()
