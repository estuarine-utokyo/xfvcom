#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI tool to generate groundwater forcing NetCDF files for FVCOM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ..io.groundwater_nc_generator import GroundwaterNetCDFGenerator


def parse_forcing_value(
    value_str: str, node_count: int | None = None
) -> float | np.ndarray:
    """
    Parse forcing value from command line string.

    Supports:
    - Single float: "0.5"
    - CSV file with single column: "flux.csv"
    - CSV file with time series: "flux_timeseries.csv:time"

    Parameters
    ----------
    value_str : str
        Command line value string
    node_count : int, optional
        Number of nodes (for validation)

    Returns
    -------
    float or np.ndarray
        Parsed forcing data
    """
    # Check if it's a file path
    if ":" in value_str or Path(value_str).exists():
        if ":" in value_str:
            file_path, time_col = value_str.split(":", 1)
            # Time series CSV
            df = pd.read_csv(file_path)
            if time_col not in df.columns:
                raise ValueError(f"Time column '{time_col}' not found in {file_path}")
            # For now, return mean values - full time series support would need enhancement
            data_cols = [col for col in df.columns if col != time_col]
            if len(data_cols) == 1:
                return float(df[data_cols[0]].mean())
            else:
                # Multiple nodes
                return cast(np.ndarray, df[data_cols].mean().to_numpy(dtype=np.float64))
        else:
            # Simple CSV with node values
            df = pd.read_csv(value_str, header=None)
            if df.shape[1] == 1:
                # Single column = values for each node
                return cast(np.ndarray, df.iloc[:, 0].to_numpy(dtype=np.float64))
            else:
                # Multiple columns = assume each column is a node
                return cast(np.ndarray, df.to_numpy(dtype=np.float64).T)
    else:
        # Single constant value
        return float(value_str)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate groundwater forcing NetCDF file for FVCOM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Constant values for all nodes and times
  %(prog)s grid.nc --start 2025-01-01T00:00Z --end 2025-01-02T00:00Z \\
    --flux 0.001 --temperature 15.0 --salinity 0.0

  # Read node-specific values from CSV files
  %(prog)s grid.dat --utm-zone 54 \\
    --start 2025-01-01T00:00Z --end 2025-12-31T23:00Z \\
    --flux flux_by_node.csv --temperature temp_by_node.csv

  # Mix constant and CSV values
  %(prog)s grid.nc --start 2025-01-01 --end 2025-01-07 \\
    --start-tz Asia/Tokyo --dt 3600 \\
    --flux groundwater_flux.csv --temperature 10.0 --salinity 0.0
""",
    )

    # Grid file
    parser.add_argument(
        "grid_file",
        type=Path,
        help="Grid file (.nc or .dat format)",
    )

    # Time parameters
    parser.add_argument(
        "--start",
        required=True,
        help="Start time (ISO-8601 format, e.g., 2025-01-01T00:00Z)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End time (ISO-8601 format, e.g., 2025-01-02T00:00Z)",
    )
    parser.add_argument(
        "--dt",
        type=int,
        default=3600,
        help="Time step in seconds (default: 3600)",
    )
    parser.add_argument(
        "--start-tz",
        default="UTC",
        help="Timezone for start/end if not specified (default: UTC)",
    )

    # Grid parameters (for .dat files)
    parser.add_argument(
        "--utm-zone",
        type=int,
        help="UTM zone number (required for .dat files)",
    )
    parser.add_argument(
        "--southern",
        action="store_true",
        help="Grid is in southern hemisphere (for UTM conversion)",
    )

    # Forcing parameters
    parser.add_argument(
        "--flux",
        default="0.0",
        help="Groundwater flux (m³/s): constant value or CSV file",
    )
    parser.add_argument(
        "--temperature",
        default="0.0",
        help="Groundwater temperature (°C): constant value or CSV file",
    )
    parser.add_argument(
        "--salinity",
        default="0.0",
        help="Groundwater salinity (PSU): constant value or CSV file",
    )

    # Time format
    parser.add_argument(
        "--ideal",
        action="store_true",
        help="Use ideal time format (days since 0.0) instead of MJD",
    )

    # Output
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output NetCDF file (default: groundwater_forcing.nc)",
    )

    args = parser.parse_args(argv)

    # Validate grid file
    if not args.grid_file.exists():
        parser.error(f"Grid file not found: {args.grid_file}")

    if args.grid_file.suffix.lower() == ".dat" and args.utm_zone is None:
        parser.error("--utm-zone is required for .dat grid files")

    # Parse forcing values
    try:
        flux = parse_forcing_value(args.flux)
        temperature = parse_forcing_value(args.temperature)
        salinity = parse_forcing_value(args.salinity)
    except Exception as e:
        parser.error(f"Error parsing forcing values: {e}")

    # Create generator
    try:
        generator = GroundwaterNetCDFGenerator(
            grid_nc=args.grid_file,
            start=args.start,
            end=args.end,
            dt_seconds=args.dt,
            utm_zone=args.utm_zone,
            northern=not args.southern,
            start_tz=args.start_tz,
            flux=flux,
            temperature=temperature,
            salinity=salinity,
            ideal=args.ideal,
        )

        # Generate output
        output_path = args.output or Path("groundwater_forcing.nc")
        generator.write(output_path)

        print(f"Groundwater forcing file written to: {output_path}")

    except Exception as e:
        parser.error(f"Error generating groundwater forcing: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
