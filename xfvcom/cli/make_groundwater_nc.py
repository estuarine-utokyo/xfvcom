#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI tool to generate groundwater forcing NetCDF files for FVCOM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from ..io.groundwater_nc_generator import GroundwaterNetCDFGenerator


def parse_forcing_value(
    value_str: str,
    node_count: int | None = None,
    time_count: int | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    dt_seconds: int | None = None,
) -> float | np.ndarray:
    """
    Parse forcing value from command line string.

    Supports:
    - Single float: "0.5"
    - CSV file with single column: "flux.csv"
    - Wide format time series: "flux_timeseries.csv:datetime"
    - Long format time series: "flux_long.csv:datetime,node_id,flux"

    Parameters
    ----------
    value_str : str
        Command line value string
    node_count : int, optional
        Number of nodes (for validation)
    time_count : int, optional
        Number of time steps (for time series)
    start_time : str, optional
        Start time for time series interpolation
    end_time : str, optional
        End time for time series interpolation
    dt_seconds : int, optional
        Time step in seconds

    Returns
    -------
    float or np.ndarray
        Parsed forcing data
    """
    # Check if it's a file path
    if ":" in value_str or Path(value_str).exists():
        if ":" in value_str:
            parts = value_str.split(":")
            file_path = parts[0]

            if len(parts) == 2:
                # Wide format time series: "file.csv:datetime"
                time_col = parts[1]
                df = pd.read_csv(file_path)
                if time_col not in df.columns:
                    raise ValueError(
                        f"Time column '{time_col}' not found in {file_path}"
                    )

                # If time parameters are provided, return full time series
                if all(
                    v is not None
                    for v in [time_count, start_time, end_time, dt_seconds]
                ):
                    df[time_col] = pd.to_datetime(df[time_col])

                    # Get expected times
                    expected_times = pd.date_range(
                        start_time, end_time, freq=f"{dt_seconds}s", inclusive="both"
                    )

                    # Initialize array
                    data_array = np.zeros((node_count or 1, time_count))

                    # Get node columns (all except time column)
                    node_cols = [col for col in df.columns if col != time_col]

                    # Map node columns to indices
                    for col in node_cols:
                        if col.startswith("node_"):
                            node_idx = int(col.split("_")[1])
                            if node_count is None or node_idx < node_count:
                                # Interpolate to expected times if needed
                                data_array[node_idx, :] = np.interp(
                                    expected_times.astype(np.int64)
                                    / 1e9,  # Convert to seconds
                                    df[time_col].astype(np.int64) / 1e9,
                                    df[col].values,
                                )

                    return data_array
                else:
                    # Legacy behavior: return mean values
                    data_cols = [col for col in df.columns if col != time_col]
                    if len(data_cols) == 1:
                        return float(df[data_cols[0]].mean())
                    else:
                        return cast(
                            np.ndarray, df[data_cols].mean().to_numpy(dtype=np.float64)
                        )

            elif len(parts) == 2 and "," in parts[1]:
                # Long format: "file.csv:datetime,node_id,value"
                col_spec = parts[1].split(",")
                if len(col_spec) == 3:
                    time_col, node_col, value_col = col_spec
                else:
                    raise ValueError(
                        "Long format requires 3 columns: datetime,node_id,value"
                    )
                df = pd.read_csv(file_path)

                if not all(
                    col in df.columns for col in [time_col, node_col, value_col]
                ):
                    raise ValueError(f"Required columns not found in {file_path}")

                df[time_col] = pd.to_datetime(df[time_col])

                # If time parameters are provided, return full time series
                if all(
                    v is not None
                    for v in [time_count, start_time, end_time, dt_seconds]
                ):
                    # Pivot to wide format
                    pivot_df = df.pivot(
                        index=time_col, columns=node_col, values=value_col
                    )

                    # Get expected times
                    expected_times = pd.date_range(
                        start_time, end_time, freq=f"{dt_seconds}s", inclusive="both"
                    )

                    # Initialize array
                    data_array = np.zeros((node_count or pivot_df.shape[1], time_count))

                    # Fill data for available nodes
                    for col_val in pivot_df.columns:
                        node_idx = int(col_val)
                        if node_count is None or node_idx < node_count:
                            # Interpolate to expected times
                            data_array[node_idx, :] = np.interp(
                                expected_times.astype(np.int64) / 1e9,
                                pivot_df.index.astype(np.int64) / 1e9,
                                pivot_df[col_val].values,
                            )

                    return data_array
                else:
                    # Legacy behavior: return mean values
                    return float(df.groupby(node_col)[value_col].mean().mean())
            else:
                # Unrecognized format
                raise ValueError(f"Unrecognized time series format: {value_str}")
        else:
            # Simple CSV with node values
            df = pd.read_csv(value_str)

            # Check if it has node_id column (sparse format)
            if "node_id" in df.columns:
                # Sparse format: only specified nodes have values
                value_col = [col for col in df.columns if col != "node_id"][0]
                if node_count is not None:
                    # Create array with zeros
                    data: np.ndarray = np.zeros(node_count, dtype=np.float64)
                    # Fill in specified nodes (convert from 1-based to 0-based)
                    for _, row in df.iterrows():
                        node_idx = int(row["node_id"]) - 1
                        if 0 <= node_idx < node_count:
                            data[node_idx] = row[value_col]
                    return data
                else:
                    # Return just the values
                    return cast(np.ndarray, df[value_col].to_numpy(dtype=np.float64))
            else:
                # Dense format
                df_values = pd.read_csv(value_str, header=None)
                if df_values.shape[1] == 1:
                    # Single column = values for each node
                    return cast(
                        np.ndarray, df_values.iloc[:, 0].to_numpy(dtype=np.float64)
                    )
                else:
                    # Multiple columns = assume each column is a node
                    return cast(np.ndarray, df_values.to_numpy(dtype=np.float64).T)
    else:
        # Single constant value
        return float(value_str)


def generate_groundwater_namelist(
    nc_file: Path,
    has_dye: bool,
    constant_flux: float | None = None,
    constant_temp: float | None = None,
    constant_salt: float | None = None,
    constant_dye: float | None = None,
) -> str:
    """
    Generate FVCOM namelist snippet for groundwater module.

    Parameters
    ----------
    nc_file : Path
        Path to the generated NetCDF file
    has_dye : bool
        Whether dye data is included
    constant_flux : float, optional
        Constant flux value if uniform
    constant_temp : float, optional
        Constant temperature value if uniform
    constant_salt : float, optional
        Constant salinity value if uniform
    constant_dye : float, optional
        Constant dye value if uniform

    Returns
    -------
    str
        Namelist content
    """
    # Determine groundwater kind
    if all(v is not None for v in [constant_flux, constant_temp, constant_salt]):
        # Check if truly constant (non-zero flux only at specific nodes still counts as variable)
        if isinstance(constant_flux, (int, float)) and constant_flux > 0:
            groundwater_kind = "constant"
        else:
            groundwater_kind = "variable"
    else:
        groundwater_kind = "variable"

    # Format values for Fortran
    def format_fortran_real(value):
        if value is None:
            return "0.0"
        return f"{float(value):.6E}"

    # Build namelist
    nml = "&NML_GROUNDWATER\n"
    nml += " GROUNDWATER_ON       = T,\n"
    nml += " GROUNDWATER_TEMP_ON  = T,\n"
    nml += " GROUNDWATER_SALT_ON  = T,\n"

    if has_dye:
        nml += " GROUNDWATER_DYE_ON   = T,\n"
    else:
        nml += " GROUNDWATER_DYE_ON   = F,\n"

    nml += f" GROUNDWATER_KIND     = '{groundwater_kind}',\n"
    nml += f" GROUNDWATER_FILE     = '{nc_file.name}',\n"

    # Default values (used as fallback)
    nml += f" GROUNDWATER_FLOW     = {format_fortran_real(constant_flux if constant_flux else 0.0)},\n"
    nml += f" GROUNDWATER_TEMP     = {format_fortran_real(constant_temp if constant_temp else 0.0)},\n"
    nml += f" GROUNDWATER_SALT     = {format_fortran_real(constant_salt if constant_salt else 0.0)}"

    if has_dye and constant_dye is not None:
        nml += f",\n GROUNDWATER_DYE      = {format_fortran_real(constant_dye)}"

    nml += "\n/\n"

    # Add dye release namelist if needed
    if has_dye:
        nml += "\n! Also ensure dye module is enabled:\n"
        nml += "&NML_DYE_RELEASE\n"
        nml += " DYE_ON = T,\n"
        nml += " DYE_RELEASE_START = 'model start time',  ! Update this\n"
        nml += " DYE_RELEASE_STOP  = 'model end time',    ! Update this\n"
        nml += " KSPE_DYE = 0,  ! No point sources (using groundwater)\n"
        nml += " MSPE_DYE = 0   ! No point sources (using groundwater)\n"
        nml += "/\n"

    return nml


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

  # Read node-specific values from CSV files (constant in time)
  %(prog)s grid.dat --utm-zone 54 \\
    --start 2025-01-01T00:00Z --end 2025-12-31T23:00Z \\
    --flux flux_by_node.csv --temperature temp_by_node.csv

  # Wide format time series CSV (columns: datetime, node_100, node_200, ...)
  %(prog)s grid.nc --start 2025-01-01T00:00Z --end 2025-12-31T23:00Z \\
    --flux flux_timeseries.csv:datetime \\
    --temperature temp_timeseries.csv:datetime \\
    --salinity 0.0

  # Long format time series CSV (columns: datetime, node_id, flux/temperature)
  %(prog)s grid.nc --start 2025-01-01T00:00Z --end 2025-12-31T23:00Z \\
    --flux groundwater_long.csv:datetime,node_id,flux \\
    --temperature groundwater_long.csv:datetime,node_id,temperature \\
    --salinity 0.0

  # Mix constant and time series values
  %(prog)s grid.nc --start 2025-01-01 --end 2025-01-07 \\
    --start-tz Asia/Tokyo --dt 3600 \\
    --flux flux_timeseries.csv:datetime --temperature 15.0 --salinity 0.0
    
  # Include dye concentration
  %(prog)s grid.dat --utm-zone 54 --start 2025-01-01T00:00Z --end 2025-01-10T00:00Z \\
    --flux 1.0 --temperature 20.0 --salinity 30.0 --dye 10.0
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
        help="Groundwater flux velocity (m/s): constant, CSV file, or time series CSV (e.g., flux.csv:datetime)",
    )
    parser.add_argument(
        "--temperature",
        default="0.0",
        help="Groundwater temperature (°C): constant, CSV file, or time series CSV",
    )
    parser.add_argument(
        "--salinity",
        default="0.0",
        help="Groundwater salinity (PSU): constant, CSV file, or time series CSV",
    )
    parser.add_argument(
        "--dye",
        default=None,
        help="Groundwater dye concentration (tracer units): constant, CSV file, or time series CSV",
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
    parser.add_argument(
        "--nml",
        "--namelist",
        action="store_true",
        help="Also generate FVCOM namelist snippet (.nml file)",
    )

    args = parser.parse_args(argv)

    # Validate grid file
    if not args.grid_file.exists():
        parser.error(f"Grid file not found: {args.grid_file}")

    if args.grid_file.suffix.lower() == ".dat" and args.utm_zone is None:
        parser.error("--utm-zone is required for .dat grid files")

    # Calculate time parameters for parsing CSV time series
    try:
        # Get time information
        from datetime import datetime

        import pytz  # type: ignore

        # Parse start/end times
        if args.start.endswith("Z"):
            start_dt = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
        else:
            tz = pytz.timezone(args.start_tz)
            start_dt = tz.localize(datetime.fromisoformat(args.start))

        if args.end.endswith("Z"):
            end_dt = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
        else:
            tz = pytz.timezone(args.start_tz)
            end_dt = tz.localize(datetime.fromisoformat(args.end))

        # Calculate time count
        time_delta = end_dt - start_dt
        time_count = int(time_delta.total_seconds() / args.dt) + 1

        # Get node count if grid file exists
        node_count = None
        if args.grid_file.exists():
            if args.grid_file.suffix.lower() == ".dat":
                from ..grid import FvcomGrid

                grid = FvcomGrid.from_dat(args.grid_file, utm_zone=args.utm_zone)
                node_count = len(grid.x)
            else:
                import xarray as xr

                with xr.open_dataset(args.grid_file) as ds:
                    node_count = ds.sizes.get("node", ds.sizes.get("nMesh2_node"))

        # Parse forcing values with time parameters
        flux = parse_forcing_value(
            args.flux,
            node_count=node_count,
            time_count=time_count,
            start_time=args.start,
            end_time=args.end,
            dt_seconds=args.dt,
        )
        temperature = parse_forcing_value(
            args.temperature,
            node_count=node_count,
            time_count=time_count,
            start_time=args.start,
            end_time=args.end,
            dt_seconds=args.dt,
        )
        salinity = parse_forcing_value(
            args.salinity,
            node_count=node_count,
            time_count=time_count,
            start_time=args.start,
            end_time=args.end,
            dt_seconds=args.dt,
        )
        dye = None
        if args.dye is not None:
            dye = parse_forcing_value(
                args.dye,
                node_count=node_count,
                time_count=time_count,
                start_time=args.start,
                end_time=args.end,
                dt_seconds=args.dt,
            )
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
            dye=dye,
            ideal=args.ideal,
        )

        # Generate output
        output_path = args.output or Path("groundwater_forcing.nc")
        generator.write(output_path)

        print(f"Groundwater forcing file written to: {output_path}")

        # Generate namelist if requested
        if args.nml:
            # Determine if values are constant
            const_flux = flux if isinstance(flux, (int, float)) else None
            const_temp = temperature if isinstance(temperature, (int, float)) else None
            const_salt = salinity if isinstance(salinity, (int, float)) else None
            const_dye = (
                dye if dye is not None and isinstance(dye, (int, float)) else None
            )

            # Generate namelist
            nml_content = generate_groundwater_namelist(
                nc_file=output_path,
                has_dye=(dye is not None),
                constant_flux=const_flux,
                constant_temp=const_temp,
                constant_salt=const_salt,
                constant_dye=const_dye,
            )

            # Write namelist file
            nml_path = output_path.with_suffix(".nml")
            nml_path.write_text(nml_content)
            print(f"Groundwater namelist written to: {nml_path}")
            print("\nNamelist content:")
            print(nml_content)

    except Exception as e:
        parser.error(f"Error generating groundwater forcing: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
