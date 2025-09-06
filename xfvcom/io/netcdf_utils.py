"""
NetCDF utilities for reading and writing FVCOM-format files.

This module provides functions for working with FVCOM NetCDF files,
particularly for river, meteorological, and groundwater forcing data.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import netCDF4 as nc
import numpy as np
import pandas as pd


def decode_fvcom_time(itime: np.ndarray, itime2: np.ndarray) -> pd.DatetimeIndex:
    """
    Decode FVCOM time format (Modified Julian Day) to pandas datetime.

    FVCOM uses two integer arrays to store time:
    - Itime: days since 1858-11-17 (Modified Julian Day)
    - Itime2: milliseconds since midnight of that day

    Parameters
    ----------
    itime : np.ndarray
        Array of Modified Julian Days
    itime2 : np.ndarray
        Array of milliseconds since midnight

    Returns
    -------
    pd.DatetimeIndex
        Decoded datetime values

    Examples
    --------
    >>> itime = np.array([58849, 58850])  # MJD for 2020-01-01 and 2020-01-02
    >>> itime2 = np.array([0, 43200000])  # Midnight and noon
    >>> times = decode_fvcom_time(itime, itime2)
    """
    mjd_epoch = pd.Timestamp("1858-11-17")
    times = []

    for day, ms in zip(itime, itime2):
        dt = (
            mjd_epoch + pd.Timedelta(days=int(day)) + pd.Timedelta(milliseconds=int(ms))
        )
        times.append(dt)

    return pd.DatetimeIndex(times)


def encode_fvcom_time(
    datetimes: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Encode datetime to FVCOM time format (Modified Julian Day).

    Parameters
    ----------
    datetimes : pd.DatetimeIndex
        Datetime values to encode

    Returns
    -------
    tuple
        (itime, itime2, times_str) where:
        - itime: Modified Julian Days (int32)
        - itime2: Milliseconds since midnight (int32)
        - times_str: Formatted time strings for Times variable
    """
    mjd_epoch = pd.Timestamp("1858-11-17")

    itime = np.zeros(len(datetimes), dtype=np.int32)
    itime2 = np.zeros(len(datetimes), dtype=np.int32)
    times_str = []

    for i, dt in enumerate(datetimes):
        # Calculate days since MJD epoch
        delta = dt - mjd_epoch
        itime[i] = delta.days

        # Calculate milliseconds since midnight
        midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        itime2[i] = int((dt - midnight).total_seconds() * 1000)

        # Format time string (YYYY-MM-DD HH:MM:SS.SSS)
        times_str.append(dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])

    return itime, itime2, times_str


def to_mjd(times: pd.DatetimeIndex) -> np.ndarray:
    """
    Convert datetime to Modified Julian Day (floating point).

    This is used for compatibility with some FVCOM tools that expect
    floating-point MJD values rather than the integer Itime/Itime2 format.

    Parameters
    ----------
    times : pd.DatetimeIndex
        Datetime values to convert

    Returns
    -------
    np.ndarray
        Modified Julian Day values as float64
    """
    mjd_epoch = pd.Timestamp("1858-11-17")
    return ((times - mjd_epoch) / pd.Timedelta("1D")).to_numpy("f8")


def read_fvcom_river_nc(filepath: Union[Path, str]) -> Dict:
    """
    Read FVCOM river NetCDF file with proper decoding.

    Parameters
    ----------
    filepath : Path or str
        Path to river NetCDF file

    Returns
    -------
    dict
        Dictionary containing:
        - 'datetime': DatetimeIndex of time points
        - 'river_flux', 'river_temp', 'river_salt': DataFrames with river data
        - 'river_names': List of river names
        - 'dimensions': Dictionary of dimension sizes
        - 'global_attrs': Dictionary of global attributes
        - '*_attrs': Dictionaries of variable attributes
    """
    filepath = Path(filepath)
    data = {}

    with nc.Dataset(filepath, "r") as ds:
        # Store metadata
        data["global_attrs"] = {attr: ds.getncattr(attr) for attr in ds.ncattrs()}
        data["dimensions"] = {dim: len(ds.dimensions[dim]) for dim in ds.dimensions}

        # Decode time
        if "Itime" in ds.variables and "Itime2" in ds.variables:
            data["datetime"] = decode_fvcom_time(
                ds.variables["Itime"][:], ds.variables["Itime2"][:]
            )
        elif "time" in ds.variables:
            # Try to decode from time variable if Itime/Itime2 not available
            time_var = ds.variables["time"]
            if "units" in time_var.ncattrs():
                # Use pandas to parse time units
                data["datetime"] = pd.to_datetime(
                    time_var[:], unit="D", origin="1858-11-17"
                )
            else:
                raise ValueError("Cannot decode time from NetCDF file")

        # Determine river dimension name
        river_dim = "river" if "river" in ds.dimensions else "rivers"
        data["river_dim"] = river_dim
        n_rivers = data["dimensions"][river_dim]

        # Read river data as pandas DataFrames
        for var_name in ["river_flux", "river_temp", "river_salt"]:
            if var_name in ds.variables:
                var = ds.variables[var_name]
                # Create DataFrame with time index and river columns
                df_data = var[:]
                if df_data.ndim == 2:
                    data[var_name] = pd.DataFrame(
                        df_data,
                        index=data["datetime"],
                        columns=[f"River_{i+1}" for i in range(n_rivers)],
                    )
                else:
                    # Handle 1D case (single river)
                    data[var_name] = pd.DataFrame(
                        df_data.reshape(-1, 1),
                        index=data["datetime"],
                        columns=["River_1"],
                    )
                # Store variable attributes
                data[f"{var_name}_attrs"] = {
                    attr: var.getncattr(attr) for attr in var.ncattrs()
                }

        # Decode river names
        if "river_names" in ds.variables:
            names_raw = ds.variables["river_names"][:]
            data["river_names"] = []

            for i in range(n_rivers):
                if names_raw.ndim == 1:
                    name = names_raw[i]
                else:
                    name = names_raw[i, :]

                # Decode name from bytes/char array
                if isinstance(name, bytes):
                    name_str = name.decode("utf-8").strip()
                elif hasattr(name, "tobytes"):
                    name_str = name.tobytes().decode("utf-8").strip("\x00").strip()
                else:
                    name_str = (
                        "".join(
                            [
                                chr(c) if isinstance(c, (int, np.integer)) else str(c)
                                for c in name.flatten()
                            ]
                        )
                        .strip("\x00")
                        .strip()
                    )

                data["river_names"].append(name_str)

            # Update DataFrame column names with river names
            for var_name in ["river_flux", "river_temp", "river_salt"]:
                if var_name in data and len(data["river_names"]) == n_rivers:
                    data[var_name].columns = data["river_names"]

    return data


def write_fvcom_river_nc(
    filepath: Union[Path, str], data: Dict, format: str = "NETCDF4_CLASSIC"
) -> None:
    """
    Write FVCOM-compatible river NetCDF file.

    Parameters
    ----------
    filepath : Path or str
        Output file path
    data : dict
        River data dictionary from read_fvcom_river_nc or similar structure
    format : str
        NetCDF format ('NETCDF4_CLASSIC', 'NETCDF3_CLASSIC', etc.)
    """
    filepath = Path(filepath)
    river_dim = data.get("river_dim", "rivers")

    with nc.Dataset(filepath, "w", format=format) as ds:
        # Create dimensions
        n_times = len(data["datetime"])
        n_rivers = data["dimensions"][river_dim]

        ds.createDimension("time", n_times)
        ds.createDimension(river_dim, n_rivers)

        # String dimensions
        if "river_names" in data:
            max_name_len = max(len(name) for name in data["river_names"]) + 1
            ds.createDimension("namelen", max_name_len)

        ds.createDimension("DateStrLen", 30)

        # Set global attributes
        for attr, value in data.get("global_attrs", {}).items():
            ds.setncattr(attr, value)

        # Add modification timestamp
        ds.setncattr(
            "history",
            f"Created/modified by xfvcom on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )

        # Encode and write time variables
        itime, itime2, times_str = encode_fvcom_time(data["datetime"])

        # Itime variable
        itime_var = ds.createVariable("Itime", "i4", ("time",))
        itime_var.units = "days since 1858-11-17 00:00:00"
        itime_var.format = "modified julian day (MJD)"
        itime_var.time_zone = "UTC"
        itime_var[:] = itime

        # Itime2 variable
        itime2_var = ds.createVariable("Itime2", "i4", ("time",))
        itime2_var.units = "msec since 00:00:00"
        itime2_var.time_zone = "UTC"
        itime2_var[:] = itime2

        # Times string array
        times_var = ds.createVariable("Times", "c", ("time", "DateStrLen"))
        times_var.time_zone = "UTC"
        for i, ts in enumerate(times_str):
            ts_array = np.zeros(30, dtype="S1")
            ts_bytes = ts.encode("utf-8")
            ts_array[: len(ts_bytes)] = list(ts_bytes)
            times_var[i, :] = ts_array

        # Write river names
        if "river_names" in data:
            names_var = ds.createVariable("river_names", "c", (river_dim, "namelen"))
            for i, name in enumerate(data["river_names"]):
                name_array = np.zeros(max_name_len, dtype="S1")
                name_bytes = name.encode("utf-8")
                name_array[: len(name_bytes)] = list(name_bytes)
                names_var[i, :] = name_array

        # Write river data variables
        default_attrs = {
            "river_flux": {"long_name": "river runoff volume flux", "units": "m^3/s"},
            "river_temp": {
                "long_name": "river runoff temperature",
                "units": "degrees Celsius",
            },
            "river_salt": {"long_name": "river runoff salinity", "units": "PSU"},
        }

        for var_name in ["river_flux", "river_temp", "river_salt"]:
            if var_name in data:
                var = ds.createVariable(var_name, "f4", ("time", river_dim))

                # Set attributes
                attrs_key = f"{var_name}_attrs"
                if attrs_key in data:
                    for attr, value in data[attrs_key].items():
                        var.setncattr(attr, value)
                else:
                    # Use default attributes
                    for attr, value in default_attrs[var_name].items():
                        var.setncattr(attr, value)

                # Write data
                var[:] = data[var_name].values


def extend_river_nc_file(
    input_path: Union[Path, str],
    output_path: Union[Path, str],
    extend_to: Union[str, pd.Timestamp],
    method: str = "ffill",
    **kwargs,
) -> None:
    """
    Extend river NetCDF file time series.

    This is a high-level convenience function that combines reading,
    extending, and writing river NetCDF files.

    Parameters
    ----------
    input_path : Path or str
        Input river NetCDF file
    output_path : Path or str
        Output river NetCDF file
    extend_to : str or pd.Timestamp
        Target end datetime (e.g., '2025-12-31 23:00:00')
    method : str
        Extension method ('ffill', 'linear', 'seasonal')
    **kwargs
        Additional arguments passed to the extension function

    Examples
    --------
    >>> extend_river_nc_file(
    ...     'river_2020.nc',
    ...     'river_2020_extended.nc',
    ...     '2021-12-31 23:00:00',
    ...     method='ffill'
    ... )
    """
    from xfvcom.utils.timeseries_utils import (
        extend_timeseries_ffill,
        extend_timeseries_linear,
        extend_timeseries_seasonal,
    )

    # Read input file
    data = read_fvcom_river_nc(input_path)

    # Choose extension function
    extension_funcs = {
        "ffill": extend_timeseries_ffill,
        "linear": extend_timeseries_linear,
        "seasonal": extend_timeseries_seasonal,
    }

    if method not in extension_funcs:
        raise ValueError(f"Unknown extension method: {method}")

    extend_func = extension_funcs[method]

    # Extend each variable
    for var_name in ["river_flux", "river_temp", "river_salt"]:
        if var_name in data:
            data[var_name] = extend_func(data[var_name], extend_to, **kwargs)

    # Update datetime index
    if "river_flux" in data:
        data["datetime"] = data["river_flux"].index
    elif "river_temp" in data:
        data["datetime"] = data["river_temp"].index
    elif "river_salt" in data:
        data["datetime"] = data["river_salt"].index

    # Write output file
    write_fvcom_river_nc(output_path, data)

    print(f"Extended {Path(input_path).name} to {extend_to}")
    print(f"Output saved to {output_path}")
