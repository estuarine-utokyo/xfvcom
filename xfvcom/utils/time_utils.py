# xfvcom/utils/time_utils.py
from __future__ import annotations

import pandas as pd
import xarray as xr


def time_label(
    obj: xr.Dataset | xr.DataArray,
    idx: int,
    fmt: str = "%Y-%m-%d %H:%M:%S",
) -> str:
    """
    Return the time coordinate at *idx* as a formatted string.

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        An xarray object that contains the `time` dimension.
    idx : int
        Positional index along the `time` dimension.
    fmt : str, default "%Y-%m-%d %H:%M:%S"
        `strftime`-compatible format string.

    Returns
    -------
    str
        Formatted timestamp.
    """
    ts = obj.time.isel(time=idx).values  # numpy.datetime64
    return pd.to_datetime(ts).strftime(fmt)
