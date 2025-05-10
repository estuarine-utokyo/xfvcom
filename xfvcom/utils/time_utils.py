# xfvcom/utils/time_utils.py
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import xarray as xr


def time_label(
    obj: xr.Dataset | xr.DataArray,
    idx: int | slice | Sequence[int],
    fmt: str = "%Y-%m-%d %H:%M:%S",
) -> str | list[str]:
    """
    Return *time* coordinate(s) at *idx* formatted as string(s).

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        Object that contains a `time` dimension.
    idx : int | slice | Sequence[int]
        Positional index / indices along the `time` dimension.
    fmt : str, default "%Y-%m-%d %H:%M:%S"
        strftime-compatible format.

    Returns
    -------
    str | list[str]
        Single formatted timestamp if *idx* is a scalar,
        otherwise a list of formatted timestamps.
    """
    # Positional selection (int, slice, list, tuple, np.ndarray …)
    sel = obj.time.isel(time=idx)

    # Format as strings via pandas → strftime
    formatted = (
        pd.to_datetime(sel.values).strftime
        if sel.ndim == 0
        else (pd.to_datetime(sel.values).strftime(fmt).tolist())
    )

    # Scalar → str, otherwise list[str]
    return (
        pd.to_datetime(sel.values).strftime(fmt)
        if sel.ndim == 0
        else pd.to_datetime(sel.values).strftime(fmt).tolist()
    )


def sliced_time_label(
    obj: xr.Dataset | xr.DataArray,
    rng: slice,
    fmt: str = "%Y-%m-%d %H:%M:%S",
) -> list[str]:
    """
    Return all timestamps within *rng* formatted as strings.

    Parameters:
    ----------
    obj : xr.Dataset | xr.DataArray
        Object that contains a `time` dimension.
    rng : slice
        Slice object to select a range of indices along the `time` dimension.
    fmt : str, default "%Y-%m-%d %H:%M:%S"
        strftime-compatible format.

    Returns:
    -------
    list[str]
        List of formatted timestamps within the specified range.
    """
    return obj.sel(time=rng).time.dt.strftime(fmt).values.tolist()
