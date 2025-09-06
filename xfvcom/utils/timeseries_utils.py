"""
Utility functions for time series manipulation and extension.

This module provides functions for extending time series data using various methods
including forward fill, linear extrapolation, and seasonal pattern repetition.
These utilities are particularly useful for preparing forcing data for FVCOM simulations.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd


def extend_timeseries_ffill(
    df: pd.DataFrame, extend_to: Union[str, pd.Timestamp], freq: Optional[str] = None
) -> pd.DataFrame:
    """
    Extend time series using forward fill method.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex containing time series data
    extend_to : str or pd.Timestamp
        Target end datetime to extend the series to
    freq : str, optional
        Frequency string (e.g., '1h', '24h'). If None, inferred from data

    Returns
    -------
    pd.DataFrame
        Extended DataFrame with forward-filled values

    Examples
    --------
    >>> data = pd.DataFrame({'discharge': [10, 20, 30]},
    ...                     index=pd.date_range('2020-01-01', periods=3, freq='D'))
    >>> extended = extend_timeseries_ffill(data, '2020-01-10')
    >>> len(extended)
    10
    """
    extend_to = pd.Timestamp(extend_to)

    if extend_to <= df.index[-1]:
        return df

    # Infer frequency if not provided
    if freq is None:
        freq = pd.infer_freq(df.index)
        if freq is None and len(df.index) > 1:
            # Calculate from first two timestamps
            delta = df.index[1] - df.index[0]
            hours = delta.total_seconds() / 3600
            # Round to nearest hour if within 0.01% tolerance (handles millisecond precision issues)
            hours_rounded = round(hours)
            if abs(hours - hours_rounded) < 0.0001:
                freq = f"{hours_rounded}h"
            else:
                freq = f"{int(hours * 60)}min"

    # Create extended index
    extended_index = pd.date_range(start=df.index[0], end=extend_to, freq=freq)

    # Reindex and forward fill
    return df.reindex(extended_index).ffill()


def extend_timeseries_linear(
    df: pd.DataFrame,
    extend_to: Union[str, pd.Timestamp],
    freq: Optional[str] = None,
    lookback_periods: int = 30,
) -> pd.DataFrame:
    """
    Extend time series using linear extrapolation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex containing time series data
    extend_to : str or pd.Timestamp
        Target end datetime to extend the series to
    freq : str, optional
        Frequency string. If None, inferred from data
    lookback_periods : int
        Number of periods to use for calculating the trend

    Returns
    -------
    pd.DataFrame
        Extended DataFrame with linearly extrapolated values
    """
    extend_to = pd.Timestamp(extend_to)

    if extend_to <= df.index[-1]:
        return df

    # Infer frequency if not provided
    if freq is None:
        freq = pd.infer_freq(df.index)
        if freq is None and len(df.index) > 1:
            delta = df.index[1] - df.index[0]
            hours = delta.total_seconds() / 3600
            # Round to nearest hour if within 0.01% tolerance (handles millisecond precision issues)
            hours_rounded = round(hours)
            if abs(hours - hours_rounded) < 0.0001:
                freq = f"{hours_rounded}h"
            else:
                freq = f"{int(hours * 60)}min"

    # Create extended index
    extended_index = pd.date_range(start=df.index[0], end=extend_to, freq=freq)
    n_original = len(df)
    n_extended = len(extended_index)

    # Calculate linear trend from last N periods
    lookback_start = max(0, n_original - lookback_periods)
    result = df.reindex(extended_index)

    for col in df.columns:
        # Fit linear trend to last N points
        y = df[col].iloc[lookback_start:].values
        x = np.arange(len(y))

        if len(y) > 1:
            # Calculate slope and intercept
            slope = np.polyfit(x, y, 1)[0]
            last_val = y[-1]

            # Extrapolate
            n_new_points = n_extended - n_original
            new_x = np.arange(len(y), len(y) + n_new_points)
            new_y = last_val + slope * (new_x - len(y) + 1)

            # Preserve the original dtype when assigning
            result.loc[result.index[n_original:], col] = new_y.astype(result[col].dtype)
        else:
            # Fall back to forward fill if not enough data
            result[col] = result[col].ffill()

    return result


def extend_timeseries_seasonal(
    df: pd.DataFrame,
    extend_to: Union[str, pd.Timestamp],
    freq: Optional[str] = None,
    period: str = "1Y",
) -> pd.DataFrame:
    """
    Extend time series by repeating seasonal patterns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex containing time series data
    extend_to : str or pd.Timestamp
        Target end datetime to extend the series to
    freq : str, optional
        Frequency string. If None, inferred from data
    period : str
        Period to repeat (e.g., '1Y' for yearly, '1M' for monthly)

    Returns
    -------
    pd.DataFrame
        Extended DataFrame with repeated seasonal patterns
    """
    extend_to = pd.Timestamp(extend_to)

    if extend_to <= df.index[-1]:
        return df

    # Infer frequency if not provided
    if freq is None:
        freq = pd.infer_freq(df.index)
        if freq is None and len(df.index) > 1:
            delta = df.index[1] - df.index[0]
            hours = delta.total_seconds() / 3600
            # Round to nearest hour if within 0.01% tolerance (handles millisecond precision issues)
            hours_rounded = round(hours)
            if abs(hours - hours_rounded) < 0.0001:
                freq = f"{hours_rounded}h"
            else:
                freq = f"{int(hours * 60)}min"

    # Create extended index
    extended_index = pd.date_range(start=df.index[0], end=extend_to, freq=freq)

    # Determine period length
    # Convert period string to days for yearly/monthly periods
    if period == "1Y":
        period_days = 365
    elif period == "1M":
        period_days = 30
    else:
        # Try to parse as Timedelta
        try:
            period_delta = pd.Timedelta(period)
            period_days = period_delta.days
        except:
            # Default to 365 days
            period_days = 365

    period_delta = pd.Timedelta(days=period_days)

    # Find the last complete period in the original data
    last_period_start = df.index[-1] - period_delta
    if last_period_start < df.index[0]:
        # If we don't have a full period, use all available data as the pattern
        pattern_data = df.copy()
    else:
        # Extract the pattern from the last complete period
        pattern_data = df.loc[last_period_start : df.index[-1]].copy()

    # Create result DataFrame
    result = df.reindex(extended_index)

    # Fill the extended part by repeating the pattern
    # Get indices where we need to fill (all NaN values after original data)
    n_original = len(df)
    pattern_length = len(pattern_data)

    # Fill all NaN values in the extended portion
    for i in range(n_original, len(result)):
        # Map to pattern index (cycle through pattern)
        pattern_idx = (i - n_original) % pattern_length

        # Copy values from pattern to result
        for col in df.columns:
            value = pattern_data.iloc[pattern_idx][col]
            # Preserve the original dtype when assigning
            if hasattr(result[col], "dtype"):
                value = np.array(value).astype(result[col].dtype).item()
            result.iloc[i, result.columns.get_loc(col)] = value

    return result


def interpolate_missing_values(
    df: pd.DataFrame,
    method: str = "linear",
    limit: Optional[int] = None,
    limit_direction: str = "forward",
) -> pd.DataFrame:
    """
    Interpolate missing values in time series data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with missing values (NaN)
    method : str
        Interpolation method ('linear', 'time', 'nearest', 'cubic', etc.)
    limit : int, optional
        Maximum number of consecutive NaNs to fill
    limit_direction : str
        Direction to limit interpolation ('forward', 'backward', 'both')

    Returns
    -------
    pd.DataFrame
        DataFrame with interpolated values
    """
    return df.interpolate(
        method=method, limit=limit, limit_direction=limit_direction, axis=0
    )


def resample_timeseries(
    df: pd.DataFrame,
    target_freq: str,
    agg_method: str = "mean",
    interpolate: bool = True,
) -> pd.DataFrame:
    """
    Resample time series to a different frequency.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    target_freq : str
        Target frequency (e.g., '1h', '6h', '1D')
    agg_method : str
        Aggregation method for downsampling ('mean', 'sum', 'max', 'min')
    interpolate : bool
        Whether to interpolate when upsampling

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame
    """
    # Determine if upsampling or downsampling
    current_freq = pd.infer_freq(df.index)

    # Resample
    resampler = df.resample(target_freq)

    if agg_method == "mean":
        result = resampler.mean()
    elif agg_method == "sum":
        result = resampler.sum()
    elif agg_method == "max":
        result = resampler.max()
    elif agg_method == "min":
        result = resampler.min()
    else:
        result = resampler.mean()

    # Interpolate if upsampling and requested
    if interpolate and result.isna().any().any():
        result = result.interpolate(method="time")

    return result
