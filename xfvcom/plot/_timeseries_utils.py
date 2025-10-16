"""Helper utilities for DYE timeseries plotting.

Provides minimal, testable helper functions for data preparation,
NaN detection, member selection, and aggregation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import xarray as xr


def detect_nans_and_raise(df: pd.DataFrame, limit: int = 50) -> None:
    """Check for NaN values and raise immediately if any are found.

    Parameters
    ----------
    df : pd.DataFrame
        Wide DataFrame to check (index=datetime, columns=groups)
    limit : int
        Maximum number of sample (time, column) pairs to report

    Raises
    ------
    ValueError
        If any NaN values are detected, with detailed diagnostics
    """
    nan_mask = df.isna()
    total_nans = nan_mask.sum().sum()

    if total_nans == 0:
        return

    # Collect sample (time, column) pairs
    samples = []
    for col in df.columns:
        col_nans = nan_mask[col]
        if col_nans.any():
            nan_times = df.index[col_nans]
            for t in nan_times[: min(limit, len(nan_times))]:
                samples.append((t, col))
                if len(samples) >= limit:
                    break
        if len(samples) >= limit:
            break

    first_time = samples[0][0] if samples else "N/A"
    sample_str = "\n".join(
        [f"  - {t.isoformat()}, column={col}" for t, col in samples[:limit]]
    )

    raise ValueError(
        f"NaN values detected in data!\n"
        f"  Total NaNs: {total_nans}\n"
        f"  First occurrence: {first_time}\n"
        f"  Sample pairs (time, column) [up to {limit}]:\n{sample_str}\n\n"
        f"Remediation hints:\n"
        f"  - Check for missing data in source files\n"
        f"  - Verify time alignment across members\n"
        f"  - Review data preprocessing steps\n"
        f"  - Use interpolation or gap-filling if appropriate"
    )


def prepare_wide_df(
    data: xr.DataArray | xr.Dataset | pd.DataFrame,
    groups: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Convert various data formats to wide DataFrame.

    Parameters
    ----------
    data : xr.DataArray, xr.Dataset, or pd.DataFrame
        Input data to convert
    groups : Iterable[str], optional
        Column names for wide format (inferred if None)

    Returns
    -------
    pd.DataFrame
        Wide format DataFrame with datetime index and group columns

    Raises
    ------
    ValueError
        If data format is unsupported or conversion fails
    """
    if isinstance(data, pd.DataFrame):
        # Already a DataFrame
        if isinstance(data.index, pd.DatetimeIndex):
            return data
        # Try to parse index as datetime
        try:
            df = data.copy()
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            raise ValueError(f"Cannot convert DataFrame index to datetime: {e}")

    elif hasattr(data, "to_dataframe"):
        # xarray DataArray or Dataset
        try:
            # If it's a Dataset, extract the first/only data variable
            if hasattr(data, "data_vars"):
                data_vars = list(data.data_vars)
                if len(data_vars) == 0:
                    raise ValueError("Dataset has no data variables")
                elif len(data_vars) == 1:
                    # Extract the single data variable
                    data = data[data_vars[0]]
                else:
                    # Multiple variables - take the first one
                    data = data[data_vars[0]]

            # Now data should be a DataArray
            if hasattr(data, "dims"):
                # Check if we have both time and ensemble/member dimensions
                if "time" in data.dims and (
                    "ensemble" in data.dims or "member" in data.dims
                ):
                    # Get the ensemble dimension name
                    ens_dim = "ensemble" if "ensemble" in data.dims else "member"

                    # Get coordinates
                    time_coord = data.coords["time"]
                    ens_coord = data.coords[ens_dim]

                    # Extract column names from ensemble coordinate
                    if isinstance(ens_coord.to_index(), pd.MultiIndex):
                        # Ensemble is MultiIndex (e.g., year, member)
                        # Extract the member level for column names
                        ens_index = ens_coord.to_index()
                        if "member" in ens_index.names:
                            columns = ens_index.get_level_values("member").values
                        else:
                            # Use the last level if 'member' not found
                            columns = ens_index.get_level_values(-1).values
                    else:
                        # Ensemble is simple index
                        columns = ens_coord.values

                    # Get values and ensure correct dimension order (time, ensemble)
                    values = data.values
                    # Check if dimensions need to be transposed
                    if data.dims[0] == ens_dim:
                        # Data is (ensemble, time) - need to transpose to (time, ensemble)
                        values = values.T

                    # Construct DataFrame directly from values
                    df = pd.DataFrame(
                        values,
                        index=pd.DatetimeIndex(time_coord.values),
                        columns=columns,
                    )

                    return df
                else:
                    # Simple case: just convert to DataFrame
                    df = data.to_dataframe()
                    if isinstance(df, pd.Series):
                        df = df.to_frame()
            else:
                df = data.to_dataframe()
                if isinstance(df, pd.Series):
                    df = df.to_frame()

            # Ensure time is the index
            if "time" in df.index.names:
                # Already has time in index
                pass
            elif "time" in df.columns:
                df = df.set_index("time")
            else:
                # Try to find a datetime-like index
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)

            # If index is MultiIndex and has ensemble/member, unstack it
            if isinstance(df.index, pd.MultiIndex):
                for level in df.index.names:
                    if level in ["ensemble", "member"] and level != "time":
                        unstacked = df.unstack(level)
                        # unstack can return Series if only one column
                        if isinstance(unstacked, pd.Series):
                            df = unstacked.to_frame()
                        else:
                            df = unstacked
                        break

            return df

        except Exception as e:
            raise ValueError(f"Cannot convert xarray to DataFrame: {e}")

    else:
        raise ValueError(
            f"Unsupported data type: {type(data)}. "
            f"Expected pd.DataFrame, xr.DataArray, or xr.Dataset"
        )


def select_members(
    data_or_df: xr.DataArray | xr.Dataset | pd.DataFrame,
    member_ids: list[int],
    member_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Select specific members by integer ID.

    Priority:
    1. If xarray: use integer 'member' coordinate
    2. If member_map provided: map IDs to column names
    3. If columns match 'member_<int>': select those
    4. Otherwise raise KeyError

    Parameters
    ----------
    data_or_df : xr.DataArray, xr.Dataset, or pd.DataFrame
        Input data
    member_ids : list[int]
        List of integer member IDs to select
    member_map : dict[int, str], optional
        Mapping from member ID to column name

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with only requested members, in requested order

    Raises
    ------
    KeyError
        If requested members cannot be found
    """
    # Convert to DataFrame if xarray
    if hasattr(data_or_df, "to_dataframe"):
        # Import xarray type for proper type checking
        from typing import cast

        import xarray as xr

        # Try to select by member coordinate first
        if hasattr(data_or_df, "member") and "member" in data_or_df.coords:
            try:
                # Cast to xarray type so mypy knows .sel() exists
                xr_data = cast("xr.DataArray | xr.Dataset", data_or_df)
                selected = xr_data.sel(member=member_ids)
                df = prepare_wide_df(selected)
                # Columns should now be integers matching member_ids
                # Rename columns if member_map provided
                if member_map:
                    # Create mapping from current column names to new names
                    rename_map = {}
                    for mid in member_ids:
                        if mid in df.columns:
                            rename_map[mid] = member_map.get(mid, f"member_{mid}")
                    df = df.rename(columns=rename_map)
                return df
            except Exception as e:
                raise KeyError(
                    f"Cannot select members {member_ids} from xarray using 'member' coordinate: {e}"
                )
        elif hasattr(data_or_df, "ensemble") and "ensemble" in data_or_df.coords:
            try:
                # Cast to xarray type so mypy knows .sel() exists
                from typing import cast

                import xarray as xr

                xr_data = cast("xr.DataArray | xr.Dataset", data_or_df)
                selected = xr_data.sel(ensemble=member_ids)
                df = prepare_wide_df(selected)
                # Rename columns if member_map provided
                if member_map:
                    rename_map = {}
                    for mid in member_ids:
                        if mid in df.columns:
                            rename_map[mid] = member_map.get(mid, f"member_{mid}")
                    df = df.rename(columns=rename_map)
                return df
            except Exception as e:
                raise KeyError(
                    f"Cannot select members {member_ids} from xarray using 'ensemble' coordinate: {e}"
                )
        else:
            df = prepare_wide_df(data_or_df)
    else:
        df = data_or_df

    # Check if columns are MultiIndex (e.g., from ensemble with year, member)
    if isinstance(df.columns, pd.MultiIndex):
        # Find the member level in the MultiIndex
        if "member" in df.columns.names:
            member_level = df.columns.names.index("member")
            # Select columns where member matches requested IDs
            cols_to_select = [
                col for col in df.columns if col[member_level] in member_ids
            ]
            selected_df = df[cols_to_select]

            # Flatten to single level using just the member ID
            if member_map:
                # Rename using member_map
                new_cols = [
                    member_map.get(col[member_level], f"member_{col[member_level]}")
                    for col in selected_df.columns
                ]
            else:
                # Use just the member ID
                new_cols = [col[member_level] for col in selected_df.columns]
            selected_df.columns = new_cols
            return selected_df
        else:
            # MultiIndex but no 'member' level - flatten and try again
            df.columns = [
                "_".join(map(str, col)) if isinstance(col, tuple) else str(col)
                for col in df.columns
            ]

    # Now we have a DataFrame - try different selection strategies
    if member_map:
        # Check if we can select by integer IDs first, then rename
        if all(mid in df.columns for mid in member_ids):
            # Select by integer IDs
            selected_df = df[member_ids]
            # Rename using member_map
            rename_map = {
                mid: member_map.get(mid, f"member_{mid}") for mid in member_ids
            }
            return selected_df.rename(columns=rename_map)

        # Otherwise try to use member_map column names directly
        cols = [member_map.get(mid) for mid in member_ids]
        if None in cols:
            missing = [mid for mid, col in zip(member_ids, cols) if col is None]
            raise KeyError(
                f"member_map missing entries for IDs: {missing}. "
                f"Available: {list(member_map.keys())}"
            )
        try:
            return df[cols]
        except KeyError as e:
            raise KeyError(
                f"Columns {cols} not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

    # Try member_<int> pattern
    member_cols = [f"member_{mid}" for mid in member_ids]
    if all(col in df.columns for col in member_cols):
        return df[member_cols]

    # Try direct integer column names
    if all(mid in df.columns for mid in member_ids):
        return df[member_ids]

    # Failed to select
    raise KeyError(
        f"Cannot select members {member_ids}. "
        f"Available columns: {list(df.columns)}. "
        f"Provide member_map or ensure columns are named 'member_<id>' or have integer names."
    )


def slice_window(
    df: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Slice DataFrame by time window.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    start : str or pd.Timestamp, optional
        Start time (inclusive)
    end : str or pd.Timestamp, optional
        End time (inclusive)

    Returns
    -------
    pd.DataFrame
        Sliced DataFrame
    """
    if start is None and end is None:
        return df

    if start is not None:
        start = pd.to_datetime(start)
    if end is not None:
        end = pd.to_datetime(end)

    return df.loc[start:end]  # type: ignore[misc]


def resample_df(
    df: pd.DataFrame,
    freq: str,
    normalize: bool = False,
) -> pd.DataFrame:
    """Resample DataFrame to specified frequency.

    For stacked physical fluxes, default is sum per bin.
    If normalize=True, values are normalized to sum to 1.0 per timestep
    AFTER resampling (so the mean is taken, then normalized).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    freq : str
        Resampling frequency (e.g., 'D', 'W', 'M', 'H')
    normalize : bool
        If True, normalize so each row sums to 1.0

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame
    """
    # For fluxes: sum by default (accumulate flux over period)
    resampled = df.resample(freq).sum()

    if normalize:
        row_sums = resampled.sum(axis=1)
        # Avoid division by zero
        row_sums = row_sums.replace(0, np.nan)
        resampled = resampled.div(row_sums, axis=0).fillna(0)

    return resampled


def align_same_clock_across_years(
    df: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """Align data by calendar position across years.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    start : str or pd.Timestamp, optional
        Start of calendar window (month-day only)
    end : str or pd.Timestamp, optional
        End of calendar window (month-day only)

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping year label to aligned DataFrame slice
    """
    result: dict[str, pd.DataFrame] = {}

    years = df.index.year.unique()

    for year in sorted(years):
        # Filter by year - ensure result is DataFrame, not Series
        year_subset = df[df.index.year == year].copy()
        if isinstance(year_subset, pd.Series):
            year_df: pd.DataFrame = year_subset.to_frame()
        else:
            year_df = year_subset

        # Apply calendar-based filtering
        if start is not None or end is not None:
            start_dt = pd.to_datetime(start) if start else None
            end_dt = pd.to_datetime(end) if end else None

            if start_dt and end_dt:
                # Extract month-day for filtering
                mask = (
                    (year_df.index.month >= start_dt.month)
                    & (year_df.index.day >= start_dt.day)
                    & (year_df.index.month <= end_dt.month)
                    & (year_df.index.day <= end_dt.day)
                )
                filtered = year_df[mask]
                # Ensure result is DataFrame, not Series
                if isinstance(filtered, pd.Series):
                    year_df = filtered.to_frame()
                elif isinstance(filtered, pd.DataFrame):
                    year_df = filtered
                else:
                    # Should never happen, but for type safety
                    year_df = pd.DataFrame(filtered)

        result[str(year)] = year_df

    return result


def climatology(
    df: pd.DataFrame,
    kind: Literal["H", "DOW", "M"] = "H",
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute climatological mean and ±1σ of total.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    kind : {"H", "DOW", "M"}
        Grouping kind:
        - "H": Hour of day (0-23)
        - "DOW": Day of week (0=Monday, 6=Sunday)
        - "M": Month (1-12)

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (mean_by_group, std_of_total)
        - mean_by_group: Mean for each column by time group
        - std_of_total: Standard deviation of row-wise total
    """
    if kind == "H":
        grouper = df.index.hour
    elif kind == "DOW":
        grouper = df.index.dayofweek
    elif kind == "M":
        grouper = df.index.month
    else:
        raise ValueError(f"Unknown kind: {kind}. Expected 'H', 'DOW', or 'M'")

    # Mean per group
    clim_mean = df.groupby(grouper).mean()

    # Calculate total for each timestep, then std per group
    total = df.sum(axis=1)
    total_df = pd.DataFrame({"total": total, "group": grouper})
    clim_std = total_df.groupby("group")["total"].std()

    return clim_mean, clim_std


def summarize_negatives(df: pd.DataFrame) -> dict:
    """Summarize negative values in DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check

    Returns
    -------
    dict
        Summary statistics about negative values
    """
    neg_mask = df < 0
    total_negs = neg_mask.sum().sum()

    if total_negs == 0:
        return {
            "total_negatives": 0,
            "any_negatives": False,
        }

    per_column = {}
    for col in df.columns:
        col_negs = neg_mask[col].sum()
        if col_negs > 0:
            per_column[str(col)] = {
                "count": int(col_negs),
                "min_value": float(df[col].min()),
                "share": float(col_negs / len(df)),
            }

    return {
        "total_negatives": int(total_negs),
        "any_negatives": True,
        "global_min": float(df.min().min()),
        "per_column": per_column,
    }


def handle_negatives(
    df: pd.DataFrame,
    policy: Literal["keep", "clip0"] = "keep",
) -> tuple[pd.DataFrame, dict]:
    """Handle negative values according to policy.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    policy : {"keep", "clip0"}
        - "keep": Leave negative values as-is
        - "clip0": Clip negative values to zero

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (processed_df, negative_stats)
    """
    stats = summarize_negatives(df)

    if policy == "keep":
        return df, stats
    elif policy == "clip0":
        return df.clip(lower=0), stats
    else:
        raise ValueError(f"Unknown policy: {policy}. Expected 'keep' or 'clip0'")


def deterministic_colors(
    groups: list[str],
    override: dict[str, str] | None = None,
) -> list[str]:
    """Generate deterministic colors for groups.

    Uses a stable color sequence based on group names.

    Parameters
    ----------
    groups : list[str]
        List of group names
    override : dict[str, str], optional
        Manual color overrides {group_name: color}

    Returns
    -------
    list[str]
        List of colors in same order as groups
    """
    # Default palette: tab20 colors for up to 20 groups
    default_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
    ]

    override = override or {}

    colors = []
    for i, group in enumerate(groups):
        if group in override:
            colors.append(override[group])
        else:
            colors.append(default_colors[i % len(default_colors)])

    return colors


def get_member_color(
    member_id: int,
    colormap: str = "auto",
    custom_colors: dict[int, str] | None = None,
    total_members: int | None = None,
) -> str:
    """Get consistent color for a member ID across all plot types.

    This ensures that member N always gets the same color, regardless of:
    - Which other members are plotted
    - The plot type (line, stacked, etc.)
    - The order of members in the data

    Parameters
    ----------
    member_id : int
        Member ID (1-based integer, e.g., 1, 2, 3, ...)
    colormap : str
        Matplotlib colormap name (default: "auto").
        - "auto": Automatically selects tab20 (≤20 members) or hsv (>20 members)
        - "tab20", "hsv", "rainbow", etc.: Manual colormap selection
    custom_colors : dict[int, str], optional
        Manual color overrides {member_id: color_spec}
    total_members : int, optional
        Total number of members (required for "auto" colormap selection).
        If not provided with "auto", defaults to tab20.

    Returns
    -------
    str
        Hex color string (e.g., "#1f77b4")

    Examples
    --------
    >>> # Auto-select: tab20 for ≤20 members, hsv for >20
    >>> get_member_color(1, colormap="auto", total_members=18)
    '#1f77b4'  # tab20[0]

    >>> # Auto-select with 30 members uses HSV
    >>> get_member_color(25, colormap="auto", total_members=30)
    '#ff8e00'  # hsv color

    >>> # Manual colormap selection
    >>> get_member_color(5, colormap="tab20")
    '#2ca02c'  # tab20[4]

    >>> # Custom color override
    >>> get_member_color(3, custom_colors={3: 'red'})
    'red'

    Notes
    -----
    **Auto-selection logic**:
    - ≤20 members → tab20 (qualitative, distinct colors)
    - >20 members → hsv (continuous, evenly distributed hues)

    This creates visual consistency across:
    - Line plots (plot_ensemble_timeseries)
    - Stacked plots (plot_dye_timeseries_stacked)
    - Statistical plots
    - Custom visualizations
    """
    # Check for custom override first
    if custom_colors and member_id in custom_colors:
        return custom_colors[member_id]

    # Auto-select colormap based on total_members
    if colormap == "auto":
        if total_members is None:
            # Default to tab20 if total_members not provided
            colormap = "tab20"
        elif total_members <= 20:
            colormap = "tab20"
        else:
            colormap = "hsv"

    # Import matplotlib colormaps
    from matplotlib import colormaps

    # Get colormap
    cmap = colormaps[colormap]

    # Get color by member ID (subtract 1 for 0-based indexing)
    # member_id=1 → index 0, member_id=2 → index 1, etc.
    color_idx = (member_id - 1) % cmap.N
    rgba = cmap(color_idx)

    # Convert RGBA to hex
    from matplotlib.colors import to_hex

    return to_hex(rgba)


def get_member_colors(
    member_ids: list[int],
    colormap: str = "auto",
    custom_colors: dict[int, str] | None = None,
) -> list[str]:
    """Get consistent colors for multiple members with auto-selection.

    Parameters
    ----------
    member_ids : list[int]
        List of member IDs
    colormap : str
        Matplotlib colormap name (default: "auto").
        - "auto": Automatically selects tab20 (≤20 members) or hsv (>20 members)
        - "tab20", "hsv", "rainbow", etc.: Manual colormap selection
    custom_colors : dict[int, str], optional
        Manual color overrides {member_id: color_spec}

    Returns
    -------
    list[str]
        List of colors in same order as member_ids

    Examples
    --------
    >>> # Auto-select: uses tab20 for ≤20 members
    >>> get_member_colors([1, 2, 3])
    ['#1f77b4', '#aec7e8', '#ff7f0e']

    >>> # Auto-select: uses hsv for >20 members
    >>> member_ids = list(range(1, 31))  # 30 members
    >>> colors = get_member_colors(member_ids)  # Uses HSV automatically
    >>> len(set(colors))  # All unique
    30

    >>> # Members plotted in different order get same colors
    >>> get_member_colors([3, 1, 2])
    ['#ff7f0e', '#1f77b4', '#aec7e8']

    >>> # Subset of members keeps same colors
    >>> get_member_colors([1, 5, 10])
    ['#1f77b4', '#2ca02c', '#c5b0d5']

    >>> # Manual colormap override
    >>> get_member_colors([1, 2, 3], colormap="rainbow")
    [...rainbow colors...]
    """
    # Auto-detect total_members for auto-selection
    total_members = max(member_ids) if member_ids else 1

    return [
        get_member_color(
            mid,
            colormap=colormap,
            custom_colors=custom_colors,
            total_members=total_members,
        )
        for mid in member_ids
    ]
