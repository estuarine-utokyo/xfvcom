"""Stacked area plots for source contribution analysis.

This module provides tools for creating stacked area plots showing
the contribution of different sources (rivers, sewers) to dye
concentration at specified locations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter, DateFormatter

from ..ensemble_analysis import MEMBER_SOURCE_NAMES, get_source_name

if TYPE_CHECKING:
    pass


def load_dye_timeseries_multi_source(
    output_dir: str | Path,
    year: int,
    basename: str,
    members: list[int],
    nodes: list[int],
    sigmas: list[int] | None = None,
    var_name: str = "DYE",
) -> pd.DataFrame:
    """Load dye time series from multiple member directories.

    Parameters
    ----------
    output_dir : str or Path
        Base output directory (e.g., 'goto2023/dye_run/output')
    year : int
        Simulation year (e.g., 2021)
    basename : str
        Case basename (e.g., 'tb_w18_r16')
    members : list of int
        List of member IDs to load (e.g., [1, 2, 3, ..., 18])
    nodes : list of int
        Node indices (0-based) to extract
    sigmas : list of int, optional
        Sigma layer indices (0-based). If None, uses surface layer [0].
    var_name : str
        Variable name in NetCDF files (default: 'DYE')

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and columns for each member,
        columns named by source name (e.g., 'Arakawa', 'Sumida', ...)
    """
    import xarray as xr

    output_dir = Path(output_dir)
    if sigmas is None:
        sigmas = [0]  # Surface layer by default

    all_series = {}

    for member in members:
        # Construct path to member output directory
        member_dir = output_dir / str(year) / str(member)

        # Find NetCDF files (exclude restart files)
        pattern = f"{basename}_{year}_{member}_*.nc"
        files = sorted([f for f in member_dir.glob(pattern) if "restart" not in f.name])

        if not files:
            print(
                f"Warning: No files found for member {member} in {member_dir}",
                file=sys.stderr,
            )
            continue

        # Open and concatenate files
        if len(files) == 1:
            ds = xr.open_dataset(files[0], decode_times=False)
        else:
            ds = xr.open_mfdataset(
                files,
                combine="nested",
                concat_dim="time",
                data_vars="minimal",
                coords="minimal",
                compat="override",
                parallel=False,
                decode_times=False,
            )

        # Decode time from MJD
        from ..dye_timeseries import decode_fvcom_time

        ds = decode_fvcom_time(ds, time_key="time")

        # Extract DYE variable
        if var_name not in ds:
            print(
                f"Warning: Variable '{var_name}' not found for member {member}",
                file=sys.stderr,
            )
            ds.close()
            continue

        dye = ds[var_name]

        # Select nodes and sigma layers, then average
        dye_sel = dye.isel(node=nodes, siglay=sigmas)
        dye_mean = dye_sel.mean(dim=["node", "siglay"])

        # Get source name for this member
        source_name = get_source_name(member, style="short")

        # Convert to Series
        time_index = pd.DatetimeIndex(ds.time.values)
        all_series[source_name] = pd.Series(dye_mean.values, index=time_index)

        ds.close()

    if not all_series:
        raise ValueError(f"No data loaded for any member in {members}")

    # Combine into DataFrame
    df = pd.DataFrame(all_series)

    # Sort columns by member ID order
    ordered_cols = [
        get_source_name(m, style="short")
        for m in members
        if get_source_name(m, style="short") in df.columns
    ]
    df = df[ordered_cols]

    return df


def plot_source_contribution_stack(
    df: pd.DataFrame,
    nodes: list[int],
    output: str | Path | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    figsize: tuple[float, float] = (14, 8),
    fontsize_title: float = 16,
    fontsize_label: float = 14,
    fontsize_tick: float = 12,
    fontsize_legend: float = 10,
    title: str | None = None,
    ylabel: str = "Dye Concentration",
    dpi: int = 300,
    colormap: str = "tab20",
    date_format: str | None = None,
    rotation: float = 0,
) -> dict:
    """Create stacked area plot showing source contributions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index and source columns
    nodes : list of int
        Node indices (for title annotation)
    output : str or Path, optional
        Output file path. If provided, saves figure.
    start : str or pd.Timestamp, optional
        Start time for plot window
    end : str or pd.Timestamp, optional
        End time for plot window
    figsize : tuple
        Figure size (width, height) in inches
    fontsize_title : float
        Title font size
    fontsize_label : float
        Axis label font size
    fontsize_tick : float
        Tick label font size
    fontsize_legend : float
        Legend font size
    title : str, optional
        Plot title. If None, auto-generated.
    ylabel : str
        Y-axis label
    dpi : int
        Output resolution (default: 300)
    colormap : str
        Matplotlib colormap name (default: 'tab20')
    date_format : str, optional
        Date format string for x-axis (e.g., '%Y-%m-%d'). If None, auto-format.
    rotation : float
        X-axis label rotation in degrees (default: 0, no slant)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'fig': matplotlib Figure object
        - 'ax': matplotlib Axes object
        - 'data_used': DataFrame with data that was plotted
    """
    print("=" * 70, file=sys.stdout)
    print("SOURCE CONTRIBUTION STACKED AREA PLOT", file=sys.stdout)
    print("=" * 70, file=sys.stdout)

    # Apply time window
    if start is not None or end is not None:
        if start is None:
            start = df.index.min()
        if end is None:
            end = df.index.max()
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]
        print(f"Time window: {start} to {end}", file=sys.stdout)

    print(
        f"Data shape: {df.shape[0]} timesteps x {df.shape[1]} sources", file=sys.stdout
    )
    print(f"Time range: {df.index.min()} to {df.index.max()}", file=sys.stdout)
    print(f"Sources: {list(df.columns)}", file=sys.stdout)

    # Check for NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values detected", file=sys.stderr)
        df = df.fillna(0)

    # Check for negative values
    has_negatives = (df < 0).any().any()
    if has_negatives:
        min_val = df.min().min()
        print(f"Warning: Negative values detected (min={min_val:.6e})", file=sys.stderr)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get colors from colormap
    from matplotlib import colormaps

    cmap = colormaps[colormap]
    n_sources = len(df.columns)
    colors = [cmap(i % cmap.N) for i in range(n_sources)]

    # Create stacked area plot
    # Reverse order so first source is on top (legend order matches visual)
    labels = list(df.columns)
    ax.stackplot(
        df.index,
        df.T.values[::-1],
        labels=labels[::-1],
        colors=colors[::-1],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.3,
    )

    # Format x-axis
    if date_format is not None:
        # Use user-specified format
        ax.xaxis.set_major_formatter(DateFormatter(date_format))
    else:
        # Auto-format based on time range
        locator = AutoDateLocator(minticks=3, maxticks=10)
        formatter = ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    # Apply rotation (0 = no slant)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation, ha="center")

    # Labels and title
    ax.set_xlabel("Time", fontsize=fontsize_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_label)

    if title is None:
        node_str = ", ".join(str(n) for n in nodes[:5])
        if len(nodes) > 5:
            node_str += f", ... ({len(nodes)} total)"
        title = f"Source Contribution at Node(s): {node_str}"
    ax.set_title(title, fontsize=fontsize_title)

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    # Y-axis starts at 0 if no negative values
    if not has_negatives:
        ax.set_ylim(bottom=0)

    # Legend - reverse order to match visual stacking
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        legend_labels[::-1],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        fontsize=fontsize_legend,
        title="Source",
    )

    # Tick parameters
    ax.tick_params(axis="both", labelsize=fontsize_tick)

    plt.tight_layout()

    # Save if output path provided
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output}", file=sys.stdout)

    print("=" * 70, file=sys.stdout)

    return {
        "fig": fig,
        "ax": ax,
        "data_used": df,
    }


def create_source_contribution_plot(
    output_dir: str | Path,
    year: int,
    basename: str,
    nodes: list[int],
    members: list[int] | None = None,
    sigmas: list[int] | None = None,
    output: str | Path | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    **kwargs,
) -> dict:
    """High-level function to create source contribution stack plot.

    This is the main entry point for creating source contribution plots.
    It loads data from multiple members and creates a stacked area plot.

    Parameters
    ----------
    output_dir : str or Path
        Base output directory (e.g., 'goto2023/dye_run/output')
    year : int
        Simulation year (e.g., 2021)
    basename : str
        Case basename (e.g., 'tb_w18_r16')
    nodes : list of int
        Node indices (0-based) to extract
    members : list of int, optional
        Member IDs to include. If None, uses 1-18 (excludes baseline 0).
    sigmas : list of int, optional
        Sigma layer indices (0-based). If None, uses surface layer [0].
    output : str or Path, optional
        Output file path
    start : str or pd.Timestamp, optional
        Start time for plot window
    end : str or pd.Timestamp, optional
        End time for plot window
    **kwargs
        Additional arguments passed to plot_source_contribution_stack()

    Returns
    -------
    dict
        Dictionary with 'fig', 'ax', 'data_used', and 'df_raw' keys
    """
    if members is None:
        # Default: all individual sources (exclude baseline 0)
        members = list(range(1, 19))

    # Load data
    print(f"Loading data for {len(members)} members...", file=sys.stdout)
    df = load_dye_timeseries_multi_source(
        output_dir=output_dir,
        year=year,
        basename=basename,
        members=members,
        nodes=nodes,
        sigmas=sigmas,
    )

    # Create plot
    result = plot_source_contribution_stack(
        df=df,
        nodes=nodes,
        output=output,
        start=start,
        end=end,
        **kwargs,
    )

    result["df_raw"] = df

    return result
