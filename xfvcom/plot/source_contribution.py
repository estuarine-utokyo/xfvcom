"""Source contribution analysis plots (stacked area and bar charts).

This module provides tools for creating:
- Stacked area plots showing time series of source contributions
- Bar charts showing time-averaged source contributions at specific nodes
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


# =============================================================================
# Bar chart functions for time-averaged source contribution
# =============================================================================


def compute_time_averaged_contribution(
    df: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    as_percentage: bool = False,
) -> pd.Series:
    """Compute time-averaged contribution from time series DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data with datetime index and source columns
    start : str or pd.Timestamp, optional
        Start time for averaging period
    end : str or pd.Timestamp, optional
        End time for averaging period
    as_percentage : bool
        If True, return as percentage (0-100%) of total

    Returns
    -------
    pd.Series
        Time-averaged values for each source
    """
    # Apply time window
    if start is not None or end is not None:
        if start is None:
            start = df.index.min()
        if end is None:
            end = df.index.max()
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]

    # Compute time average
    mean_values = df.mean(axis=0)

    # Convert to percentage if requested
    if as_percentage:
        total = mean_values.sum()
        if total > 0:
            mean_values = (mean_values / total) * 100.0

    return mean_values


def plot_source_contribution_bar(
    data: pd.Series | pd.DataFrame,
    output: str | Path | None = None,
    orientation: str = "horizontal",
    sort_by: str = "value",
    show_values: bool = True,
    as_percentage: bool = False,
    figsize: tuple[float, float] | None = None,
    fontsize_title: float = 14,
    fontsize_label: float = 12,
    fontsize_tick: float = 10,
    fontsize_value: float = 9,
    fontsize_legend: float = 10,
    colormap: str = "tab20",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    value_format: str | None = None,
    dpi: int = 300,
    top_n: int | None = None,
    show_others: bool = True,
) -> dict:
    """Create bar chart showing source contributions.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        - Series: single node's source contributions (index=source names)
        - DataFrame: multi-node comparison (columns=nodes, index=sources)
    output : str or Path, optional
        Output file path. If provided, saves figure.
    orientation : str
        'horizontal' (bars go left-right) or 'vertical' (bars go up-down)
    sort_by : str
        'value': sort by value (descending), 'name': alphabetical, 'none': input order
    show_values : bool
        If True, display values at bar ends
    as_percentage : bool
        If True, format axis as percentage (0-100%)
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated.
    fontsize_title : float
        Title font size
    fontsize_label : float
        Axis label font size
    fontsize_tick : float
        Tick label font size
    fontsize_value : float
        Value annotation font size
    fontsize_legend : float
        Legend font size (for multi-node mode)
    colormap : str
        Matplotlib colormap name
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    value_format : str, optional
        Format string for value annotations (e.g., '{:.2f}', '{:.1%}')
    dpi : int
        Output resolution
    top_n : int, optional
        If provided, only show top N sources (by value)
    show_others : bool
        If top_n is set, whether to show "Others" category for remaining sources

    Returns
    -------
    dict
        Dictionary with 'fig', 'ax', 'data_used' keys
    """
    from matplotlib import colormaps

    print("=" * 70, file=sys.stdout)
    print("SOURCE CONTRIBUTION BAR CHART", file=sys.stdout)
    print("=" * 70, file=sys.stdout)

    # Determine if multi-node mode
    is_multi_node = isinstance(data, pd.DataFrame)

    if is_multi_node:
        print(
            f"Mode: Multi-node comparison ({len(data.columns)} nodes)", file=sys.stdout
        )
        print(f"Sources: {len(data.index)}", file=sys.stdout)
    else:
        print(f"Mode: Single node", file=sys.stdout)
        print(f"Sources: {len(data)}", file=sys.stdout)

    # Handle top_n filtering
    if top_n is not None and not is_multi_node:
        if len(data) > top_n:
            sorted_data = data.sort_values(ascending=False)
            top_sources = sorted_data.head(top_n)
            if show_others:
                others_sum = sorted_data.iloc[top_n:].sum()
                top_sources = pd.concat(
                    [top_sources, pd.Series({"Others": others_sum})]
                )
            data = top_sources
            print(
                f"Showing top {top_n} sources" + (" + Others" if show_others else ""),
                file=sys.stdout,
            )

    # Sort data
    if not is_multi_node:
        if sort_by == "value":
            data = data.sort_values(
                ascending=True
            )  # ascending for horizontal bars (top = highest)
        elif sort_by == "name":
            data = data.sort_index(ascending=False)
        # 'none' keeps original order

    # Auto-calculate figure size
    if figsize is None:
        if is_multi_node:
            n_sources = len(data.index)
            n_nodes = len(data.columns)
            if orientation == "horizontal":
                figsize = (10, max(6, n_sources * 0.4 * n_nodes))
            else:
                figsize = (max(8, n_sources * 0.8), 8)
        else:
            n_sources = len(data)
            if orientation == "horizontal":
                figsize = (10, max(4, n_sources * 0.35))
            else:
                figsize = (max(8, n_sources * 0.6), 6)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get colors
    cmap = colormaps[colormap]

    if is_multi_node:
        # Multi-node grouped bar chart
        n_sources = len(data.index)
        n_nodes = len(data.columns)
        bar_width = 0.8 / n_nodes
        x = np.arange(n_sources)

        for i, node in enumerate(data.columns):
            offset = (i - n_nodes / 2 + 0.5) * bar_width
            values = data[node].values
            color = cmap(i % cmap.N)

            if orientation == "vertical":
                bars = ax.bar(
                    x + offset,
                    values,
                    bar_width,
                    label=f"Node {node}",
                    color=color,
                    alpha=0.85,
                )
                if show_values:
                    for bar, val in zip(bars, values):
                        if val > 0:
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                bar.get_height(),
                                _format_value(val, as_percentage, value_format),
                                ha="center",
                                va="bottom",
                                fontsize=fontsize_value,
                                rotation=90,
                            )
            else:
                bars = ax.barh(
                    x + offset,
                    values,
                    bar_width,
                    label=f"Node {node}",
                    color=color,
                    alpha=0.85,
                )
                if show_values:
                    for bar, val in zip(bars, values):
                        if val > 0:
                            ax.text(
                                bar.get_width(),
                                bar.get_y() + bar.get_height() / 2,
                                " " + _format_value(val, as_percentage, value_format),
                                ha="left",
                                va="center",
                                fontsize=fontsize_value,
                            )

        # Set tick labels
        if orientation == "vertical":
            ax.set_xticks(x)
            ax.set_xticklabels(data.index, rotation=45, ha="right")
        else:
            ax.set_yticks(x)
            ax.set_yticklabels(data.index)

        # Legend
        ax.legend(loc="best", fontsize=fontsize_legend)

    else:
        # Single node bar chart
        sources = data.index.tolist()
        values = data.values
        n_sources = len(sources)
        colors = [cmap(i % cmap.N) for i in range(n_sources)]

        if orientation == "horizontal":
            y_pos = np.arange(n_sources)
            bars = ax.barh(
                y_pos,
                values,
                color=colors,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sources)

            if show_values:
                for bar, val in zip(bars, values):
                    # Position text at the end of bar
                    ax.text(
                        bar.get_width(),
                        bar.get_y() + bar.get_height() / 2,
                        " " + _format_value(val, as_percentage, value_format),
                        ha="left",
                        va="center",
                        fontsize=fontsize_value,
                    )
        else:
            x_pos = np.arange(n_sources)
            bars = ax.bar(
                x_pos,
                values,
                color=colors,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(sources, rotation=45, ha="right")

            if show_values:
                for bar, val in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        _format_value(val, as_percentage, value_format),
                        ha="center",
                        va="bottom",
                        fontsize=fontsize_value,
                    )

    # Labels and title
    if title:
        ax.set_title(title, fontsize=fontsize_title)

    if orientation == "horizontal":
        if xlabel is None:
            xlabel = "Percentage (%)" if as_percentage else "Dye Concentration"
        if ylabel is None:
            ylabel = "Source"
        ax.set_xlabel(xlabel, fontsize=fontsize_label)
        ax.set_ylabel(ylabel, fontsize=fontsize_label)
        ax.set_xlim(left=0)
    else:
        if xlabel is None:
            xlabel = "Source"
        if ylabel is None:
            ylabel = "Percentage (%)" if as_percentage else "Dye Concentration"
        ax.set_xlabel(xlabel, fontsize=fontsize_label)
        ax.set_ylabel(ylabel, fontsize=fontsize_label)
        ax.set_ylim(bottom=0)

    # Grid
    if orientation == "horizontal":
        ax.grid(True, alpha=0.3, linestyle="--", axis="x")
    else:
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")

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
        "data_used": data,
    }


def _format_value(val: float, as_percentage: bool, value_format: str | None) -> str:
    """Format a value for display on bar chart."""
    if value_format:
        return value_format.format(val)
    elif as_percentage:
        return f"{val:.1f}%"
    else:
        if abs(val) >= 1:
            return f"{val:.2f}"
        elif abs(val) >= 0.01:
            return f"{val:.3f}"
        else:
            return f"{val:.2e}"


def create_source_contribution_bar_plot(
    output_dir: str | Path,
    year: int,
    basename: str,
    nodes: list[int],
    members: list[int] | None = None,
    sigmas: list[int] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    as_percentage: bool = False,
    output: str | Path | None = None,
    **kwargs,
) -> dict:
    """High-level function to create source contribution bar chart.

    This is the main entry point for creating time-averaged bar charts.
    It loads data, computes time average, and creates the bar chart.

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
    start : str or pd.Timestamp, optional
        Start time for averaging period
    end : str or pd.Timestamp, optional
        End time for averaging period
    as_percentage : bool
        If True, show as percentage of total
    output : str or Path, optional
        Output file path
    **kwargs
        Additional arguments passed to plot_source_contribution_bar()

    Returns
    -------
    dict
        Dictionary with 'fig', 'ax', 'data_used', 'df_raw', and 'mean_values' keys
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

    # Compute time average
    print("Computing time average...", file=sys.stdout)
    mean_values = compute_time_averaged_contribution(
        df=df,
        start=start,
        end=end,
        as_percentage=as_percentage,
    )

    print(f"Time range: {df.index.min()} to {df.index.max()}", file=sys.stdout)
    if start or end:
        print(
            f"Averaging period: {start or df.index.min()} to {end or df.index.max()}",
            file=sys.stdout,
        )

    # Generate default title
    if "title" not in kwargs or kwargs.get("title") is None:
        node_str = ", ".join(
            str(n + 1) for n in nodes[:3]
        )  # Convert to 1-based for display
        if len(nodes) > 3:
            node_str += f", ... ({len(nodes)} nodes)"
        period_str = ""
        if start or end:
            period_str = f"\n({start or 'start'} to {end or 'end'})"
        kwargs["title"] = (
            f"Time-Averaged Source Contribution at Node(s): {node_str}{period_str}"
        )

    # Create plot
    result = plot_source_contribution_bar(
        data=mean_values,
        output=output,
        as_percentage=as_percentage,
        **kwargs,
    )

    result["df_raw"] = df
    result["mean_values"] = mean_values

    return result
