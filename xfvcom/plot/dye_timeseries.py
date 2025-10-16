"""Stacked area plots for DYE time series.

Simple stacked area visualization of DYE concentration time series.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

from ._timeseries_utils import (
    detect_nans_and_raise,
    get_member_colors,
    prepare_wide_df,
    select_members,
)
from .config import FvcomPlotConfig

if TYPE_CHECKING:
    import xarray as xr


def plot_dye_timeseries_stacked(
    data: xr.DataArray | xr.Dataset | pd.DataFrame,
    member_ids: list[int] | None = None,
    member_map: dict[int, str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    colors: dict[str, str] | None = None,
    cfg: FvcomPlotConfig | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    ylabel: str = "Dye Concentration",
    output: str | None = None,
    colormap: str = "auto",
    custom_colors: dict[int, str] | None = None,
) -> dict:
    """Create stacked area plot of DYE time series.

    This creates a simple stacked area chart showing the contribution of each
    member to the total concentration over time, preserving the fluctuations
    in the original data.

    **Auto-selects best colormap** based on number of members:
    - ≤20 members: tab20 (qualitative, distinct colors)
    - >20 members: hsv (continuous, evenly distributed hues)

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset or pd.DataFrame
        DYE concentration time series data. If Dataset, extracts first data variable.
    member_ids : list of int, optional
        List of member IDs to include in the plot. If None, uses all members.
    member_map : dict, optional
        Mapping from member ID (int) to custom label (str).
        Example: {1: "Urban", 2: "Forest", 3: "Agriculture"}
    start : str or pd.Timestamp, optional
        Start time for the plot window. If None, uses all data.
    end : str or pd.Timestamp, optional
        End time for the plot window. If None, uses all data.
    colors : dict, optional
        Custom color mapping. Keys are member labels (str), values are color specs.
    cfg : FvcomPlotConfig, optional
        Plot configuration object for consistent styling. If None, uses default configuration
        matching plot_ensemble_timeseries defaults.
    figsize : tuple of float, optional
        Figure size (width, height) in inches. Overrides cfg.figsize if provided.
        If None, uses cfg.figsize (default: (14, 6)).
    title : str, optional
        Plot title. If None, uses default.
    ylabel : str, default "Dye Concentration"
        Y-axis label.
    output : str, optional
        Output file path. If provided, saves figure to this path.
    colormap : str, default "auto"
        Matplotlib colormap name for member colors.
        - "auto": Automatically selects tab20 (≤20) or hsv (>20)
        - "tab20", "hsv", "rainbow", etc.: Manual selection
    custom_colors : dict[int, str], optional
        Manual color overrides for specific member IDs.
        Example: {1: "red", 5: "blue", 10: "#00ff00"}

    Returns
    -------
    dict
        Dictionary with keys:
        - 'fig': matplotlib Figure object
        - 'ax': matplotlib Axes object
        - 'data_used': pandas DataFrame with data that was plotted
        - 'legend_labels': list of legend labels in plot order

    Examples
    --------
    >>> # Simple stacked plot with all members (auto-selects colormap)
    >>> result = plot_dye_timeseries_stacked(ds)
    >>> # 18 members → tab20, 30 members → hsv (automatic!)

    >>> # Select specific members and time window
    >>> result = plot_dye_timeseries_stacked(
    ...     ds,
    ...     member_ids=[1, 2, 3, 4, 5],
    ...     start="2021-01-01",
    ...     end="2021-01-31",
    ...     title="DYE Concentration - January 2021"
    ... )

    >>> # Custom member labels
    >>> result = plot_dye_timeseries_stacked(
    ...     ds,
    ...     member_ids=[1, 2, 3],
    ...     member_map={1: "Urban", 2: "Forest", 3: "Agriculture"},
    ...     ylabel="Concentration (mmol m$^{-3}$)"
    ... )

    >>> # Manual colormap override
    >>> result = plot_dye_timeseries_stacked(
    ...     ds,
    ...     colormap="rainbow",  # Force rainbow colormap
    ... )
    """
    print("=" * 70, file=sys.stdout)
    print("DYE TIMESERIES STACKED AREA PLOT", file=sys.stdout)
    print("=" * 70, file=sys.stdout)

    # Create config with default font sizes matching plot_ensemble_timeseries if not provided
    if cfg is None:
        cfg = FvcomPlotConfig(
            figsize=(14, 6),
            fontsize={
                "xticks": 13,
                "yticks": 13,
                "xlabel": 14,
                "ylabel": 14,
                "title": 15,
                "legend": 12,
            },
            linewidth={"plot": 1.8},
        )

    # Use figsize parameter if provided (overrides config)
    actual_figsize = figsize if figsize is not None else cfg.figsize

    # Step 1: Convert to wide DataFrame
    df = prepare_wide_df(data)
    print(
        f"Data loaded: {df.shape[0]} timesteps × {df.shape[1]} members",
        file=sys.stdout,
    )
    print(f"Time range: {df.index.min()} to {df.index.max()}", file=sys.stdout)

    # Step 2: NaN detection - HARD FAIL
    detect_nans_and_raise(df)
    print("✓ No NaNs detected", file=sys.stdout)

    # Step 3: Member selection
    if member_ids is not None:
        print(f"Selecting members: {member_ids}", file=sys.stdout)
        df = select_members(df, member_ids, member_map)
        print(
            f"After selection: {df.shape[0]} timesteps × {df.shape[1]} members",
            file=sys.stdout,
        )

    # Step 4: Time window selection
    if start is not None or end is not None:
        if start is None:
            start = df.index.min()
        if end is None:
            end = df.index.max()
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]
        print(f"Time window: {start} to {end}", file=sys.stdout)
        print(f"  {len(df)} timesteps in window", file=sys.stdout)

    # Step 5: Check for negative values
    has_negatives = (df < 0).any().any()
    if has_negatives:
        min_val = df.min().min()
        print(
            f"⚠ Warning: Negative values detected (min={min_val:.6e})",
            file=sys.stdout,
        )
        print("  These will be displayed as-is in the plot", file=sys.stdout)

    # Step 6: Create stacked area plot
    fig, ax = plt.subplots(figsize=actual_figsize)

    # Get column labels
    labels = [str(col) for col in df.columns]

    # Determine colors using member-based mapping for consistency
    from typing import Any

    colors_list: list[Any]
    if colors is None:
        # Extract member IDs from column names for consistent color mapping
        extracted_member_ids: list[int | None] = []
        for col in df.columns:
            try:
                # Try to interpret column as integer member ID
                extracted_member_ids.append(int(col))
            except (ValueError, TypeError):
                # If column name is not an integer, use position-based fallback
                extracted_member_ids.append(None)

        # Use member-based color mapping for consistency across plot types
        # This ensures member N always gets the same color as in line plots
        if all(mid is not None for mid in extracted_member_ids):
            # All columns are valid member IDs - use member-based colors
            # Type narrowing: we know all elements are int here
            valid_member_ids = [mid for mid in extracted_member_ids if mid is not None]
            colors_list = get_member_colors(
                valid_member_ids, colormap=colormap, custom_colors=custom_colors
            )
        else:
            # Fallback to position-based colors using specified colormap
            from matplotlib import colormaps

            # Handle "auto" colormap selection
            n_members = len(df.columns)
            if colormap == "auto":
                # Auto-select colormap based on number of members
                actual_colormap = "tab20" if n_members <= 20 else "hsv"
            else:
                actual_colormap = colormap

            cmap = colormaps[actual_colormap]
            colors_list = [cmap(i % cmap.N) for i in range(n_members)]
    else:
        # User-provided custom colors
        colors_list = [colors.get(label, f"C{i}") for i, label in enumerate(labels)]

    # Create stackplot
    # Note: Reverse the order so the first member (top of legend) is visually on top
    # matplotlib.stackplot draws first series at bottom, but legend shows first at top
    ax.stackplot(
        df.index,
        df.T.values[::-1],  # Reverse data order
        labels=labels[::-1],  # Reverse label order
        colors=colors_list[::-1],  # Reverse color order
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )

    # Format x-axis
    locator = AutoDateLocator()
    formatter = ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Labels and title
    ax.set_xlabel("Time", fontsize=cfg.fontsize_xlabel)
    ax.set_ylabel(ylabel, fontsize=cfg.fontsize_ylabel)
    if title is None:
        title = "DYE Concentration Time Series (Stacked)"
    ax.set_title(title, fontsize=cfg.fontsize_title)

    # Grid
    ax.grid(
        True,
        alpha=cfg.grid_alpha,
        linestyle=cfg.grid_linestyle,
        linewidth=cfg.grid_linewidth,
        color=cfg.grid_color,
        axis="y",
    )

    # Y-axis starts at 0 if no negative values
    if not has_negatives:
        ax.set_ylim(bottom=0)

    # Legend - reverse the order to match the visual stacking
    # (since we reversed the stackplot order, we need to reverse legend too)
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],  # Reverse handles
        legend_labels[::-1],  # Reverse labels
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        fontsize=cfg.fontsize_legend,
        title="Member",
    )

    # Tick parameters
    ax.tick_params(
        axis="both",
        labelsize=cfg.fontsize_xticks,
        width=cfg.linewidth_tick_params,
    )

    plt.tight_layout()

    # Save if output path provided
    if output:
        fig.savefig(output, dpi=cfg.dpi, bbox_inches="tight")
        print(f"✓ Saved to: {output}", file=sys.stdout)

    print("=" * 70, file=sys.stdout)
    print()

    return {
        "fig": fig,
        "ax": ax,
        "data_used": df,
        "legend_labels": labels,
    }
