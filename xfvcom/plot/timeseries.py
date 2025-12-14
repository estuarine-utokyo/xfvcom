"""Time series plotting utilities for xfvcom.

This module provides utilities for creating publication-quality time series plots
with intelligent time axis formatting and consistent styling using FvcomPlotConfig.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from ._timeseries_utils import get_member_color
from .config import FvcomPlotConfig
from .variable_meta import get_label

# =============================================================================
# Default plot settings for publication-quality figures
# =============================================================================
DEFAULT_FIGSIZE = (8, 2)
DEFAULT_DPI = 300
DEFAULT_FONTSIZE = 11
DEFAULT_LINEWIDTH = 1.0
DEFAULT_MINTICKS = 6
DEFAULT_MAXTICKS = 12


def apply_smart_time_ticks(
    ax: Axes,
    fig: Figure | None = None,
    minticks: int = DEFAULT_MINTICKS,
    maxticks: int = DEFAULT_MAXTICKS,
    rotation: int = 0,
) -> Axes:
    """Apply smart datetime tick formatting using matplotlib's built-in tools.

    Uses AutoDateLocator + ConciseDateFormatter for clean, non-overlapping labels.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object
    fig : Figure, optional
        Matplotlib figure. If None, gets from ax.figure
    minticks : int
        Minimum number of ticks (default: 6)
    maxticks : int
        Maximum number of ticks (default: 12)
    rotation : int
        Label rotation angle in degrees (default: 0, horizontal)

    Returns
    -------
    Axes
        The modified axes object
    """
    _ = fig  # unused, kept for API compatibility

    # Use matplotlib's intelligent date locator with more ticks
    locator = AutoDateLocator(minticks=minticks, maxticks=maxticks)

    # Use ConciseDateFormatter for clean, non-overlapping labels
    formatter = ConciseDateFormatter(locator)

    # Apply to axis
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Rotate labels if requested (default is horizontal)
    if rotation != 0:
        for label in ax.get_xticklabels():
            label.set_rotation(rotation)
            label.set_ha("right")

    return ax


def plot_timeseries(
    data: xr.DataArray | xr.Dataset,
    var_name: str | None = None,
    ax: Axes | None = None,
    cfg: FvcomPlotConfig | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    xlabel: str = "",
    color: str = "#1f77b4",
    linewidth: float = DEFAULT_LINEWIDTH,
    alpha: float = 1.0,
    minticks: int = DEFAULT_MINTICKS,
    maxticks: int = DEFAULT_MAXTICKS,
    grid: bool = True,
    use_latex: bool = True,
    output: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a publication-quality time series from xarray data.

    Creates a clean, publication-ready time series plot with:
    - LaTeX-formatted axis labels with proper units
    - Horizontal datetime labels with many ticks
    - Compact figure size suitable for papers
    - High DPI output (300)

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Data to plot. If Dataset, var_name must be specified.
    var_name : str, optional
        Variable name to plot (required if data is Dataset)
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure
    cfg : FvcomPlotConfig, optional
        Plot configuration for styling
    title : str, optional
        Plot title. If None, no title is shown
    ylabel : str, optional
        Y-axis label. If None, auto-generates from variable metadata with LaTeX units
    xlabel : str
        X-axis label (default: "" - no label, datetime ticks are self-explanatory)
    color : str
        Line color (default: "#1f77b4" - matplotlib default blue)
    linewidth : float
        Line width (default: 1.0)
    alpha : float
        Line transparency (default: 1.0)
    minticks : int
        Minimum number of x-axis ticks (default: 6)
    maxticks : int
        Maximum number of x-axis ticks (default: 12)
    grid : bool
        Whether to show grid (default: True)
    use_latex : bool
        Use LaTeX-formatted units in labels (default: True)
    output : str or Path, optional
        Path to save figure. Parent directories created automatically.
    dpi : int
        DPI for saved figure (default: 300)
    **kwargs
        Additional arguments passed to ax.plot()

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects

    Examples
    --------
    >>> import xarray as xr
    >>> from xfvcom.plot import plot_timeseries
    >>>
    >>> # Simple one-liner for publication plot
    >>> ds = xr.open_dataset("weather.nc")
    >>> plot_timeseries(ds["air_temperature"].isel(node=0), output="temp.png")
    >>>
    >>> # Customize appearance
    >>> plot_timeseries(
    ...     ds["air_temperature"].isel(node=0),
    ...     title="Station A",
    ...     color="red",
    ...     output="temp_custom.png",
    ... )
    """
    # Handle Dataset vs DataArray input
    if isinstance(data, xr.Dataset):
        if var_name is None:
            raise ValueError("var_name must be specified when data is a Dataset")
        if var_name not in data:
            raise ValueError(
                f"Variable '{var_name}' not found. Available: {list(data.data_vars)}"
            )
        da = data[var_name]
    else:
        da = data
        if var_name is None:
            var_name = da.name or "value"

    # Create default config optimized for publication
    if cfg is None:
        cfg = FvcomPlotConfig(
            figsize=DEFAULT_FIGSIZE,
            fontsize={
                "xticks": DEFAULT_FONTSIZE,
                "yticks": DEFAULT_FONTSIZE,
                "xlabel": DEFAULT_FONTSIZE,
                "ylabel": DEFAULT_FONTSIZE,
                "title": DEFAULT_FONTSIZE + 2,
            },
            linewidth={"plot": linewidth},
        )

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=cfg.figsize)
    else:
        fig = ax.figure

    # Get time coordinate
    time_coord = None
    for coord_name in ["time", "Time", "datetime"]:
        if coord_name in da.coords:
            time_coord = coord_name
            break

    if time_coord is None:
        raise ValueError("No time coordinate found in data")

    time_values = da[time_coord].values
    data_values = da.values

    # Plot
    ax.plot(
        time_values,
        data_values,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        **kwargs,
    )

    # Apply smart time ticks (horizontal, many ticks)
    apply_smart_time_ticks(ax, fig, minticks=minticks, maxticks=maxticks, rotation=0)

    # Set y-axis label with LaTeX units from variable metadata
    if ylabel is None:
        ylabel = get_label(da, var_name, use_latex=use_latex)
    ax.set_ylabel(ylabel, fontsize=cfg.fontsize_ylabel)

    # Set x-axis label (usually empty for datetime axes)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=cfg.fontsize_xlabel)

    # Set title (optional)
    if title is not None:
        ax.set_title(title, fontsize=cfg.fontsize_title)

    # Grid
    if grid:
        ax.grid(
            True,
            alpha=cfg.grid_alpha,
            linestyle=cfg.grid_linestyle,
            linewidth=cfg.grid_linewidth,
            color=cfg.grid_color,
        )

    # Tick parameters
    ax.tick_params(axis="both", labelsize=cfg.fontsize_xticks)

    # Adjust layout
    fig.tight_layout()

    # Save if output path provided
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
        print(f"Saved: {output_path}")

    return fig, ax


def plot_timeseries_multi(
    ds: xr.Dataset,
    var_names: list[str],
    cfg: FvcomPlotConfig | None = None,
    use_latex: bool = True,
    grid: bool = True,
    minticks: int = DEFAULT_MINTICKS,
    maxticks: int = DEFAULT_MAXTICKS,
    output: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
    panel_height: float = 1.6,
    suptitle: str | None = None,
    **kwargs: Any,
) -> tuple[Figure, list[Axes]]:
    """Plot multiple time series variables in a single multi-panel figure.

    Creates a vertically stacked figure with one panel per variable,
    saved as a single PNG file.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the variables to plot
    var_names : list[str]
        List of variable names to plot
    cfg : FvcomPlotConfig, optional
        Plot configuration for styling
    use_latex : bool
        Use LaTeX-formatted units in labels (default: True)
    grid : bool
        Whether to show grid (default: True)
    minticks : int
        Minimum number of x-axis ticks (default: 6)
    maxticks : int
        Maximum number of x-axis ticks (default: 12)
    output : str or Path, optional
        Path to save figure as single PNG file.
    dpi : int
        DPI for saved figure (default: 300)
    panel_height : float
        Height of each panel in inches (default: 1.6)
    suptitle : str, optional
        Super title displayed above all panels
    **kwargs
        Additional arguments passed to ax.plot()

    Returns
    -------
    tuple[Figure, list[Axes]]
        Matplotlib figure and list of axes objects

    Examples
    --------
    >>> import xarray as xr
    >>> from xfvcom.plot import plot_timeseries_multi
    >>>
    >>> ds = xr.open_dataset("weather.nc")
    >>> fig, axes = plot_timeseries_multi(
    ...     ds.isel(node=0),
    ...     ["air_temperature", "short_wave", "relative_humidity"],
    ...     output="weather_multi.png",
    ... )
    """
    n_vars = len(var_names)
    if n_vars == 0:
        raise ValueError("var_names must not be empty")

    # Filter to valid variables only
    valid_vars = [v for v in var_names if v in ds.data_vars]
    if not valid_vars:
        raise ValueError(
            f"None of the variables found in dataset. "
            f"Available: {list(ds.data_vars)}"
        )

    n_panels = len(valid_vars)

    # Create default config optimized for compact multi-panel display
    if cfg is None:
        fontsize = 9  # Smaller font for multi-panel
        cfg = FvcomPlotConfig(
            figsize=(8, panel_height * n_panels),
            fontsize={
                "xticks": fontsize,
                "yticks": fontsize,
                "xlabel": fontsize,
                "ylabel": fontsize,
                "title": fontsize + 1,
            },
            linewidth={"plot": DEFAULT_LINEWIDTH},
        )

    # Create figure with subplots
    fig_width = cfg.figsize[0] if cfg.figsize else 8
    fig_height = panel_height * n_panels
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(fig_width, fig_height), sharex=True
    )

    # Handle single panel case
    if n_panels == 1:
        axes = [axes]

    # Get time coordinate name
    time_coord = None
    for coord_name in ["time", "Time", "datetime"]:
        if coord_name in ds.coords:
            time_coord = coord_name
            break

    if time_coord is None:
        raise ValueError("No time coordinate found in dataset")

    # Plot each variable
    for i, var_name in enumerate(valid_vars):
        ax = axes[i]
        da = ds[var_name]

        time_values = da[time_coord].values
        data_values = da.values

        # Plot
        ax.plot(
            time_values,
            data_values,
            color="#1f77b4",
            linewidth=cfg.linewidth_plot,
            **kwargs,
        )

        # Y-axis label with units
        ylabel = get_label(da, var_name, use_latex=use_latex)
        ax.set_ylabel(ylabel, fontsize=cfg.fontsize_ylabel)

        # Grid
        if grid:
            ax.grid(
                True,
                alpha=cfg.grid_alpha,
                linestyle=cfg.grid_linestyle,
                linewidth=cfg.grid_linewidth,
                color=cfg.grid_color,
            )

        # Tick parameters
        ax.tick_params(axis="both", labelsize=cfg.fontsize_xticks)

    # Apply smart time ticks to bottom panel only (shared x-axis)
    apply_smart_time_ticks(
        axes[-1], fig, minticks=minticks, maxticks=maxticks, rotation=0
    )

    # Add super title if provided
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=cfg.fontsize_title, y=1.0)

    # Adjust layout
    fig.tight_layout()

    # Adjust top margin if suptitle is present
    if suptitle is not None:
        fig.subplots_adjust(top=0.96)

    # Save if output path provided
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
        print(f"Saved: {output_path}")

    return fig, axes


def plot_ensemble_timeseries(
    ds: xr.Dataset,
    var_name: str = "dye",
    ax: Axes | None = None,
    cfg: FvcomPlotConfig | None = None,
    max_lines: int | None = None,
    alpha: float = 0.7,
    legend_outside: bool = True,
    title: str | None = None,
    ylabel: str | None = None,
    minticks: int = 3,
    maxticks: int = 7,
    rotation: int = 30,
    colormap: str = "auto",
    custom_colors: dict[int, str] | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plot time series for ensemble data with automatic datetime formatting.

    Uses matplotlib's AutoDateLocator and ConciseDateFormatter for clean,
    non-overlapping datetime labels.

    **Auto-selects best colormap** based on number of members:
    - ≤20 members: tab20 (qualitative, distinct colors)
    - >20 members: hsv (continuous, evenly distributed hues)

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the variable to plot
    var_name : str
        Variable name to plot (default: "dye")
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure
    cfg : FvcomPlotConfig, optional
        Plot configuration. If None, uses default with increased font sizes
    max_lines : int, optional
        Maximum number of ensemble members to plot. If None, plots all members (default: None)
    alpha : float
        Line transparency (default: 0.7)
    legend_outside : bool
        Place legend outside plot area (default: True)
    title : str, optional
        Plot title. If None, auto-generates from var_name
    ylabel : str, optional
        Y-axis label. If None, uses var_name
    minticks : int
        Minimum number of ticks on x-axis (default: 3)
    maxticks : int
        Maximum number of ticks on x-axis (default: 7)
    rotation : int
        Rotation angle for x-axis labels in degrees (default: 30)
    colormap : str
        Matplotlib colormap name for member colors (default: "auto").
        - "auto": Automatically selects tab20 (≤20) or hsv (>20)
        - "tab20", "hsv", "rainbow", etc.: Manual selection
    custom_colors : dict[int, str], optional
        Manual color overrides for specific member IDs.
        Example: {1: "red", 5: "blue", 10: "#00ff00"}
    **kwargs
        Additional arguments passed to ax.plot()

    Returns
    -------
    tuple
        (Figure, Axes) objects

    Examples
    --------
    >>> # Default: auto-selects colormap based on number of members
    >>> fig, ax = plot_ensemble_timeseries(ds, var_name="dye")
    >>> # 18 members → tab20, 30 members → hsv (automatic!)

    >>> # Manual colormap override
    >>> fig, ax = plot_ensemble_timeseries(
    ...     ds, var_name="dye", colormap="rainbow"
    ... )

    >>> # Custom colors for specific members
    >>> fig, ax = plot_ensemble_timeseries(
    ...     ds, var_name="dye",
    ...     custom_colors={1: "red", 5: "blue", 10: "green"}
    ... )
    """
    # Create config with larger font sizes if not provided
    if cfg is None:
        cfg = FvcomPlotConfig(
            figsize=(12, 6),
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

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=cfg.figsize, constrained_layout=True)
    else:
        fig = ax.figure

    # Check if variable exists
    if var_name not in ds:
        raise ValueError(f"Variable '{var_name}' not found in dataset")

    data = ds[var_name]

    # Check if ensemble dimension exists
    if "ensemble" in data.dims:
        n_ensemble = len(data.ensemble)
        # If max_lines is None, plot all ensemble members
        n_plot = n_ensemble if max_lines is None else min(n_ensemble, max_lines)

        # Extract all member IDs for auto-colormap selection
        all_member_ids = []
        for i in range(n_ensemble):
            if hasattr(data.ensemble, "to_index") and isinstance(
                data.ensemble.to_index(), pd.MultiIndex
            ):
                ensemble_val = data.ensemble[i].values.item()
                if isinstance(ensemble_val, tuple):
                    _, member = ensemble_val
                    all_member_ids.append(int(member))

        # Determine total_members for auto-selection
        total_members = max(all_member_ids) if all_member_ids else n_ensemble

        for i in range(n_plot):
            series = data.isel(ensemble=i)

            # Get label and member ID from ensemble coordinates
            member_id = None  # Will be used for color mapping
            if hasattr(data.ensemble, "to_index") and isinstance(
                data.ensemble.to_index(), pd.MultiIndex
            ):
                # MultiIndex case: get the tuple value
                ensemble_val = data.ensemble[i].values.item()
                if isinstance(ensemble_val, tuple):
                    year, member = ensemble_val
                    member_id = int(member)  # Extract member ID for color mapping
                    label = f"Year {year}, Member {member}"
                else:
                    label = f"Ensemble {i}"
            else:
                label = f"Ensemble {i}"

            # Use member-based color mapping for consistency across plot types
            # Auto-selects colormap: tab20 for ≤20 members, hsv for >20
            if member_id is not None:
                color = get_member_color(
                    member_id,
                    colormap=colormap,
                    custom_colors=custom_colors,
                    total_members=total_members,
                )
            else:
                # Fallback to position-based color if member ID not available
                color = cfg.color_cycle[i % len(cfg.color_cycle)]

            # Use matplotlib's plot directly instead of xarray's plot to avoid date formatting conflicts
            ax.plot(
                series.time.values,
                series.values,
                label=label,
                alpha=alpha,
                color=color,
                linewidth=cfg.linewidth_plot,
                **kwargs,
            )

        # Show annotation only if we're limiting the number of lines
        if max_lines is not None and n_ensemble > max_lines:
            ax.text(
                0.98,
                0.02,
                f"(Showing {n_plot} of {n_ensemble} ensemble members)",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=cfg.fontsize_annotation,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
    else:
        # No ensemble dimension - use matplotlib's plot directly
        ax.plot(
            data.time.values,
            data.values,
            label=var_name.upper(),
            alpha=alpha,
            linewidth=cfg.linewidth_plot,
            **kwargs,
        )

    # Apply smart datetime formatting using matplotlib's built-in tools
    if "time" in ds.coords:
        apply_smart_time_ticks(
            ax, fig, minticks=minticks, maxticks=maxticks, rotation=rotation
        )
        ax.tick_params(axis="x", which="major", labelsize=cfg.fontsize_xticks)

    # Labels and title
    ax.set_xlabel("Time", fontsize=cfg.fontsize_xlabel)
    ax.set_ylabel(ylabel or var_name.capitalize(), fontsize=cfg.fontsize_ylabel)

    if title is None:
        title = f"{var_name.upper()} Time Series"
    ax.set_title(title, fontsize=cfg.fontsize_title)

    # Grid
    ax.grid(
        True,
        alpha=cfg.grid_alpha,
        linestyle=cfg.grid_linestyle,
        linewidth=cfg.grid_linewidth,
        color=cfg.grid_color,
    )

    # Legend
    if legend_outside:
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=cfg.fontsize_legend,
            framealpha=0.9,
        )
    else:
        ax.legend(fontsize=cfg.fontsize_legend, framealpha=0.9)

    # Tick parameters
    ax.tick_params(
        axis="both", labelsize=cfg.fontsize_xticks, width=cfg.linewidth_tick_params
    )

    return fig, ax


def plot_ensemble_statistics(
    ds: xr.Dataset,
    var_name: str = "dye",
    fig: Figure | None = None,
    cfg: FvcomPlotConfig | None = None,
    title: str | None = None,
    minticks: int = 3,
    maxticks: int = 7,
    rotation: int = 30,
    **kwargs,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot ensemble mean with standard deviation and coefficient of variation.

    Uses matplotlib's AutoDateLocator and ConciseDateFormatter for clean,
    non-overlapping datetime labels.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the variable to plot
    var_name : str
        Variable name to plot (default: "dye")
    fig : Figure, optional
        Matplotlib figure. If None, creates new figure
    cfg : FvcomPlotConfig, optional
        Plot configuration. If None, uses default with increased font sizes
    title : str, optional
        Main title for the figure
    minticks : int
        Minimum number of ticks on x-axis (default: 3)
    maxticks : int
        Maximum number of ticks on x-axis (default: 7)
    rotation : int
        Rotation angle for x-axis labels in degrees (default: 30)
    **kwargs
        Additional arguments

    Returns
    -------
    tuple
        (Figure, (ax1, ax2)) where ax1 is mean+std plot, ax2 is CV plot
    """
    # Create config with larger font sizes if not provided
    if cfg is None:
        cfg = FvcomPlotConfig(
            figsize=(12, 8),
            fontsize={
                "xticks": 13,
                "yticks": 13,
                "xlabel": 14,
                "ylabel": 14,
                "title": 14,
                "suptitle": 15,
            },
            linewidth={"plot": 2.0},
        )

    # Create figure if needed
    if fig is None:
        fig, axes = plt.subplots(2, 1, figsize=cfg.figsize, constrained_layout=True)
    else:
        axes = fig.subplots(2, 1)

    ax1, ax2 = axes

    # Check if variable and ensemble exist
    if var_name not in ds:
        raise ValueError(f"Variable '{var_name}' not found in dataset")

    data = ds[var_name]

    if "ensemble" not in data.dims:
        raise ValueError("Dataset does not have 'ensemble' dimension")

    # Compute statistics
    data_mean = data.mean(dim="ensemble")
    data_std = data.std(dim="ensemble")

    # Check if we have time coordinate
    has_time = "time" in ds.coords

    # Plot 1: Mean with std envelope
    # Use matplotlib's plot directly instead of xarray's plot to avoid date formatting conflicts
    ax1.plot(
        data_mean.time.values,
        data_mean.values,
        label="Ensemble Mean",
        color="black",
        linewidth=cfg.linewidth_plot,
    )
    ax1.fill_between(
        data_mean.time.values,
        (data_mean - data_std).values,
        (data_mean + data_std).values,
        alpha=0.3,
        label="±1 Std Dev",
        color=cfg.color_cycle[0],
    )
    ax1.set_ylabel(f"{var_name.capitalize()}", fontsize=cfg.fontsize_ylabel)
    ax1.set_xlabel("")  # Remove xlabel from top plot
    ax1.set_title("Ensemble Mean ± Standard Deviation", fontsize=cfg.fontsize_title)
    ax1.legend(fontsize=cfg.fontsize_legend)
    ax1.grid(
        True,
        alpha=cfg.grid_alpha,
        linestyle=cfg.grid_linestyle,
        linewidth=cfg.grid_linewidth,
        color=cfg.grid_color,
    )
    ax1.tick_params(
        axis="both", labelsize=cfg.fontsize_yticks, width=cfg.linewidth_tick_params
    )

    # Plot 2: Coefficient of variation
    cv = (data_std / data_mean).where(data_mean != 0)
    # Use matplotlib's plot directly instead of xarray's plot to avoid date formatting conflicts
    ax2.plot(
        cv.time.values,
        cv.values,
        color=cfg.color_cycle[1],
        linewidth=cfg.linewidth_plot,
    )
    ax2.set_ylabel("Coefficient of Variation", fontsize=cfg.fontsize_ylabel)
    ax2.set_xlabel("Time", fontsize=cfg.fontsize_xlabel)
    ax2.set_title("Ensemble Variability (Std / Mean)", fontsize=cfg.fontsize_title)
    ax2.grid(
        True,
        alpha=cfg.grid_alpha,
        linestyle=cfg.grid_linestyle,
        linewidth=cfg.grid_linewidth,
        color=cfg.grid_color,
    )
    ax2.tick_params(
        axis="both", labelsize=cfg.fontsize_yticks, width=cfg.linewidth_tick_params
    )

    # Apply smart datetime formatting using matplotlib's built-in tools
    if has_time:
        apply_smart_time_ticks(
            ax1, fig, minticks=minticks, maxticks=maxticks, rotation=rotation
        )
        apply_smart_time_ticks(
            ax2, fig, minticks=minticks, maxticks=maxticks, rotation=rotation
        )
        ax1.tick_params(axis="x", which="major", labelsize=cfg.fontsize_xticks)
        ax2.tick_params(axis="x", which="major", labelsize=cfg.fontsize_xticks)

    # Overall title
    # Note: With constrained_layout, don't manually set y position
    # Let the layout engine handle spacing automatically
    if title:
        fig.suptitle(title, fontsize=cfg.fontsize_suptitle)

    return fig, (ax1, ax2)
