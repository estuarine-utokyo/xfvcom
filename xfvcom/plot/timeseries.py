"""Time series plotting utilities for xfvcom.

This module provides utilities for creating publication-quality time series plots
with intelligent time axis formatting and consistent styling using FvcomPlotConfig.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter, DateFormatter
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    import xarray as xr

from .config import FvcomPlotConfig


def apply_smart_time_ticks(ax, fig=None, minticks=3, maxticks=7, rotation=30):
    """Apply smart datetime tick formatting using matplotlib's built-in tools.

    This is the proper way to handle datetime axes without overlap issues.
    Uses AutoDateLocator + ConciseDateFormatter with manual label rotation.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object
    fig : Figure, optional
        Matplotlib figure. If None, gets from ax.figure
    minticks : int
        Minimum number of ticks (default: 3)
    maxticks : int
        Maximum number of ticks (default: 7)
    rotation : int
        Label rotation angle in degrees (default: 30)

    Returns
    -------
    Axes
        The modified axes object
    """
    if fig is None:
        fig = ax.figure

    # Use matplotlib's intelligent date locator
    locator = AutoDateLocator(minticks=minticks, maxticks=maxticks)

    # Use ConciseDateFormatter for clean, non-overlapping labels
    formatter = ConciseDateFormatter(locator)

    # Apply to axis
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Rotate labels manually (compatible with constrained_layout)
    # Note: We don't use fig.autofmt_xdate() because it's incompatible with constrained_layout
    if rotation != 0:
        for label in ax.get_xticklabels():
            label.set_rotation(rotation)
            label.set_ha("right")

    return ax


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
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plot time series for ensemble data with automatic datetime formatting.

    Uses matplotlib's AutoDateLocator and ConciseDateFormatter for clean,
    non-overlapping datetime labels.

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
    **kwargs
        Additional arguments passed to ax.plot()

    Returns
    -------
    tuple
        (Figure, Axes) objects
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

        for i in range(n_plot):
            series = data.isel(ensemble=i)

            # Get label from ensemble coordinates
            if hasattr(data.ensemble, "to_index") and isinstance(
                data.ensemble.to_index(), pd.MultiIndex
            ):
                # MultiIndex case: get the tuple value
                ensemble_val = data.ensemble[i].values.item()
                if isinstance(ensemble_val, tuple):
                    year, member = ensemble_val
                    label = f"Year {year}, Member {member}"
                else:
                    label = f"Ensemble {i}"
            else:
                label = f"Ensemble {i}"

            # Use color cycle from config
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
