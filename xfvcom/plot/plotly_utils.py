"""
Plotly-based interactive plotting utilities for xfvcom.

This module provides functions for creating interactive visualizations
using plotly, particularly useful for time series data comparison and
exploration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def check_plotly_availability():
    """Check if plotly is available and raise informative error if not."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is not installed. Install it with:\n"
            "  pip install plotly\n"
            "or\n"
            "  conda install -c plotly plotly"
        )


def plot_timeseries_comparison(
    data_dict: Dict[str, pd.DataFrame],
    title: str = "Time Series Comparison",
    ylabel: str = "Value",
    subplot_titles: Optional[List[str]] = None,
    height: int = 800,
    colors: Optional[Dict[str, str]] = None,
    reference_time: Optional[pd.Timestamp] = None,
    show_legend: bool = True,
    hovermode: str = "x unified",
) -> go.Figure:
    """
    Create an interactive plotly figure comparing multiple time series.

    This function is particularly useful for comparing original data with
    various extension methods (forward fill, linear, seasonal, etc.).

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary mapping method names to DataFrames with time series data.
        Each DataFrame should have a DatetimeIndex.
        Special key 'original' will be plotted on all subplots if present.
    title : str
        Main title for the figure
    ylabel : str
        Y-axis label (applied to all subplots)
    subplot_titles : List[str], optional
        Titles for each subplot. If None, uses keys from data_dict
    height : int
        Figure height in pixels
    colors : Dict[str, str], optional
        Dictionary mapping method names to colors
    reference_time : pd.Timestamp, optional
        If provided, adds a vertical line at this time (e.g., extension start)
    show_legend : bool
        Whether to show the legend
    hovermode : str
        Plotly hover mode ('x unified', 'x', 'y', 'closest')

    Returns
    -------
    go.Figure
        Plotly figure object

    Examples
    --------
    >>> import pandas as pd
    >>> from xfvcom.plot.plotly_utils import plot_timeseries_comparison
    >>>
    >>> # Create sample data
    >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
    >>> original = pd.DataFrame({'value': np.random.randn(100)}, index=dates)
    >>>
    >>> # Create extensions
    >>> extended_dates = pd.date_range('2020-01-01', periods=200, freq='D')
    >>> ffill = pd.DataFrame({'value': np.concatenate([
    ...     original['value'].values,
    ...     np.full(100, original['value'].iloc[-1])
    ... ])}, index=extended_dates)
    >>>
    >>> # Plot comparison
    >>> fig = plot_timeseries_comparison(
    ...     {'original': original, 'forward_fill': ffill},
    ...     title="River Discharge Extension",
    ...     ylabel="Discharge (m³/s)",
    ...     reference_time=original.index[-1]
    ... )
    >>> fig.show()
    """
    check_plotly_availability()

    # Default colors
    default_colors = {
        "original": "blue",
        "forward_fill": "red",
        "ffill": "red",
        "linear": "green",
        "seasonal": "magenta",
        "interpolated": "orange",
    }

    if colors:
        default_colors.update(colors)

    # Separate original data if present
    original_data = data_dict.pop("original", None)

    # Create subplots
    n_methods = len(data_dict)
    if n_methods == 0:
        raise ValueError("No data to plot (excluding 'original')")

    if subplot_titles is None:
        subplot_titles = [name.replace("_", " ").title() for name in data_dict.keys()]

    # Increase vertical spacing to prevent title overlap
    vertical_spacing = 0.15 if n_methods <= 3 else 0.12 if n_methods <= 5 else 0.1

    fig = make_subplots(
        rows=n_methods,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=vertical_spacing,
    )

    # Plot each method
    for idx, (method_name, data) in enumerate(data_dict.items(), 1):
        # Add original data to each subplot if available
        if original_data is not None:
            # Ensure we're working with a Series or single-column DataFrame
            orig_values = (
                original_data.iloc[:, 0]
                if original_data.shape[1] > 1
                else original_data.squeeze()
            )

            fig.add_trace(
                go.Scatter(
                    x=orig_values.index,
                    y=orig_values.values,
                    mode="lines",
                    name="Original",
                    line=dict(color=default_colors.get("original", "blue"), width=2),
                    showlegend=(idx == 1 and show_legend),
                ),
                row=idx,
                col=1,
            )

        # Add method data
        method_values = data.iloc[:, 0] if data.shape[1] > 1 else data.squeeze()

        fig.add_trace(
            go.Scatter(
                x=method_values.index,
                y=method_values.values,
                mode="lines",
                name=method_name.replace("_", " ").title(),
                line=dict(
                    color=default_colors.get(method_name, "gray"),
                    width=1.5,
                    dash="dash",
                ),
                opacity=0.7,
                showlegend=show_legend,
            ),
            row=idx,
            col=1,
        )

        # Add reference line if provided
        if reference_time is not None:
            fig.add_vline(
                x=reference_time,
                line_dash="dot",
                line_color="red",
                opacity=0.5,
                row=idx,
                col=1,
            )

    # Update layout
    fig.update_xaxes(title_text="Date", row=n_methods, col=1)

    for idx in range(1, n_methods + 1):
        fig.update_yaxes(title_text=ylabel, row=idx, col=1)

    fig.update_layout(
        height=height,
        title_text=f"{title}<br><sub>Hover for values, drag to zoom, double-click to reset</sub>",
        hovermode=hovermode,
        showlegend=show_legend,
    )

    return fig


def plot_timeseries_multi_variable(
    data: pd.DataFrame,
    variables: List[str],
    title: str = "Multi-Variable Time Series",
    subplot_titles: Optional[List[str]] = None,
    height: int = 200,  # per subplot
    colors: Optional[List[str]] = None,
    show_legend: bool = True,
) -> go.Figure:
    """
    Create a multi-panel plot for different variables from the same DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with DatetimeIndex and multiple columns
    variables : List[str]
        List of column names to plot
    title : str
        Main title for the figure
    subplot_titles : List[str], optional
        Titles for each subplot
    height : int
        Height per subplot (total height = height * n_variables)
    colors : List[str], optional
        List of colors for each variable
    show_legend : bool
        Whether to show legend

    Returns
    -------
    go.Figure
        Plotly figure object
    """
    check_plotly_availability()

    n_vars = len(variables)

    if subplot_titles is None:
        subplot_titles = [var.replace("_", " ").title() for var in variables]

    if colors is None:
        colors = ["blue", "green", "red", "orange", "purple", "brown"]

    fig = make_subplots(
        rows=n_vars,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1 / n_vars if n_vars > 1 else 0.1,
    )

    for idx, var in enumerate(variables, 1):
        if var in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[var],
                    mode="lines",
                    name=var.replace("_", " ").title(),
                    line=dict(color=colors[idx - 1 % len(colors)], width=2),
                    showlegend=show_legend,
                ),
                row=idx,
                col=1,
            )

    # Update layout
    fig.update_xaxes(title_text="Date", row=n_vars, col=1)

    fig.update_layout(
        height=height * n_vars,
        title_text=title,
        hovermode="x unified",
        showlegend=show_legend,
    )

    return fig


def print_plotly_instructions():
    """Print instructions for using plotly interactive features."""
    print("\n" + "=" * 60)
    print("Plotly Interactive Features:")
    print("=" * 60)
    print("  📍 Mouse Interactions:")
    print("    • Hover: View exact values at cursor position")
    print("    • Click & Drag: Zoom into selected area")
    print("    • Double-click: Reset view to original zoom")
    print("    • Scroll: Zoom in/out (over plot area)")
    print("")
    print("  🔧 Toolbar Controls (top right):")
    print("    • Camera icon: Download plot as PNG")
    print("    • Zoom: Click and drag to zoom")
    print("    • Pan: Move around the plot")
    print("    • Box Select: Select rectangular region")
    print("    • Lasso Select: Freeform selection")
    print("    • Zoom in/out: Fixed increment zoom")
    print("    • Autoscale: Fit all data in view")
    print("    • Reset axes: Return to default view")
    print("")
    print("  📊 Legend Interactions:")
    print("    • Single-click: Toggle series visibility")
    print("    • Double-click: Isolate/show all series")


def create_river_extension_plot(
    original_data: Dict,
    extend_to: Union[str, pd.Timestamp],
    river_idx: int = 0,
    methods: List[str] = None,
    **kwargs,
) -> go.Figure:
    """
    Create a comparison plot for river time series extensions.

    This is a convenience function specifically for river data extension visualization.

    Parameters
    ----------
    original_data : Dict
        Dictionary from read_fvcom_river_nc containing river data
    extend_to : str or pd.Timestamp
        Target end datetime for extensions
    river_idx : int
        Index of river to plot (default: 0)
    methods : List[str]
        Extension methods to use. Default: ['ffill', 'linear', 'seasonal']
    **kwargs
        Additional arguments passed to plot_timeseries_comparison

    Returns
    -------
    go.Figure
        Plotly figure object
    """
    from xfvcom.utils.timeseries_utils import (
        extend_timeseries_ffill,
        extend_timeseries_linear,
        extend_timeseries_seasonal,
    )

    if methods is None:
        methods = ["ffill", "linear", "seasonal"]

    # Get river name
    river_name = (
        original_data["river_names"][river_idx]
        if "river_names" in original_data
        else f"River_{river_idx+1}"
    )

    # Prepare data dictionary
    data_dict = {}

    # Add original data
    if "river_flux" in original_data:
        data_dict["original"] = original_data["river_flux"].iloc[:, [river_idx]]

    # Generate extensions
    for method in methods:
        if method == "ffill" or method == "forward_fill":
            data_dict["Forward Fill"] = extend_timeseries_ffill(
                original_data["river_flux"].iloc[:, [river_idx]], extend_to
            )
        elif method == "linear":
            data_dict["Linear"] = extend_timeseries_linear(
                original_data["river_flux"].iloc[:, [river_idx]],
                extend_to,
                lookback_periods=kwargs.pop("lookback_periods", 60),
            )
        elif method == "seasonal":
            data_dict["Seasonal"] = extend_timeseries_seasonal(
                original_data["river_flux"].iloc[:, [river_idx]],
                extend_to,
                period=kwargs.pop("period", "1Y"),
            )

    # Default kwargs
    default_kwargs = {
        "title": f"{river_name} - Extension Methods Comparison",
        "ylabel": "Discharge (m³/s)",
        "reference_time": original_data["datetime"][-1],
        "subplot_titles": [
            f"{river_name} - {method}"
            for method in data_dict.keys()
            if method != "original"
        ],
    }
    default_kwargs.update(kwargs)

    return plot_timeseries_comparison(data_dict, **default_kwargs)
