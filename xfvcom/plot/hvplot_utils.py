"""
hvPlot/Bokeh-based interactive plotting utilities for xfvcom.

This module provides the InteractivePlotter class and functions for creating
interactive visualizations using hvPlot/Bokeh. Useful for large time series
data exploration with pan/zoom capabilities and standalone HTML export.

hvPlot is an optional dependency. Install with:
    conda install -c conda-forge hvplot bokeh holoviews
or:
    pip install xfvcom[interactive]
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import xarray as xr

try:
    import holoviews as hv
    import hvplot.xarray  # noqa: F401 - registers hvplot accessor

    HVPLOT_AVAILABLE = True
except ImportError:
    HVPLOT_AVAILABLE = False

if TYPE_CHECKING:
    import holoviews as hv

from .variable_meta import get_label

# =============================================================================
# Default settings for interactive plots
# =============================================================================
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 250
DEFAULT_FONTSIZE = "12pt"


def check_hvplot_availability() -> None:
    """Check if hvplot is available and raise informative error if not."""
    if not HVPLOT_AVAILABLE:
        raise ImportError(
            "hvplot is not installed. Install with:\n"
            "  conda install -c conda-forge hvplot bokeh holoviews\n"
            "or:\n"
            "  pip install xfvcom[interactive]"
        )


class InteractivePlotter:
    """Interactive plotter for xarray data using hvPlot/Bokeh.

    Creates publication-quality interactive plots with:
    - Proper axis labels with LaTeX-style units
    - Pan/zoom/reset tools
    - Standalone HTML export

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to plot from
    width : int
        Default plot width in pixels (default: 800)
    height : int
        Default plot height in pixels (default: 250)

    Examples
    --------
    >>> import xarray as xr
    >>> from xfvcom.plot import InteractivePlotter
    >>>
    >>> ds = xr.open_dataset("weather.nc")
    >>> plotter = InteractivePlotter(ds.isel(node=0))
    >>>
    >>> # Create and display plot
    >>> plot = plotter.timeseries("air_temperature")
    >>> plot  # Display in Jupyter
    >>>
    >>> # Save to HTML
    >>> plotter.timeseries("air_temperature", output="temp.html")
    """

    def __init__(
        self,
        ds: xr.Dataset,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ) -> None:
        check_hvplot_availability()
        self.ds = ds
        self.width = width
        self.height = height
        self._default_tools = ["pan", "wheel_zoom", "box_zoom", "reset", "save"]

    def _get_time_coord(self, da: xr.DataArray) -> str:
        """Find time coordinate name in DataArray."""
        for coord_name in ["time", "Time", "datetime"]:
            if coord_name in da.coords:
                return coord_name
        raise ValueError("No time coordinate found in data")

    def _get_label(self, da: xr.DataArray, var_name: str) -> str:
        """Generate y-axis label using variable metadata."""
        # Use plain text for Bokeh (no LaTeX support)
        return get_label(da, var_name, use_latex=False)

    def _save_plot(self, plot: hv.Element, output: str | Path) -> None:
        """Save plot to HTML file."""
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        hv.save(plot, output_path, backend="bokeh")
        print(f"Saved: {output_path}")

    def timeseries(
        self,
        var_name: str,
        title: str | None = None,
        ylabel: str | None = None,
        xlabel: str = "",
        color: str = "#1f77b4",
        line_width: float = 1.0,
        width: int | None = None,
        height: int | None = None,
        grid: bool = True,
        hover: bool = True,
        output: str | Path | None = None,
        **kwargs: Any,
    ) -> hv.Element:
        """Create an interactive time series plot.

        Parameters
        ----------
        var_name : str
            Variable name to plot
        title : str, optional
            Plot title. If None, no title
        ylabel : str, optional
            Y-axis label. If None, auto-generates from variable metadata
        xlabel : str
            X-axis label (default: "" - datetime ticks are self-explanatory)
        color : str
            Line color (default: "#1f77b4")
        line_width : float
            Line width (default: 1.0)
        width : int, optional
            Plot width. If None, uses instance default
        height : int, optional
            Plot height. If None, uses instance default
        grid : bool
            Whether to show grid (default: True)
        hover : bool
            Whether to enable hover tooltips (default: True)
        output : str or Path, optional
            Path to save as HTML file
        **kwargs
            Additional arguments passed to hvplot

        Returns
        -------
        hv.Element
            HoloViews plot object
        """
        if var_name not in self.ds:
            raise ValueError(
                f"Variable '{var_name}' not found. Available: {list(self.ds.data_vars)}"
            )

        da = self.ds[var_name]
        time_coord = self._get_time_coord(da)

        if ylabel is None:
            ylabel = self._get_label(da, var_name)

        # Build hvplot options
        plot_opts: dict[str, Any] = {
            "x": time_coord,
            "ylabel": ylabel,
            "width": width or self.width,
            "height": height or self.height,
            "color": color,
            "line_width": line_width,
            "grid": grid,
            "hover": hover,
            "tools": self._default_tools,
            "fontsize": {
                "labels": DEFAULT_FONTSIZE,
                "ticks": DEFAULT_FONTSIZE,
                "title": DEFAULT_FONTSIZE,
            },
        }

        # Only add title if provided
        if title is not None:
            plot_opts["title"] = title

        # Only add xlabel if provided
        if xlabel:
            plot_opts["xlabel"] = xlabel

        plot = da.hvplot.line(**plot_opts, **kwargs)

        if output is not None:
            self._save_plot(plot, output)

        return plot

    def timeseries_multi(
        self,
        var_names: list[str],
        title: str | None = None,
        width: int | None = None,
        height: int = DEFAULT_HEIGHT,
        output: str | Path | None = None,
        **kwargs: Any,
    ) -> hv.Layout:
        """Create multiple stacked time series plots.

        Parameters
        ----------
        var_names : list[str]
            List of variable names to plot
        title : str, optional
            Overall title for the layout
        width : int, optional
            Plot width. If None, uses instance default
        height : int
            Height per subplot (default: 250)
        output : str or Path, optional
            Path to save as HTML file
        **kwargs
            Additional arguments passed to hvplot

        Returns
        -------
        hv.Layout
            HoloViews layout object
        """
        plots = []
        for var_name in var_names:
            if var_name not in self.ds:
                print(f"Warning: Variable '{var_name}' not found, skipping")
                continue

            da = self.ds[var_name]
            time_coord = self._get_time_coord(da)
            ylabel = self._get_label(da, var_name)

            plot = da.hvplot.line(
                x=time_coord,
                ylabel=ylabel,
                width=width or self.width,
                height=height,
                tools=self._default_tools,
                fontsize={
                    "labels": DEFAULT_FONTSIZE,
                    "ticks": DEFAULT_FONTSIZE,
                },
                **kwargs,
            )
            plots.append(plot)

        if not plots:
            raise ValueError("No valid variables to plot")

        layout = hv.Layout(plots).cols(1)
        if title:
            layout = layout.opts(title=title)

        if output is not None:
            self._save_plot(layout, output)

        return layout


# Standalone function for simple use cases
def hvplot_timeseries(
    data: xr.DataArray | xr.Dataset,
    var_name: str | None = None,
    output: str | Path | None = None,
    **kwargs: Any,
) -> hv.Element:
    """Create an interactive time series plot (convenience function).

    For more control, use InteractivePlotter class.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Data to plot
    var_name : str, optional
        Variable name (required if data is Dataset)
    output : str or Path, optional
        Path to save as HTML file
    **kwargs
        Arguments passed to InteractivePlotter.timeseries()

    Returns
    -------
    hv.Element
        HoloViews plot object
    """
    check_hvplot_availability()

    if isinstance(data, xr.Dataset):
        if var_name is None:
            raise ValueError("var_name required when data is Dataset")
        ds = data
    else:
        var_name = data.name or "value"
        ds = data.to_dataset(name=var_name)

    plotter = InteractivePlotter(ds)
    return plotter.timeseries(var_name, output=output, **kwargs)
