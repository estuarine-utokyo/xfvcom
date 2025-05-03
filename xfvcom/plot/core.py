# xfvcom.py: A Python module for loading, analyzing, and plotting FVCOM model output data in xfvcom package.
# Author: Jun Sasaki
from __future__ import annotations

import inspect
import logging
import warnings
from collections.abc import Hashable, Sequence
from typing import Any

import cartopy.crs as ccrs
import matplotlib.axes as maxes
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pyproj
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.colorbar import Colorbar
from matplotlib.colors import BoundaryNorm
from matplotlib.dates import DateFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogFormatter, LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import KDTree

from ..analysis import FvcomAnalyzer
from ..decorators import precedence
from ..io import FvcomDataLoader
from ..plot_options import FvcomPlotOptions
from ..utils.helpers import PlotHelperMixin, pick_first
from .config import FvcomPlotConfig
from .utils import add_colorbar

"""
Suppress Shapely RuntimeWarning when linestrings encounter NaN coords.
Cartopy may generate such NaNs internally (e.g. at dateline).
"""
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in linestrings",
    category=RuntimeWarning,
)

# ------------------------------------------------------------------
# Module-level logger (keeps external behaviour unchanged)
# ------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# If user hasn’t configured logging, fall back to console INFO output
# (→ behaviour almost identical to former `print()` calls).
# ------------------------------------------------------------------
if not logger.hasHandlers():
    _h = logging.StreamHandler()  # stdout/stderr 自動選択
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

_TRICF_SIG = set(inspect.signature(maxes.Axes.tricontourf).parameters)


class FvcomPlotter(PlotHelperMixin):
    """
    Creates plots from FVCOM datasets.
    """

    def __init__(self, dataset, plot_config):
        """
        Initialize the FvcomPlotter instance.

        Parameters:
        - dataset: An xarray.Dataset object containing FVCOM model output or input.
        - plot_config: An instance of FvcomPlotConfig with plot configuration settings.
        """
        self.ds = dataset
        self.cfg = plot_config

    def plot_timeseries(
        self,
        var_name,
        index,
        log=False,
        k=None,
        start=None,
        end=None,
        rolling_window=None,
        ax=None,
        save_path=None,
        **kwargs,
    ):
        """
        Plot a time series for a specified variable at a given node or element index.

        Parameters:
        - var_name: Name of the variable to plot.
        - index: Index of the `node` or `nele` to plot.
        - k: Vertical layer index for 3D variables (optional).
        - dim: Dimension to use ('node' or 'nele' or nobc).
        - start: Start time for the plot (datetime or string).
        - end: End time for the plot (datetime or string).
        - rolling_window: Size of the rolling window for moving average (optional).
        - ax: matplotlib axis object. If None, a new axis will be created.
        - save_path: Path to save the plot as an image (optional).
        - **kwargs: Additional arguments for customization (e.g., dpi, figsize).
        """
        if var_name not in self.ds:
            msg = f"Error: the variable '{var_name}' is not found in the dataset."
            logger.error(msg)
            # print(msg)          # keep previous console output
            return None

        # Validate the dimension
        variable_dims = self.ds[var_name].dims
        if "node" in variable_dims:
            dim = "node"
        elif "nele" in variable_dims:
            dim = "nele"
        elif "nobc" in variable_dims:
            dim = "nobc"
        else:
            raise ValueError(
                f"Variable {var_name} does not have 'node' or 'nele' as a dimension."
            )

        variable_dims = self.ds[var_name].dims
        if "siglay" in variable_dims:
            dimk = "siglay"
        elif "siglev" in variable_dims:
            dimk = "siglev"
        else:
            if k is not None:
                raise ValueError(
                    f"Variable {var_name} does not have 'siglay' or 'siglev' as a dimension."
                )

        # Select the data
        if k is not None:
            data = self.ds[var_name].isel({dim: index, dimk: k})
        else:
            data = self.ds[var_name].isel({dim: index})

        # Apply rolling mean if specified
        if rolling_window:
            data = data.rolling(time=rolling_window, center=True).mean()

        # Time range filtering
        time = self.ds["time"]
        if start:
            start = np.datetime64(start)
        if end:
            end = np.datetime64(end)
        time_mask = (time >= start) & (time <= end) if start and end else slice(None)
        data = data.isel(time=time_mask)
        time = time[time_mask]

        if log:  # Check if log scale is requested
            if data.min() <= 0:

                logger.warning(
                    "Logarithmic scale cannot be used with non-positive values; switching to linear scale."
                )
                log = False

        # If no axis is provided, create a new one
        if ax is None:
            figsize = kwargs.get("figsize", self.cfg.figsize)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure  # Get the figure from the provided axis
        # Plotting
        color = kwargs.pop("color", self.cfg.plot_color)
        if k is not None:
            label = f"{var_name} ({dim}={index}, {dimk}={k})"
        else:
            label = f"{var_name} ({dim}={index})"
        ax.plot(time, data, label=label, color=color, **kwargs)
        if log:  # Set log scale if specified
            ax.set_yscale("log")

        # Formatting
        rolling_text = (
            f" with {rolling_window}-hour Rolling Mean" if rolling_window else ""
        )
        if k is not None:
            title = (
                f"Time Series of {var_name} ({dim}={index}, {dimk}={k}){rolling_text}"
            )
        else:
            title = f"Time Series of {var_name} ({dim}={index}){rolling_text}"
        ax.set_title(title, fontsize=self.cfg.fontsize["title"])
        ax.set_xlabel("Time", fontsize=self.cfg.fontsize["xlabel"])
        ax.set_ylabel(var_name, fontsize=self.cfg.fontsize["ylabel"])
        date_format = kwargs.get("date_format", self.cfg.date_format)
        ax.xaxis.set_major_formatter(DateFormatter(date_format))
        fig.autofmt_xdate()
        ax.grid(True)
        ax.legend()

        # Save or show the plot
        if save_path:
            dpi = kwargs.get("dpi", self.cfg.dpi)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return ax

    def hvplot_time_series(self, var, siglay=None, node=None, **kwargs):
        """
        Plot a time series for the specified variable.
        """
        da = self.ds[var]
        return da[:, siglay, node].hvplot(
            x="time",
            width=self.cfg.width,
            height=self.cfg.height,
            fontsize=self.cfg.fontsize,
            **kwargs,
        )

    def ts_river(
        self,
        da: xr.DataArray = None,
        varname: str = None,
        river_index: int = None,
        rolling_window: int = None,
        title=None,
        verbose=False,
        ax=None,
        **kwargs,
    ):
        """
        Plot a river variable time-series by delegating to self.ts_plot().

        Parameters
        ----------
        da : xr.DataArray, optional
            DataArray with 'time' and 'rivers' dimension.
        varname : str
            Variable name with 'time' and 'rivers' dimension.
        river_index : int
            Index along the 'rivers' dimension.
        rolling_window : int | None
            Centered rolling-mean window (hours).
        title : str | None
            Title for the plot. If None, a default title is generated.
        verbose : bool
            If True, prints the river name and index.
        ax : matplotlib.axes.Axes | None
            Pre-created axis. A new fig/ax is created if None.
        **kwargs
            Extra keyword arguments forwarded to self.ts_plot().

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            The figure and axis containing the plot.
        """

        # 1) Validate variable
        if da is None:
            # da is extracted from self.ds[varname]
            if varname is None:
                raise ValueError("Either 'da' or 'varname' must be provided.")
            da = self.ds[varname]
        elif varname is not None:
            raise ValueError("Only one of 'da' or 'varname' should be provided.")

        # 2) Extract 1-D DataArray for selected river
        da = da.isel(rivers=river_index)

        # 3) Resolve river name for labels
        if "river_names" in self.ds:
            raw = self.ds["river_names"].isel(rivers=river_index).values
            name = raw.item() if isinstance(raw, np.ndarray) else raw
            if isinstance(name, (bytes, bytearray)):
                name = name.decode("utf-8")
            river_name = str(name).strip()
        else:
            river_name = f"river {river_index}"

        # 4) Default title / xlabel / ylabel
        if isinstance(title, str):
            title = title.strip()  # explicit string
        elif title:  # Truthy but not str (e.g. True, 1)
            roll_txt = (
                f" with {rolling_window}-hour Rolling Mean" if rolling_window else ""
            )
            title = f"Time Series of {varname} for {river_name} (river={river_index}){roll_txt}"
        else:  # None, False, "", 0 …
            title = ""  # no label

        xlabel = kwargs.pop("xlabel", None)
        ylabel = kwargs.pop("ylabel", True)
        if verbose:
            default_label = f"{river_name} (index={river_index})"
        else:
            default_label = f"{river_name}"
        label = kwargs.pop("label", default_label)

        # 5) Ensure we have an axis (create if None)
        if ax is None:
            fig, ax = plt.subplots(figsize=self.cfg.figsize, dpi=self.cfg.dpi)
        else:
            fig = ax.figure

        # 6) Delegate to ts_plot (ax provided so it draws on same axis)
        fig, ax = self.ts_plot(
            da=da,
            rolling_window=rolling_window,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            label=label,
            ax=ax,
            **kwargs,
        )

        return fig, ax

    def ts_vector(
        self,
        da_x: xr.DataArray = None,
        da_y: xr.DataArray = None,
        varname_x: str = None,
        varname_y: str = None,
        index: int = None,
        xlabel: str = "",
        ylabel: str = "Wind speed (m/s)",
        title: str = None,
        rolling_window: int = None,
        show_legend: bool = True,
        with_magnitude: bool = True,
        show_vec_legend: bool = True,
        vec_legend_speed: float = 10,
        vec_legend_loc: tuple = (0.85, 0.1),
        ax=None,
        opts: FvcomPlotOptions | None = None,
        **kwargs,
    ):
        """
        Plot time-series of vector components (u, v) and optionally their magnitude.
        Accept either (da_x, da_y) or (varname_x, varname_y) exclusively.
        Passing the DataArray objects directly is effective when you need to avoid performance degradation in parallel processing.

        Parameters:
        -----------
        da_x (2D/1D DataArray), optional: DataArray for the x-component (e.g., ds.uwind_speed).
        da_y (2D/1D DataArray), optional: DataArray for the y-component (e.g., ds.vwind_speed).
        varname_x (str), optional: Dataset variable name for the x-component (e.g., 'uwind_speed').
        varname_y (str), optional: Dataset variable name for the y-component (e.g., 'vwind_speed').
        index (int), optional: Index of the node/nele to plot (for 2D DataArrays).
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        rolling_window (int): Size of the rolling window for moving average.
        show_legend (bool): If True, show the legend.
        with_magnitude (bool): If True, plot the magnitude of the vector.
        show_vec_legend (bool): If True, show the vector legend.
        vec_legend_speed (float): Speed for the vector legend.
        vec_legend_loc (tuple): Location of the vector legend in axes coordinates.
        ax: matplotlib axis object. If None, a new figure and axis will be created.
        **kwargs: Additional keyword arguments for customization.

        Returns:
        --------
        fig: The matplotlib figure object.
        ax: The matplotlib axis object.
        """

        if opts is None:
            opts = FvcomPlotOptions.from_kwargs(**kwargs)
        else:
            opts.extra.update(kwargs)

        # ------------------------------------------------------------
        # 1. Validate the input combination
        # ------------------------------------------------------------
        has_da = (da_x is not None) or (da_y is not None)
        has_varname = (varname_x is not None) or (varname_y is not None)

        # Both groups specified or neither → error
        if (has_da and has_varname) or (not has_da and not has_varname):
            raise ValueError(
                "Specify either (da_x & da_y) *or* (varname_x & varname_y), but not both or neither."
            )

        # Only one of each pair given → error
        if (da_x is None) != (da_y is None):
            raise ValueError("Both da_x and da_y must be supplied together.")
        if (varname_x is None) != (varname_y is None):
            raise ValueError("Both varname_x and varname_y must be supplied together.")

        # ------------------------------------------------------------
        # 2. Convert variable names to DataArrays when necessary
        # ------------------------------------------------------------
        if has_varname:
            if varname_x not in self.ds or varname_y not in self.ds:
                raise ValueError(
                    f"Variables '{varname_x}' or '{varname_y}' not found in the dataset."
                )
            da_x = self.ds[varname_x]
            da_y = self.ds[varname_y]

        # ---- after this point da_x / da_y are definitely DataArray ----
        assert (
            da_x is not None and da_y is not None
        ), "da_x and da_y must be DataArray after validation"

        # ------------------------------------------------------------
        # 3. Slice by index or use full 1-D series
        # ------------------------------------------------------------
        if da_x.ndim == da_y.ndim == 2:
            if index is None:
                raise ValueError("Index must be provided for 2D DataArrays.")
            u = da_x[:, index]
            v = da_y[:, index]
            index_name = da_x.dims[1]
        elif da_x.ndim == da_y.ndim == 1:
            u = da_x
            v = da_y
            index_name = None
        else:
            raise ValueError(
                f"Both da_x and da_y must be 2‑D or 1-D; "
                f"got shapes {da_x.shape} and {da_y.shape}."
            )

        # Apply rolling mean if specified
        if rolling_window:
            # u = u.rolling(time=rolling_window, center=True).mean().dropna(dim="time")
            # v = v.rolling(time=rolling_window, center=True).mean().dropna(dim="time")
            # Ensure time alignment after rolling and dropna
            uv = xr.Dataset({"u": u, "v": v})
            uv = uv.rolling(time=rolling_window, center=True).mean().dropna(dim="time")
            u = uv["u"]
            v = uv["v"]

        time = u["time"]

        # Compute wind speed magnitude
        speed = np.sqrt(u**2 + v**2)

        # Adjust scale for quiver plot
        max_speed = float(np.max(speed)) if speed.size else 0.0
        # scale_factor = max_speed / 10  # Adjust to fit arrows within the plot

        figsize = kwargs.get("figsize", self.cfg.figsize)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=self.cfg.dpi)
        else:
            fig = ax.figure
        # Plot wind speed magnitude
        if with_magnitude:
            ax.plot(
                time,
                speed,
                label="Wind Speed (m/s)",
                color=self.cfg.plot_color,
                alpha=0.5,
            )

        # Quiver plot for wind vectors
        # Prepare quiver keyword arguments
        #     Allow user to pass scale="auto" (default) for automatic scaling
        scale_kwarg = kwargs.get("scale", "auto")
        if scale_kwarg == "auto":
            # empirical: make the longest arrow fill ~10 % of y‑axis height
            scale = 1.0
        else:
            scale = scale_kwarg

        quiver_opts = dict(
            angles=kwargs.get("angles", self.cfg.arrow_angles),
            headlength=kwargs.get("headlength", self.cfg.arrow_headlength),
            headwidth=kwargs.get("headwidth", self.cfg.arrow_headwidth),
            headaxislength=kwargs.get("headaxislength", self.cfg.arrow_headaxislength),
            width=kwargs.get("width", self.cfg.arrow_width),
            color=kwargs.get("color", self.cfg.arrow_color),
            alpha=kwargs.get("alpha", self.cfg.arrow_alpha),
            scale_units="y",
            scale=scale,
        )

        Q = ax.quiver(time, np.zeros(len(time)), u, v, **quiver_opts)

        # Add quiverkey
        if show_vec_legend:
            ref_speed = (
                (max_speed * 0.3) if vec_legend_speed is None else vec_legend_speed
            )
            ax.quiverkey(
                Q,
                *vec_legend_loc,
                ref_speed,
                f"{ref_speed:.1f} m/s",
                labelpos="E",
                coordinates="axes",
                color=quiver_opts["color"],
                alpha=quiver_opts["alpha"],
                fontproperties={"size": self.cfg.fontsize_legend},
            )

        # Format y-axis to accommodate negative and positive values
        max_v: float = np.max(np.abs(max_speed))
        ax.set_ylim(-max_v, max_v)

        # Format axes
        rolling_text = (
            f" with {rolling_window}-hour rolling mean" if rolling_window else ""
        )
        index_text = f"({index_name}={index})" if index_name is not None else ""
        if title is None:
            title = f"Wind vector time series {index_text}{rolling_text}"
        ax.set_title(title, fontsize=self.cfg.fontsize["title"])
        ax.set_xlabel(xlabel, fontsize=self.cfg.fontsize["xlabel"])
        ax.set_ylabel(ylabel, fontsize=self.cfg.fontsize["ylabel"])
        date_format = kwargs.get("date_format", self.cfg.date_format)
        ax.xaxis.set_major_formatter(DateFormatter(date_format))
        fig.autofmt_xdate()

        # Add y=0 line
        frame_color = ax.spines[
            "left"
        ].get_edgecolor()  # Retrieve the color of the left spine
        ax.axhline(
            0.0,  # y = 0
            color=frame_color,  # Use the same color as the frame
            linewidth=self.cfg.linewidth_axes * 0.3,  # Make it thinner
            zorder=5,  # Ensure it is above the quiver arrows and other elements
        )

        if show_legend and with_magnitude:
            ax.legend(fontsize=self.cfg.fontsize["legend"])

        return fig, ax

    def ts_contourf_z(
        self,
        da: xr.DataArray,
        index: int = None,
        xlim: tuple = None,
        ylim: tuple = None,
        xlabel: str = "Time",
        ylabel: str = "Depth (m)",
        title: str = None,
        rolling_window: int | None = None,
        ax=None,
        cmap=None,
        label=None,
        contourf_kwargs: dict = None,
        colorbar_kwargs: dict = None,
        plot_surface: bool = False,
        surface_kwargs: dict | None = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes, Colorbar]:
        """
        Plot a 2D time-series contour (time vs depth) for the specified variable.
        This method is specialized for z-coordinate (depth) data and does not support sigma coordinates.
        Parameters:
        - da (xarray.DataArray): The DataArray to plot.
        - index (int): Index of the node/nele for spatial dimension (required if data has 'node' or 'nele' dim).
        - xlim (tuple): (start_time, end_time) for the x-axis time range.
        - ylim (tuple): Depth range for the y-axis (e.g., (0, 100)) in meters.
        - xlabel, ylabel (str): Axis labels for time and depth.
        - title (str): Plot title. If None, a default title is generated.
        - rolling_window (int | None): Window size (in time steps) for centered rolling mean smoothing.
        - ax (matplotlib.axes.Axes): Axis to plot on. If None, a new figure and axis are created.
        - cmap: Colormap for the contour. Uses default from config if None.
        - label (str): Label for the colorbar (overrides variable long_name/units if provided).
        - contourf_kwargs (dict): Additional keyword arguments for contourf.
        - colorbar_kwargs (dict): Additional keyword arguments for colorbar.
        - plot_surface (bool): If True, plot the surface elevation on top of the contour.
        - surface_kwargs (dict): Additional keyword arguments for surface plot.
        - **kwargs: Additional contourf keyword arguments (levels, etc.).
        Returns:
        - (fig, ax, cbar): Figure, Axes, and Colorbar objects for the plot.
        """
        # 0. Common metadata ----------------------------------------------
        long_name = da.attrs.get("long_name", da.name)

        # 1. Verify da has the required dimensions
        if "time" not in da.dims:
            raise ValueError(f"DataArray must have 'time' dimension, got {da.dims}")

        # 2. Handle spatial dimension (node/nele) – require index if present
        spatial_dim = next((d for d in ("node", "nele") if d in da.dims), None)
        if spatial_dim:
            if index is None:
                raise ValueError(
                    f"Index must be provided for spatial dimension '{spatial_dim}'."
                )
            da = da.isel({spatial_dim: index})
        # elif index is not None:
        #    raise ValueError(f"No spatial dimension in '{var_name}', but index was provided.")
        elif index is not None:
            raise ValueError(
                f"No spatial dimension in '{da.name}', but index was provided."
            )

        # 3. Verify vertical (sigma) dimension is present (required for depth plot)
        vertical_dim = (
            "siglay"
            if "siglay" in da.dims
            else ("siglev" if "siglev" in da.dims else None)
        )
        # if vertical_dim is None:
        #    raise ValueError(f"Variable '{var_name}' has no sigma layer dimension ('siglay' or 'siglev'), cannot plot depth profile.")
        if vertical_dim is None:
            raise ValueError(
                f"Variable '{da.name}' has no sigma layer dimension "
                "('siglay' or 'siglev'); cannot plot depth profile."
            )

        # 4. Apply rolling mean on time axis if specified
        da = self._apply_rolling(
            da, rolling_window
        )  # uses centered rolling mean&#8203;:contentReference[oaicite:0]{index=0}

        # 5. Filter data by time range (xlim) if provided
        da = self._apply_time_filter(
            da, xlim
        )  # uses start/end from xlim to slice time&#8203;:contentReference[oaicite:1]{index=1}

        # 6. Prepare depth (z) values for the selected location and times
        #    We assume the dataset contains 'z' (depth) with same dims (time, vertical, spatial)
        if spatial_dim:
            z_da = self.ds["z"].isel({spatial_dim: index})
        else:
            z_da = self.ds["z"]
        z_da = self._apply_time_filter(z_da, xlim)
        # Align dimensions ordering for consistent (time, vertical) shape
        z_da = z_da.transpose("time", vertical_dim)
        da = da.transpose("time", vertical_dim)
        # Assign depth values as a 2D coordinate for the DataArray (for plotting)
        da.coords["Depth"] = (("time", vertical_dim), z_da.values)

        # 7. Create figure and axis if not provided
        if ax is None:
            fig = plt.figure(figsize=self.cfg.figsize, dpi=self.cfg.dpi)
            ax = fig.add_subplot(1, 1, 1)
        else:
            fig = ax.figure

        # 8. Determine contourf parameters (levels, cmap, etc.), merging **kwargs
        #    and using defaults from config when not specified
        if cmap is not None:
            kwargs["cmap"] = cmap
        merged_cf_kwargs, levels, cmap_used, vmin, vmax, extend = (
            self._prepare_contourf_args(da, None, kwargs)
        )  # unify contour args&#8203;:contentReference[oaicite:2]{index=2}

        # 9. Plot the filled contour using time vs depth
        contour = da.plot.contourf(
            x="time",
            y="Depth",
            levels=levels,
            cmap=cmap_used,
            vmin=vmin,
            vmax=vmax,
            extend=extend,
            ax=ax,
            add_colorbar=False,
            **merged_cf_kwargs,
        )

        # Optionally add contour lines on top (if desired, similar to original add_contour logic)
        # Example:
        # if kwargs.get("add_contour"):
        #     cs = ax.contour(da["time"].values, da.coords["Depth"].values, da.values,
        #                     levels=levels, colors="k", linewidths=0.5)
        #     if kwargs.get("label_contours"):
        #         ax.clabel(cs, inline=True, fontsize=8)

        # 10. Optional: plot water surface elevation line
        #       surface_kwargs: dict passed to ax.plot (e.g. color, linewidth)
        if plot_surface:
            skw = surface_kwargs or {}
            # zeta を取得し、同じ spatial_dim,index で抽出
            surf = self.ds["zeta"]
            if spatial_dim:
                surf = surf.isel({spatial_dim: index})
            # 時間フィルタ
            surf = self._apply_time_filter(surf, xlim)
            # プロット
            ax.plot(surf["time"], surf.values, **skw)

        # 11. Invert y-axis so that depth=0 is at the top
        # ax.invert_yaxis()

        # 12. Set axis labels, title, and format the time axis
        # Construct default title if none provided
        if title is None:
            long_name = da.attrs.get("long_name", da.name)
            rolling_text = (
                f" with {rolling_window}-step Rolling Mean" if rolling_window else ""
            )
            title = (
                f"Time Series of {long_name}"
                + (f" ({spatial_dim}={index})" if spatial_dim else "")
                + f"{rolling_text}"
            )
        self._format_time_axis(ax, title, xlabel, ylabel, self.cfg.date_format)

        # Apply depth limits if provided
        if ylim is not None:
            ax.set_ylim(ylim)

        # 13. Create and attach colorbar
        # Determine colorbar label from variable metadata or provided `label`
        units = da.attrs.get("units", "")
        cbar_label = (
            label
            if label is not None
            else (f"{long_name} ({units})" if units else long_name)
        )
        cbar = self._make_colorbar(ax, contour, cbar_label, colorbar_kwargs or {})
        return fig, ax, cbar

    def plot_timeseries_2d(
        self,
        var_name,
        index=None,
        start=None,
        end=None,
        depth=False,
        rolling_window=None,
        ax=None,
        ylim=None,
        levels=20,
        vmin=None,
        vmax=None,
        cmap=None,
        save_path=None,
        method="contourf",
        add_contour=False,
        label_contours=False,
        **kwargs,
    ):
        """
        Obsolete. Remove this method in future versions. Use ts_contourf_z instead.
        Plot a 2D time series for a specified variable as a contour map with time on the x-axis and a vertical coordinate (siglay/siglev) on the y-axis.

        Parameters:
        - var_name: Name of the variable to plot.
        - index: Index of the `node` or `nele` to plot (default: None).
        - start: Start time for the plot (datetime or string).
        - end: End time for the plot (datetime or string).
        - depth: If True, plot depth instead of vertical coordinate.
        - rolling_window: Size of the rolling window for moving average (optional).
        - ax: matplotlib axis object. If None, a new axis will be created.
        - ylim: Y-axis limits (optional).
        - levels: Number of contour levels or specific levels (optional).
        - vmin: Minimum value for color scale (optional).
        - vmax: Maximum value for color scale (optional).
        - cmap: Color map for the plot (optional).
        - save_path: Path to save the plot as an image (optional).
        - method: Plotting method ('contourf' or 'pcolormesh').
        - add_contour: If True, add contour lines on top of the filled contour.
        - label_contours: If True, label contour lines.
        - **kwargs: Additional arguments for customization (e.g., levels).
        """
        if var_name not in self.ds:
            msg = f"Error: The variable '{var_name}' is not found in the dataset."
            logger.error(msg)
            return None

        # Auto-detect the vertical coordinate
        if "siglay" in self.ds[var_name].dims:
            y_coord = "siglay"
        elif "siglev" in self.ds[var_name].dims:
            y_coord = "siglev"
        else:
            raise ValueError(
                f"Variable {var_name} does not have 'siglay' or 'siglev' as a vertical coordinate."
            )

        # Validate the variable's dimensions
        if "time" not in self.ds[var_name].dims:
            raise ValueError(
                f"Variable {var_name} does not have 'time' as a dimension."
            )

        # Select the data for the specified index
        # Validate the dimension
        variable_dims = self.ds[var_name].dims
        if "node" in variable_dims:
            dim = "node"
        elif "nele" in variable_dims:
            dim = "nele"
        else:
            raise ValueError(
                f"Variable {var_name} does not have 'node' or 'nele' as a dimension."
            )

        data = self.ds[var_name]
        if index is not None:
            data = data.isel({dim: index})
        else:
            raise ValueError("Index for 'node' or 'nele' must be provided.")

        # Apply rolling mean if specified
        if rolling_window:
            data = data.rolling(time=rolling_window, center=True).mean()

        # Time range filtering
        time = data["time"]
        if start:
            start = np.datetime64(start)
        if end:
            end = np.datetime64(end)
        time_mask = (time >= start) & (time <= end) if start and end else slice(None)
        data = data.isel(time=time_mask)
        # Transpose data to ensure (Ny, Nx) shape
        data = data.transpose(y_coord, "time")

        # Extract time, vertical coordinate, and data values
        time_vals = data["time"].values  # (Nx,)
        y_vals = data[y_coord].values  # (Ny,)
        values = data.values  # (Ny, Nx)

        # Ensure data dimensions are correct
        if values.shape != (len(y_vals), len(time_vals)):
            raise ValueError(
                f"Shape mismatch: data={values.shape}, time={len(time_vals)}, vertical={len(y_vals)}"
            )

        # Create 2D grid for time and vertical coordinate
        time_grid, y_grid = np.meshgrid(time_vals, y_vals, indexing="xy")
        if depth:
            z = self.ds.z.isel(time=time_mask)[:, :, index].T.values
            if z.shape != (len(y_vals), len(time_vals)):
                raise ValueError(
                    f"Shape mismatch: depth={z.shape}, time={len(time_vals)}, vertical={len(y_vals)}"
                )
            else:
                y_grid = z
                y_coord = "Depth (m)"

        # Create a new axis if not provided
        if ax is None:
            figsize = kwargs.get("figsize", self.cfg.figsize)
            fig, ax = plt.subplots(figsize=figsize)

        # Plot using contourf or pcolormesh
        cmap = cmap or kwargs.pop("cmap", "viridis")
        vmin = vmin or kwargs.pop("vmin", values.min())
        vmax = vmax or kwargs.pop("vmax", values.max())
        levels = levels or kwargs.get("levels", 20)  # Number of contour levels
        if isinstance(levels, int):
            levels = np.linspace(vmin, vmax, levels)
        elif isinstance(levels, (list, np.ndarray)):
            levels = np.array(levels)
        if method == "contourf":
            norm = BoundaryNorm(levels, plt.get_cmap(cmap).N, clip=False)
            cf = ax.contourf(
                time_grid,
                y_grid,
                values,
                levels=levels,
                cmap=cmap,
                norm=norm,
                extend="both",
                **kwargs,
            )
            cbar = plt.colorbar(cf, ax=ax, extend="both")
            if add_contour:
                cs = ax.contour(
                    time_grid, y_grid, values, levels=levels, colors="k", linewidths=0.5
                )
                if label_contours:
                    plt.clabel(cs, inline=True, fontsize=8)
        elif method == "pcolormesh":
            mesh = ax.pcolormesh(time_grid, y_grid, values, cmap=cmap, **kwargs)
            cbar = plt.colorbar(mesh, ax=ax)
        else:
            raise ValueError(f"Invalid method '{method}' for plotting 2D time series.")

        # Format colorbar
        cbar.set_label(var_name, fontsize=self.cfg.fontsize["ylabel"])
        # Add contour labels if requested

        # Format axes
        if ylim is not None:
            ax.set_ylim(ylim)
        rolling_text = (
            f" with {rolling_window}-hour Rolling Mean" if rolling_window else ""
        )
        title = f"Time Series of {var_name} ({dim}={index}){rolling_text}"

        ax.set_title(title, fontsize=self.cfg.fontsize["title"])
        ax.set_xlabel("Time", fontsize=self.cfg.fontsize["xlabel"])
        ax.set_ylabel(y_coord, fontsize=self.cfg.fontsize["ylabel"])
        date_format = kwargs.get("date_format", self.cfg.date_format)
        ax.xaxis.set_major_formatter(DateFormatter(date_format))
        fig.autofmt_xdate()

        # Save or show the plot
        if save_path:
            dpi = kwargs.get("dpi", self.cfg.dpi)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return ax

    @precedence("ax", "lw", "color", "linestyle")
    def plot_2d(
        self,
        *,
        da: xr.DataArray | None = None,
        save_path=None,
        post_process_func=None,
        opts: FvcomPlotOptions | None = None,
        local: dict[str, Any] | None = None,
        **_,
    ):
        """
        Plot scalar field (DataArray) or mesh-only figure.

        Parameters:
        ----------
        da : xr.DataArray, optional
            Field to plot. ``None`` allowed when ``with_mesh=True`` in **kwargs.
        save_path : str, optional
            PNG output path. If omitted, the figure is not saved.
        post_process_func : callable(ax[, da, time]), optional
            Callback executed after base map is drawn.
        **kwargs :
            All legacy keywords (with_mesh, coastlines, obclines, vmin, vmax,
            levels, projection, plot_grid, add_tiles, tile_provider, tile_zoom,
            verbose, xlim, ylim, ...) are still accepted.

        Note:
        -------
        projection in the following can be ccrs.Mercator(), which is the best in mid-latitudes.
            plt.subplots(figsize=figsize, subplot_kw={'projection': projection})
        The other parts, transform=ccrs.PlateCarree() must be set to inform lon/lat coords are used.
        Mercator is not lon/lat coords, so transform=ccrs.PlateCarree() is necessary.

        """

        # unify option source --------------------------------
        assert opts is not None  # for safety; can be omitted
        extra = opts.extra

        # Flag for "scalar field is already drawn" so vector map can skip |U|
        opts.da_is_scalar = da is not None

        projection = opts.projection  # map projection
        use_latlon = opts.use_latlon  # lon/lat or Cartesian
        self.use_latlon = use_latlon
        transform = ccrs.PlateCarree() if self.use_latlon else None

        with_mesh = opts.with_mesh  # draw mesh lines
        coastlines = opts.coastlines  # draw coastlines
        obclines = opts.obclines  # draw open-boundary lines
        plot_grid = opts.plot_grid  # lat/lon grid
        add_tiles = opts.add_tiles  # web tiles
        tile_provider = opts.tile_provider
        tile_zoom = opts.tile_zoom

        verbose = opts.verbose  # console logging
        xlim = opts.xlim  # (xmin,xmax)
        ylim = opts.ylim  # (ymin,ymax)

        # Extract coordinates
        if self.use_latlon:
            x = self.ds["lon"].values
            y = self.ds["lat"].values
        else:
            x = self.ds["x"].values
            y = self.ds["y"].values

        if da is not None:
            values = da.values
            default_cbar_label = f"{da.long_name} ({da.units})"
            cbar_label = extra.get("cbar_label", default_cbar_label)
        else:
            with_mesh = True
        # Extract triangle connectivity
        # nv = self.ds["nv"].values.T - 1  # Convert to 0-based indexing
        nv = self.ds["nv_zero"]
        # Output ranges and connectivity
        if xlim is None:
            xmin, xmax = x.min(), x.max()
        else:
            xmin, xmax = xlim
        if ylim is None:
            ymin, ymax = y.min(), y.max()
        else:
            ymin, ymax = ylim
        if verbose:
            logger.debug(f"x range: {xmin} to {xmax}")
            logger.debug(f"y range: {ymin} to {ymax}")
            logger.debug(
                "nv_ccw shape: %s, nv...min: %s, nv_ccw max: %s",
                self.ds.nv_ccw.shape,
                self.ds.nv_ccw.min(),
                self.ds.nv_ccw.max(),
            )

        # Validate nv_ccw and coordinates
        if verbose:
            if np.isnan(x).any() or np.isnan(y).any():
                raise ValueError("NaN values found in node coordinates.")
            if np.isinf(x).any() or np.isinf(y).any():
                raise ValueError("Infinite values found in node coordinates.")
            if (self.ds.nv_ccw < 0).any() or (self.ds.nv_ccw >= len(x)).any():
                raise ValueError(
                    "Invalid indices in nv_ccw. Check if nv_ccw points to valid nodes."
                )

        # Reverse node order for counter-clockwise triangles that matplotlib expects.
        # nv = nv[:, ::-1]

        # Create Triangulation
        try:
            triang = mtri.Triangulation(x, y, triangles=nv)
            if verbose:
                logger.debug("Number of triangles: %d", len(triang.triangles))
        except ValueError as e:
            logger.error("Error creating Triangulation: %s", e)

            return None

        # Set up axis
        ax = (local or {}).get("ax")  # pre-created Axes or None
        if ax is None:
            figsize = opts.figsize or self.cfg.figsize
            if self.use_latlon:
                fig, ax = plt.subplots(
                    figsize=figsize, subplot_kw={"projection": projection}
                )
            else:
                fig, ax = plt.subplots(figsize=figsize)  # No projection for Cartesian
        else:
            fig = ax.figure

        # Add map tiles if requested
        if add_tiles and self.use_latlon:
            if tile_provider is None:
                raise ValueError(
                    "Tile provider is not set. Please provide a valid tile provider, \
                                 e.g., GoogleTiles(style='satellite')"
                )
            else:
                ax.add_image(tile_provider, tile_zoom)  # type: ignore[call-arg]
            # ax.add_image(tile_provider, 8)  # Zoom level 8 is suitable for regional plots

        # Argument treatment to avoid conflicts with **kwargs
        # with_mesh = opts.with_mesh  # Remove with_mesh from kwargs

        # --------------------------------------------
        # precedence resolution for style parameters
        # --------------------------------------------
        lw = pick_first((local or {}).get("lw"), opts.mesh_linewidth, 0.5)
        color = pick_first((local or {}).get("color"), opts.mesh_color, "#36454F")
        coastline_color = pick_first(
            (local or {}).get("coastline_color"), opts.coastline_color, "gray"
        )
        obcline_color = pick_first(
            (local or {}).get("obcline_color"), opts.obcline_color, "blue"
        )
        linestyle = pick_first(
            (local or {}).get("linestyle"), opts.grid_linestyle, "--"
        )
        # Prepare color plot
        if da is not None:
            cmap_raw = extra.get("cmap", opts.cmap)
            cmap_obj = plt.get_cmap(cmap_raw) if isinstance(cmap_raw, str) else cmap_raw

            # --- build tc_kwargs -----------------------------------------
            tc_kwargs = {k: extra[k] for k in _TRICF_SIG if k in extra}

            # ---- range ---------------------------------------------------
            vmin = extra.get("vmin", opts.vmin)
            vmax = extra.get("vmax", opts.vmax)
            if vmin is None:
                vmin = float(values.min())
            if vmax is None:
                vmax = float(values.max())

            # ---- levels + norm ------------------------------------------
            levels = extra.get("levels", opts.levels)

            if isinstance(levels, int):
                # int  -> 線形レベル配列（境界数 = levels）
                levels_arr = np.linspace(vmin, vmax, levels)
            else:  # list / ndarray
                levels_arr = np.asarray(levels, float)

            # values: ndarray of the DataArray at this time step
            data_min = float(values.min())
            data_max = float(values.max())

            extend_flag = "neither"
            if data_min < vmin and data_max > vmax:
                extend_flag = "both"
            elif data_min < vmin:
                extend_flag = "min"
            elif data_max > vmax:
                extend_flag = "max"

            tc_kwargs.update(
                {
                    "vmin": vmin,
                    "vmax": vmax,
                    "levels": levels_arr,
                    "norm": BoundaryNorm(levels_arr, cmap_obj.N, clip=False),
                }
            )
        # Handle Cartesian coordinates
        if not self.use_latlon:
            title = extra.get("title", "FVCOM Mesh (Cartesian)")
            if da is not None:
                cf = ax.tricontourf(
                    triang,
                    values,
                    cmap=cmap_obj,
                    transform=None,
                    extend=extend_flag,
                    **tc_kwargs,
                )
                cbar = self._make_colorbar(ax, cf, cbar_label, opts=opts)
            if with_mesh:
                ax.triplot(triang, color=color, lw=lw)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_title(title, fontsize=self.cfg.fontsize["title"])
            xlabel = extra.get("xlabel", "X (m)")
            ylabel = extra.get("ylabel", "Y (m)")
            ax.set_xlabel(xlabel, fontsize=self.cfg.fontsize["xlabel"])
            ax.set_ylabel(ylabel, fontsize=self.cfg.fontsize["ylabel"])
            ax.set_aspect("equal")
        # Handle lat/lon coordinates
        else:
            title = extra.get("title", "")
            if da is not None:
                cf = ax.tricontourf(
                    triang,
                    values,
                    cmap=cmap_obj,
                    transform=ccrs.PlateCarree(),
                    extend=extend_flag,
                    **tc_kwargs,
                )
                # cbar = plt.colorbar(cf, ax=ax, extend=extend_flag, orientation='vertical', shrink=1.0)
                cbar = self._make_colorbar(ax, cf, cbar_label, opts=opts)
                # cbar.set_label(cbar_label, fontsize=self.cfg.fontsize["cbar_label"], labelpad=10)
            if with_mesh:
                # Always use PlateCarree here.
                ax.triplot(triang, color=color, lw=lw, transform=ccrs.PlateCarree())
            # Use set_extent for lat/lon ranges
            ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())  # type: ignore[union-attr]
            # ax.set_extent([xmin, xmax, ymin, ymax], crs=projection)
            ax.set_title(title, fontsize=self.cfg.fontsize["title"])
            xlabel = extra.get("xlabel", "Longitude")
            ylabel = extra.get("ylabel", "Latitude")
            ax.set_xlabel(xlabel, fontsize=self.cfg.fontsize["xlabel"])
            ax.set_ylabel(ylabel, fontsize=self.cfg.fontsize["ylabel"])
            ax.set_aspect("equal")

            # Add gridlines for lat/lon. Always use PlateCarree here.
            if plot_grid:
                gl = ax.gridlines(  # type: ignore[attr-defined,union-attr]
                    draw_labels=True, crs=ccrs.PlateCarree(), linestyle=linestyle, lw=lw
                )
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {"size": 11}
                gl.ylabel_style = {"size": 11}
            else:
                gl = ax.gridlines(  # type: ignore[attr-defined,union-attr]
                    draw_labels=False, crs=ccrs.PlateCarree(), linestyle=linestyle, lw=0
                )
                lon_ticks = gl.xlocator.tick_values(xmin, xmax)
                lat_ticks = gl.ylocator.tick_values(ymin, ymax)
                ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
                ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
                ax.xaxis.set_major_formatter(LongitudeFormatter())
                ax.yaxis.set_major_formatter(LatitudeFormatter())
                x_min_proj, y_min_proj = projection.transform_point(
                    xmin, ymin, src_crs=ccrs.PlateCarree()
                )
                x_max_proj, y_max_proj = projection.transform_point(
                    xmax, ymax, src_crs=ccrs.PlateCarree()
                )
                ax.set_xlim(x_min_proj, x_max_proj)
                ax.set_ylim(y_min_proj, y_max_proj)
                ax.tick_params(labelsize=11, labelcolor="black")

        if coastlines:
            logger.info("Plotting coastlines...")

            nv = self.ds.nv_ccw.values
            nbe = np.array(
                [
                    [nv[n, j], nv[n, (j + 2) % 3]]
                    for n in range(len(triang.neighbors))
                    for j in range(3)
                    if triang.neighbors[n, j] == -1
                ]
            )
            for m in range(len(nbe)):
                # ax.plot(x[nbe[m,:]], y[nbe[m,:]], color='gray', linewidth=1, transform=transform)
                ax.plot(
                    x[nbe[m, :]],
                    y[nbe[m, :]],
                    color=coastline_color,
                    linewidth=1,
                    transform=transform,
                )

        if obclines:
            # Plot open boundary lines
            logger.info("Plotting open boundary lines...")

            if "node_bc" not in self.ds:
                raise ValueError(
                    "Dataset does not contain 'node_bc' variable for open boundary lines."
                    " obcfile must be read in FvcomDataLoader."
                )
            node_bc = self.ds.node_bc.values
            ax.plot(
                x[node_bc[:]],
                y[node_bc[:]],
                color=obcline_color,
                linewidth=1,
                transform=transform,
            )

        # -------- vector-overlay hook inside plot_2d -----------------
        if opts.plot_vec2d:
            # --- 1) derive positional index from scalar DataArray -----
            da_time_idx = None
            if da is not None and "time" in da.coords:
                # works for both scalar (0-D) and length-1 1-D coordinates
                label = da.coords["time"].values.item()
                da_time_idx = self._label_to_index(label)  # helper you added

            # --- 2) decide final time for vector plot -----------------
            if opts.vec_time is not None:
                time_for_vector = opts.vec_time  # explicit override
            elif da_time_idx is not None:
                time_for_vector = da_time_idx  # derived from scalar
            else:
                raise ValueError(
                    "plot_vec2d is True but vec_time is not specified and "
                    "the scalar DataArray provides no matching time index. "
                    "Set opts.vec_time explicitly."
                )

            # --- 3) call vector plot ----------------------------------
            self.plot_vector2d(
                time=time_for_vector,
                siglay=opts.vec_siglay,
                reduce=opts.vec_reduce,
                skip=opts.skip,
                ax=ax,
                color=opts.arrow_color,
                opts=opts,
            )

        # --- user post-processing -----------------------------------------
        if post_process_func:
            # Analyse function signature
            func_sig = inspect.signature(post_process_func)
            valid_args = func_sig.parameters.keys()

            # Build kwargs dynamically
            dyn_kwargs = {}
            frame_locals = locals()  # everything defined inside plot_2d
            frame_globals = globals()  # module-level names

            for arg in valid_args:
                if arg == "ax":
                    dyn_kwargs[arg] = ax
                elif arg == "da":
                    dyn_kwargs[arg] = da
                elif arg in frame_locals:
                    dyn_kwargs[arg] = frame_locals[arg]
                elif arg in frame_globals:
                    dyn_kwargs[arg] = frame_globals[arg]
                else:
                    logger.warning("Unable to resolve argument '%s'.", arg)

            # Call the user callback
            post_process_func(**dyn_kwargs)

        # Save the plot if requested
        if save_path:
            dpi = opts.dpi or self.cfg.dpi
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            logger.info("Plot saved to: %s", save_path)

        return ax

    def add_marker(
        self, ax=None, x=None, y=None, marker="o", color="red", size=20, **kwargs
    ):
        """
        Add a marker to the existing plot (e.g., a mesh plot). Must be used with plot_2d.

        Parameters:
        - ax: matplotlib axis object. If None, raise an error because this method must follow plot_2d().
        - x: X coordinate (Cartesian) or longitude (geographic) of the marker.
        - y: Y coordinate (Cartesian) or latitude (geographic) of the marker.
        - marker: Marker style (default: "o").
        - color: Marker color (default: "red").
        - size: Marker size (default: 20).
        - **kwargs: Additional arguments passed to scatter.

        Returns:
        - ax: The axis object with the added marker.
        """
        # Ensure ax is provided
        if ax is None:
            raise ValueError(
                "An axis object (ax) must be provided. Use plot_2d() first."
            )

        # Ensure use_latlon is determined by plot_2d
        if not hasattr(self, "use_latlon"):
            raise AttributeError("The 'use_latlon' attribute must be set by plot_2d().")

        # Ensure x and y are provided
        if x is None or y is None:
            raise ValueError("Both x and y must be specified.")

        transform = ccrs.PlateCarree() if self.use_latlon else None
        ax.scatter(
            x, y, transform=transform, marker=marker, color=color, s=size, **kwargs
        )

        return ax

    def plot_vector2d(
        self,
        *,
        time: int | slice | list | tuple,
        siglay: int | slice | list | tuple | None = None,
        reduce: dict[str, str] | None = None,  # {"time": "mean", "siglay": "mean"}
        skip: int | str | None = None,  # "auto" or explicit integer
        var_u: str = "u",
        var_v: str = "v",
        ax: plt.Axes | None = None,
        opts: FvcomPlotOptions | None = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot a 2-D current vector map for a single (time, siglay) slice.
        This is step-1 minimal version: no averaging, no automatic scaling.

        Parameters
        ----------
        time, siglay : int | slice | list | tuple
            Selection for time and vertical indices.  Examples:
                time=0
                time=slice("2020-01-01","2020-02-01")
                time=[0,1,2,3]
        reduce : dict, optional
            Mapping {'time': 'mean'|'sum'|None, 'siglay': 'mean'|'sum'|None}.
            Use it to compute residual currents or vertical means.
        skip : int | str, optional
            Arrow subsampling interval.  "auto" selects a reasonable value
            based on the mesh size (default "auto").

        """
        # 0) option merge
        if opts is None:
            opts = FvcomPlotOptions()
        opts.extra.update(kwargs)

        # ----------------------------------------------------------
        # Sanitize kwargs: remove internal flags before quiver call
        # ----------------------------------------------------------
        if "with_magnitude" in kwargs:  # user passed as kw-arg
            opts.with_magnitude = bool(kwargs.pop("with_magnitude"))

        if skip == "auto" and opts.skip != "auto":
            skip = opts.skip

        # 1) determine skip
        nele_total = self.ds.sizes.get("nele", len(self.ds["lonc"]))

        if skip is None:  # no thinning
            skip_val = 1
        elif skip == "auto":
            skip_val = self._auto_skip(nele_total)
        else:
            skip_val = int(skip)

        # 2) slice & reduce u,v
        uc, vc = self._select_and_reduce_uv(
            self.ds[var_u],
            self.ds[var_v],
            time_sel=time,
            siglay_sel=siglay,
            reduce=reduce,
        )

        # 3) prepare base map if ax is None (unchanged)
        if ax is None:
            ax = self.plot_2d(da=None, opts=FvcomPlotOptions(with_mesh=True))

        # 4) build quiver kwargs ------------------------------------
        arrow_kwargs = {
            "scale_units": "xy",
            "angles": "xy",
            "color": kwargs.get("color", opts.arrow_color),
            "alpha": opts.arrow_alpha,
            "width": opts.arrow_width,
            # 'scale' will be injected later only if needed
        }
        arrow_kwargs.update(kwargs)  # allow user override

        # ---- decide scale -------------------------------------------
        # priority: explicit **kwargs > opts.scale
        scale_val = kwargs.get("scale", opts.scale)

        if scale_val is None:
            arrow_kwargs.pop("scale", None)  # 自動
        else:
            arrow_kwargs["scale"] = float(scale_val)

        # 5) draw magnitude ------------------------------------
        draw_mag = opts.with_magnitude and not getattr(opts, "da_is_scalar", False)

        if draw_mag:
            import matplotlib.tri as mtri

            lon_n = self.ds["lon"].values
            lat_n = self.ds["lat"].values
            tri_nv = self.ds["nv_zero"].values
            triang = mtri.Triangulation(lon_n, lat_n, triangles=tri_nv)

            mag = np.hypot(uc, vc)  # (nele,) element-centre magnitude
            cf = ax.tripcolor(
                triang,
                facecolors=mag,
                cmap=opts.cmap,
                transform=ccrs.PlateCarree(),
                shading="flat",
                zorder=opts.vec_zorder - 1,
            )

            # -------- move colorbar INSIDE the draw_mag block ----------
            self._make_colorbar(ax, cf, label="|U| (m/s)", opts=opts)

        # 6) quiver plot --------------------------------------------
        q = ax.quiver(
            self.ds["lonc"][::skip_val],
            self.ds["latc"][::skip_val],
            uc[::skip_val],
            vc[::skip_val],
            transform=ccrs.PlateCarree(),
            zorder=opts.vec_zorder,
            **arrow_kwargs,
        )

        # 7) add quiverkey ---------------------------------------------
        if opts.show_vec_legend:
            # automatic reference speed = 30 % of max |u,v|
            ref_speed = (
                np.hypot(uc, vc).max() * 0.3
                if opts.vec_legend_speed is None
                else opts.vec_legend_speed
            )
            ax.quiverkey(
                q,
                *opts.vec_legend_loc,
                ref_speed,
                f"{ref_speed:.2f} m/s",
                labelpos="E",
                coordinates="axes",
                color=kwargs.get("color", opts.arrow_color),
                fontproperties={"size": self.cfg.fontsize_legend},
            )

        return ax

    def ts_contourf(
        self,
        da: xr.DataArray,
        index: int = None,
        x="time",
        y="siglay",
        xlim=None,
        ylim=None,
        xlabel="Time",
        ylabel="Sigma",
        title=None,
        rolling_window=None,
        min_periods=None,
        ax=None,
        date_format=None,
        contourf_kwargs: dict = None,
        colorbar_kwargs: dict = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes, Colorbar]:
        """
        Plot a contour map of vertical time-series DataArray.
        contourf_kwargs and **kwargs are combined to flexibly pass any contourf parameters; colorbar_kwargs is for colorbar settings.

        Parameters:
        ----------
        da (DataArray): DataArray for specified var_name with the dimension of (time, siglay/siglev [, node/nele]).
        index (int): Index of the node or element to plot (optional).
        x (str): Name of the x-axis coordinate. Default is 'time'.
        y (str): Name of the y-axis coordinate. Default is 'siglay'.
        xlim (tuple): Tuple of start and end times (e.g., ('2010-01-01', '2022-12-31')).
        ylim (tuple): Vertical range for the y-axis (e.g., (0, 1)).
        xlabel (str): Label for the x-axis. Default is 'Time'.
        ylabel (str): Label for the y-axis. Default is 'Depth (m)'.
        title (str): Title for the plot. Default is None.
        rolling_window (int): Size of the rolling window for moving average in hours (Default: None).
            24*30+1 (monthly mean)
        min_periods (int): Minimum number of data points required in the rolling window.
                        If None, defaults to window // 2 + 1.
        ax (matplotlib.axes.Axes): An existing axis to plot on. If None, a new axis will be created.
        date_format (str): Date format for the x-axis. Default is None.
        contourf_kwargs (dict): Arguments for contourf.
        colorbar_kwargs (dict): Arguments for colorbar.
        **kwargs: Arguments for contourf. Not supporting additional kwargs for colorbar.

        Returns:
        ----------
        tuple: (fig, ax, cbar)
        """
        contourf_kwargs = contourf_kwargs or {}
        colorbar_kwargs = colorbar_kwargs or {}

        # Automatically detect the spatial dimension ('node' or 'nele') and apply the given index
        spatial_dim = next((d for d in ("node", "nele") if d in da.dims), None)
        if spatial_dim:
            if index is None:
                raise ValueError(
                    f"Index must be provided for '{spatial_dim}' dimension"
                )
            da = da.isel({spatial_dim: index})
        elif index is not None:
            raise ValueError(
                f"No 'node' or 'nele' dimension found in DataArray dims {da.dims}"
            )

        # Apply rolling mean if specified
        if rolling_window:
            if min_periods is None:
                min_periods = rolling_window // 2 + 1
            da = da.rolling(
                time=rolling_window, center=True, min_periods=min_periods
            ).mean()

        # Extract metadata for labels
        long_name = da.attrs.get("long_name", da.name)
        units = da.attrs.get("units", "-")
        cbar_label = f"{long_name} ({units})"
        if title is None:
            rolling_text = (
                f" with {rolling_window}-hour Rolling Mean" if rolling_window else ""
            )
            if spatial_dim is not None:
                title = (
                    f"Time Series of {long_name} ({spatial_dim}={index}){rolling_text}"
                )
            else:
                title = f"Time Series of {long_name}{rolling_text}"

        date_format = date_format or self.cfg.date_format

        # Time range filtering via xlim tuple
        da = self._apply_time_filter(da, xlim)

        # Create figure and axis if not provided
        if ax is None:
            fig = plt.figure(figsize=self.cfg.figsize, dpi=self.cfg.dpi)
            gs = GridSpec(1, 1, left=0.1, right=0.9, top=0.9, bottom=0.1)
            ax = fig.add_subplot(gs[0])
        else:
            fig = ax.figure

        # Merge kwargs with contourf_kwargs
        merged_contourf_kwargs, levels, cmap, vmin, vmax, extend = (
            self._prepare_contourf_args(da, contourf_kwargs, kwargs)
        )

        # Plot the contour map
        # plot = da.plot.contourf(x=x, y=y, ylim=ylim, levels=levels, cmap=cmap,
        #                        vmin=vmin, vmax=vmax, extend=extend, ax=ax,
        #                        add_colorbar=False, **merged_contourf_kwargs)
        # if ylim is None:
        #    ylim = (da[y].min().item(), da[y].max().item())
        #    ax.set_ylim(ylim)
        # Build kwargs for contourf
        plot_kwargs = {
            "x": x,
            "y": y,
            "levels": levels,
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
            "extend": extend,
            "ax": ax,
            "add_colorbar": False,
        }
        # Only include ylim if explicitly set
        if ylim is not None:
            plot_kwargs["ylim"] = ylim

        # Call contourf
        plot = da.plot.contourf(**plot_kwargs, **merged_contourf_kwargs)

        # Set axis formatting
        self._format_time_axis(ax, title, xlabel, ylabel, date_format)

        # ax.invert_yaxis()

        # Add colorbar
        cb_copy = dict(colorbar_kwargs or {})
        label_to_use = cb_copy.pop("label", cbar_label)
        cbar = self._make_colorbar(ax, plot, label_to_use, cb_copy)

        return fig, ax, cbar

    def ts_plot(
        self,
        da: xr.DataArray = None,
        varname: str = None,
        index: int = None,
        k: int = None,
        ax=None,
        xlabel: str | bool | None = None,
        ylabel: str | bool | None = True,
        title: str | bool | None = None,
        color: str = None,
        linestyle: str = None,
        date_format: str = None,
        xlim: tuple = None,
        ylim: tuple = None,
        rolling_window=None,
        log=False,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        1-D time series plot. Provides a DataArray or variable name to extract from the dataset.
        Accept either da or varname exclusively.
        Passing the DataArray objects directly is effective when you need to avoid performance degradation in parallel processing.

        Parameters:
        ----------
        da : xr.DataArray, optional
            DataArray with a 'time' dimension.
        varname : str, optional
            Variable name to extract from the dataset. Default: None.
        index : int, optional
            Index for spatial dimension (node/nele). Not needed for pure time series.
        k : int, optional
            Layer index for vertical dimension (siglay/siglev). Not needed for 2D or 1D series.
        ax : matplotlib.axes.Axes, optional
            Existing axis. Creates new one if None.
        xlabel : str | bool | None, optional
            X-axis label. If True, use default label 'Time'. If None, no label. Default None.
        ylabel : str, optional
            Y-axis label. Default: da.long_name or da.name.
        title : str, optional
            Plot title. Default: '<ylabel> Time Series'.
        color : str, optional
            Line color. Default: self.cfg.plot_color.
        linestyle : str, optional
            Line style. Default: '-'.
        date_format : str, optional
            Date formatter. Default: self.cfg.date_format.
        xlim : tuple, optional
            X-axis limits (start, end). Default: None.
        ylim : tuple, optional
            Y-axis limits (ymin, ymax). Default: None.
        rolling_window : int, optional
            Size of the rolling window for moving average. Default: None.
        log : bool, optional
            If True, use logarithmic scale. Default: False.
        **kwargs : dict
            Extra keyword args for ax.plot().

        Returns:
        ----------
        tuple: (fig, ax)
        """
        # 1) Slice da based on its dimensions
        if da is None:
            # da is extracted from self.ds[varname]
            if varname is None:
                raise ValueError("Either 'da' or 'varname' must be provided.")
            da = self.ds[varname]
        elif varname is not None:
            raise ValueError("Only one of 'da' or 'varname' should be provided.")

        data, spatial_dim, layer_dim = self._slice_time_series(da, index, k)

        # 2) Apply rolling mean before time filtering
        data = self._apply_rolling(data, rolling_window)

        # 3) Prepare labels and title
        xlabel, ylabel, title = self._prepare_ts_labels(
            data,
            spatial_dim,
            layer_dim,
            index,
            k,
            rolling_window,
            xlabel,
            ylabel,
            title,
        )

        # 4) Time filtering and formatting
        date_format = date_format or self.cfg.date_format
        data = self._apply_time_filter(data, xlim)
        times = data["time"].values
        values = data.values

        # 5) Prepare figure/axis
        if ax is None:
            fig, ax = plt.subplots(figsize=self.cfg.figsize, dpi=self.cfg.dpi)
        else:
            fig = ax.figure

        # 6) Plot
        color = color or self.cfg.plot_color
        linestyle = linestyle or "-"
        ax.plot(times, values, color=color, linestyle=linestyle, **kwargs)
        # if log:
        #    ax.set_yscale("log")
        # Apply log scale via helper (handles warnings for non-positive data)
        self._apply_log_scale(ax, data, log)

        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.cfg.fontsize["xlabel"])
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.cfg.fontsize["ylabel"])
        if title:
            ax.set_title(title, fontsize=self.cfg.fontsize["title"])

        # 7) Y‑axis limits
        if ylim is not None:
            ymin, ymax = ylim
            curr_min, curr_max = ax.get_ylim()
            ymin = curr_min if ymin is None else ymin
            ymax = curr_max if ymax is None else ymax
            ax.set_ylim(ymin, ymax)

        # 8) Final formatting
        self._format_time_axis(ax, title, xlabel, ylabel, date_format)

        return fig, ax

    def section_contourf_z(
        self,
        da: xr.DataArray,
        lat: float = None,
        lon: float = None,
        line: list[tuple[float, float]] = None,
        spacing: float = 100.0,
        xlim: tuple = None,
        ylim: tuple = None,
        xlabel: str = "Distance (m)",
        ylabel: str = "Depth (m)",
        title: str = None,
        ax=None,
        land_color: str = "#A0522D",  # Default seabed/land color (sienna)
        contourf_kwargs: dict = None,
        colorbar_kwargs: dict = None,
        **kwargs,
    ):
        """
        Plot a vertical section of a 3D variable (da) on FVCOM mesh.

        Parameters:
          da: DataArray with dims (siglay/siglev, node) at single time
          lat, lon: constant latitude or longitude for section
          line: list of (lon, lat) pairs defining arbitrary transect
          spacing: sampling interval (m)
          xlim, ylim: axis limits
          xlabel, ylabel, title: plot labels
          ax: existing Matplotlib Axes
          land_color: seabed/land color (default: "#A0522D" # Sienna)
          contourf_kwargs: dict of base contourf args
          colorbar_kwargs: dict for colorbar
          **kwargs: extra contourf keywords (override contourf_kwargs)

        Returns: fig, ax, cbar
        """

        # 0 Validate that requested lat/lon lie within the dataset domain
        lon_vals = self.ds["lon"].values if "lon" in self.ds else None
        lat_vals = self.ds["lat"].values if "lat" in self.ds else None
        if lat is not None and lat_vals is not None:
            lat_min, lat_max = float(lat_vals.min()), float(lat_vals.max())
            if not (lat_min <= lat <= lat_max):
                raise ValueError(
                    f"Latitude {lat} is outside domain bounds [{lat_min}, {lat_max}]."
                )
        if lon is not None and lon_vals is not None:
            lon_min, lon_max = float(lon_vals.min()), float(lon_vals.max())
            if not (lon_min <= lon <= lon_max):
                raise ValueError(
                    f"Longitude {lon} is outside domain bounds [{lon_min}, {lon_max}]."
                )

        # Determine vertical dimension
        vert_dim = "siglay" if "siglay" in da.dims else "siglev"

        # Get depth array at same time
        z_all = self.ds["z"]
        if "time" in z_all.dims:
            if "time" in da.coords:
                z_slice = z_all.sel(time=da["time"], method="nearest")
            else:
                z_slice = z_all.isel(time=0)
            z2d = z_slice.values  # (vertical, node)
        else:
            z2d = z_all.values

        # Prepare mesh triangulation for domain test
        lon_n = self.ds["lon"].values
        lat_n = self.ds["lat"].values
        tris = self.ds["nv"].values.T - 1
        triang = mtri.Triangulation(lon_n, lat_n, triangles=tris)
        trifinder = triang.get_trifinder()

        # Build KDTree on projected nodes for nearest-node lookup
        mean_lon, mean_lat = lon_n.mean(), lat_n.mean()
        zone = int((mean_lon + 180) // 6) + 1
        hemi = "north" if mean_lat >= 0 else "south"
        proj = pyproj.Proj(f"+proj=utm +zone={zone} +{hemi} +datum=WGS84")
        x_n, y_n = proj(lon_n, lat_n)
        tree = KDTree(np.column_stack((x_n, y_n)))

        # Define transect endpoints
        if line:
            pts = line
        elif lat is not None:
            pts = [(float(lon_n.min()), lat), (float(lon_n.max()), lat)]
        elif lon is not None:
            pts = [(lon, float(lat_n.min())), (lon, float(lat_n.max()))]
        else:
            raise ValueError("Specify lat, lon, or line for section.")

        lons, lats, distances = self._sample_transect(
            lat=lat, lon=lon, line=line, spacing=spacing
        )
        # Domain mask
        tri_idx = trifinder(lons, lats)
        inside = tri_idx != -1

        # Nearest-node for each sample
        x_s, y_s = proj(lons, lats)
        _, idx_n = tree.query(np.column_stack((x_s, y_s)))

        # Extract variable and depth
        X, Y, V = self._extract_section_data(da, lons, lats, distances)
        # Plot
        fig = (
            ax.figure if ax else plt.figure(figsize=self.cfg.figsize, dpi=self.cfg.dpi)
        )
        ax = ax or fig.add_subplot(1, 1, 1)

        # Build DataArray for section:
        # - dims: (siglay, distance)
        # - coords:
        #    * distance: 1D array of sampled distances
        #    * depth:    2D array of shape (siglay, distance)
        sec_da = xr.DataArray(
            V,
            dims=("siglay", "distance"),
            coords={"distance": distances, "depth": (("siglay", "distance"), Y)},
        )

        # 1 Prepare contourf args on section-DataArray
        merged_cf_kwargs, levels, cmap_used, vmin, vmax, extend = (
            self._prepare_contourf_args(sec_da, contourf_kwargs, kwargs)
        )

        # 2 use xarray's contourf wrapper (same as ts_contourf)
        cs = sec_da.plot.contourf(
            x="distance",
            y="depth",
            levels=levels,
            corner_mask=False,
            cmap=cmap_used,
            vmin=vmin,
            vmax=vmax,
            extend=extend,
            # linewidths=0,
            antialiased=False,
            ax=ax,
            add_colorbar=False,
            **merged_cf_kwargs,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        # Set x-axis to distance
        if xlim is not None:
            ax.set_xlim(*xlim)
        else:
            ax.set_xlim(distances.min(), distances.max())
        # Set y-axis so shallow (near-zero) is at top and deep (large negative) at bottom
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(np.nanmin(Y), np.nanmax(Y))

        # 3 Now that y-limits are fixed, fill seabed and mesh-missing regions
        # a) land/water mask along transect
        mask_land = np.all(np.isnan(V), axis=0)  # True → 陸列
        mask_water = ~mask_land

        # b) seabed profile (deepest valid depth for each column)
        #    mask Land columns → use masked array to suppress All-NaN warning
        Y_water = np.where(mask_land, np.nan, Y)  # land is NaN
        bottom_depth = (
            np.ma.masked_invalid(Y_water)  # safe for all NaN columns
            .min(axis=0)
            .filled(np.nan)  # land columns are still NaN
        )

        # c) 基準線など
        ymin_axis, ymax_axis = ax.get_ylim()
        fill_base = min(ymin_axis, ymax_axis)

        # a) fill below seabed line (land patch under ocean)
        ax.fill_between(
            distances,
            bottom_depth,
            fill_base,
            where=mask_water,
            facecolor=land_color,
            edgecolor=None,
            zorder=cs.zorder - 0.5,  # between axes background and contourf
            clip_on=True,
        )
        # b) fill entire vertical for land columns
        # mask_nan = np.all(np.isnan(V), axis=0)
        ax.fill_between(
            distances,
            fill_base,
            ymax_axis,
            where=mask_land,
            facecolor=land_color,
            edgecolor=None,
            zorder=cs.zorder - 0.5,
            clip_on=True,
        )

        # Plot the seabed line on top (use true bottom = deepest depth)
        ax.plot(
            distances,
            bottom_depth,
            color="k",
            linestyle="-",
            linewidth=0.5,
            zorder=cs.zorder + 1,
        )

        # Add colorbar
        cbar = self._make_colorbar(
            ax,
            cs,
            da.attrs.get("long_name", da.name)
            + (f" ({da.attrs.get('units','')})" if "units" in da.attrs else ""),
            colorbar_kwargs or {},
        )

        return fig, ax, cbar

    # --------------------------------
    # Private helper methods
    # --------------------------------

    def _sample_transect(
        self,
        lat: float = None,
        lon: float = None,
        line: list[tuple[float, float]] = None,
        spacing: float = 200.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate evenly spaced sample points along a transect.
        Returns: lons, lats, cumulative distances (m).
        """

        # Determine transect line endpoints
        if line:
            pts = line
        elif lat is not None:
            pts = [
                (float(self.ds["lon"].min()), lat),
                (float(self.ds["lon"].max()), lat),
            ]
        elif lon is not None:
            pts = [
                (lon, float(self.ds["lat"].min())),
                (lon, float(self.ds["lat"].max())),
            ]
        else:
            raise ValueError("Specify lat, lon, or line for section.")

        geod = pyproj.Geod(ellps="WGS84")
        samples = [pts[0]]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            lon0, lat0 = p0
            lon1, lat1 = p1
            _, _, dist = geod.inv(lon0, lat0, lon1, lat1)
            steps = int(dist // spacing)
            for i in range(1, steps + 1):
                lon_i, lat_i, _ = geod.fwd(
                    lon0, lat0, geod.inv(lon0, lat0, lon1, lat1)[0], i * spacing
                )
                samples.append((lon_i, lat_i))
            samples.append((lon1, lat1))

        lons = np.array([p[0] for p in samples])
        lats = np.array([p[1] for p in samples])
        dists = np.zeros(len(samples))
        for i in range(1, len(samples)):
            _, _, seg = geod.inv(lons[i - 1], lats[i - 1], lons[i], lats[i])
            dists[i] = dists[i - 1] + seg
        return lons, lats, dists

    def _extract_section_data(
        self, da: xr.DataArray, lons: np.ndarray, lats: np.ndarray, dists: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build 2D grids X (distance), Y (depth), and V (values).
        """

        # Triangulate mesh for domain test
        lon_n = self.ds["lon"].values
        lat_n = self.ds["lat"].values
        triangles = self.ds["nv"].values.T - 1
        triang = mtri.Triangulation(lon_n, lat_n, triangles=triangles)
        trifinder = triang.get_trifinder()

        # Project nodes to UTM for KDTree
        mean_lon, mean_lat = lon_n.mean(), lat_n.mean()
        zone = int((mean_lon + 180) // 6) + 1
        hemi = "north" if mean_lat >= 0 else "south"
        proj = pyproj.Proj(f"+proj=utm +zone={zone} +{hemi} +datum=WGS84")
        x_n, y_n = proj(lon_n, lat_n)
        tree = KDTree(np.column_stack((x_n, y_n)))

        # Determine in-domain samples
        tri_idx = trifinder(lons, lats)
        inside = tri_idx != -1

        # Nearest-node lookup
        x_s, y_s = proj(lons, lats)
        _, idx_n = tree.query(np.column_stack((x_s, y_s)))

        # Extract variable values and mask
        V = da.values[:, idx_n]
        V[:, ~inside] = np.nan

        # Extract depth z and mask
        z_all = self.ds["z"]
        if "time" in z_all.dims and "time" in da.coords:
            z2d = z_all.sel(time=da["time"], method="nearest").values
        else:
            z2d = z_all.values
        Y = z2d[:, idx_n]
        Y[:, ~inside] = np.nan

        # Build distance grid
        X = np.broadcast_to(dists[np.newaxis, :], Y.shape)
        return X, Y, V

    def _prepare_contourf_args(self, da, contourf_kwargs, extra_kwargs):
        """
        Merge contourf_kwargs and extra_kwargs, and extract levels, cmap, vmin, vmax, extend.
        Returns:
            merged_kwargs, levels, cmap, vmin, vmax, extend
        """
        contourf_kwargs = contourf_kwargs or {}
        merged = {**contourf_kwargs, **extra_kwargs}

        # Extract and handle contourf parameters with appropriate defaults
        levels = merged.pop("levels", getattr(self.cfg, "levels", None))
        cmap = merged.pop("cmap", getattr(self.cfg, "cmap", None))
        raw_vmin = merged.pop("vmin", None)
        raw_vmax = merged.pop("vmax", None)

        # Determine vmin and vmax from data if not explicitly provided
        vmin = raw_vmin if raw_vmin is not None else da.min().item()
        vmax = raw_vmax if raw_vmax is not None else da.max().item()

        # Convert levels if necessary:
        if levels is None:
            # Default to config levels if available, otherwise fallback to 21 levels
            levels = self.cfg.levels if hasattr(self.cfg, "levels") else None
            if levels is None:
                levels = 21
        if isinstance(levels, (int, np.integer)):
            # If levels is an integer, create that many linearly spaced levels
            n_levels = int(levels)
            if vmin == vmax:
                # Handle degenerate case: all data equal
                levels = np.array([vmin, vmax])
            else:
                levels = np.linspace(vmin, vmax, n_levels)
        elif isinstance(levels, (list, np.ndarray)):
            # Use the list/array of levels directly
            levels = np.asarray(levels)

        # If no explicit vmin/vmax given, align vmin/vmax with the levels range
        if raw_vmin is None and raw_vmax is None and isinstance(levels, np.ndarray):
            if levels.size > 0:
                vmin = levels.min()
                vmax = levels.max()

        # Determine the "extend" for contourf (how to handle values outside levels range)
        if "extend" in merged:
            extend = merged.pop("extend")
        else:
            data_min = da.min().item()
            data_max = da.max().item()
            if vmin <= data_min and vmax >= data_max:
                extend = "neither"
            elif vmin > data_min and vmax >= data_max:
                extend = "min"
            elif vmax < data_max and vmin <= data_min:
                extend = "max"
            else:
                extend = "both"

        return merged, levels, cmap, vmin, vmax, extend

    def _make_colorbar(
        self,
        ax,
        mappable,
        label,
        colorbar_kwargs: dict | None = None,
        opts: FvcomPlotOptions | None = None,
    ):
        """
        Create and attach a colorbar to `ax` for the given mappable (QuadContourSet).
        Obeying priority:
            opts.*  >  self.cfg.*  > hard-coded

        Parameters
        ----------
        ax              : matplotlib Axes
        mappable        : QuadContourSet / ScalarMappable
        label           : str  (default label)
        colorbar_kwargs : dict
        opts            : FvcomPlotOptions | None
                        No colorbar if opts.colorbar=False
                        Override by opts.cbar_size / cbar_pad / cbar_kwargs
        """

        if opts is not None and opts.colorbar is False:  # ← ここを追加
            return None

        size = (
            opts.cbar_size
            if (opts and opts.cbar_size is not None)
            else self.cfg.cbar_size
        )
        pad = (
            opts.cbar_pad if (opts and opts.cbar_pad is not None) else self.cfg.cbar_pad
        )

        extra = dict(self.cfg.cbar_kwargs)
        if opts and opts.cbar_kwargs:
            extra.update(opts.cbar_kwargs)
        if colorbar_kwargs:
            extra.update(colorbar_kwargs)

        divider = make_axes_locatable(ax)
        if isinstance(ax, GeoAxes):
            cax = divider.append_axes("right", size=size, pad=pad, axes_class=plt.Axes)
        else:
            cax = divider.append_axes("right", size=size, pad=pad)

        # cax = divider.append_axes("right", size=size, pad=pad)
        cbar = ax.figure.colorbar(mappable, cax=cax, **extra)

        lb = opts.cbar_label if (opts and opts.cbar_label is not None) else label
        if lb:
            cbar.set_label(
                lb,
                fontsize=self.cfg.fontsize["cbar_label"],
                labelpad=extra.pop("labelpad", 10),
            )

        return cbar

    def _format_time_axis(
        self, ax: plt.Axes, title: str, xlabel: str, ylabel: str, date_format: str
    ) -> None:
        """
        Helper function to format the time axis for time series plots.
        """
        ax.set_title(title, fontsize=self.cfg.fontsize["title"])
        ax.set_xlabel(xlabel, fontsize=self.cfg.fontsize["xlabel"])
        ax.set_ylabel(ylabel, fontsize=self.cfg.fontsize["ylabel"])
        ax.xaxis.set_major_formatter(DateFormatter(date_format))
        ax.figure.autofmt_xdate()

    def _apply_time_filter(self, da: xr.DataArray, xlim: tuple | None) -> xr.DataArray:
        """
        Apply time filtering to da based on xlim=(start, end).
        If xlim is None, return da unchanged.
        """
        if xlim is None:
            return da

        start, end = xlim
        start_sel = np.datetime64(start) if start is not None else None
        end_sel = np.datetime64(end) if end is not None else None
        return da.sel(time=slice(start_sel, end_sel))

    def _slice_time_series(
        self, da: xr.DataArray, index: int = None, k: int = None
    ) -> tuple[xr.DataArray, Hashable | None, str | None]:
        """
        Slice a DataArray for 1D or vertical time series.

        Returns:
          sliced DataArray, spatial dimension name, layer dimension name
        """
        dims = da.dims

        # 3D time series (time, layer, space)
        if "time" in dims and ("siglay" in dims or "siglev" in dims):
            layer_dim = "siglay" if "siglay" in dims else "siglev"
            spatial_dim = next(d for d in dims if d not in ("time", layer_dim))
            if index is None or k is None:
                raise ValueError(f"Both index and k are required for dims {dims}")
            sliced = da.isel({spatial_dim: index, layer_dim: k})

        # 2D time series (time, space)
        elif "time" in dims and ("node" in dims or "nele" in dims):
            layer_dim = None
            spatial_dim = "node" if "node" in dims else "nele"
            if index is None:
                raise ValueError(f"Index is required for dims {dims}")
            sliced = da.isel({spatial_dim: index})

        # Pure 1D time series (time,)
        elif dims == ("time",):
            sliced = da
            spatial_dim = layer_dim = None

        else:
            raise ValueError(f"Unsupported DataArray dims: {dims}")

        return sliced, spatial_dim, layer_dim

    def _prepare_ts_labels(
        self,
        data: xr.DataArray,
        spatial_dim: Hashable | None,
        layer_dim: str | None,
        index: int | None,
        k: int | None,
        rolling_window: int | None,
        xlabel: str | bool | None,
        ylabel: str | bool | None,
        title: str | bool | None,
    ) -> tuple[str, str, str]:
        """
        Prepare and return xlabel, ylabel, title for ts_plot.

        Returns: (xlabel, ylabel, title)
        """
        # Use long_name or variable name for label base
        long_name = data.attrs.get("long_name", data.name)
        units = data.attrs.get("units", "")

        # Set default xlabel only when None
        if isinstance(xlabel, str):
            xlabel = xlabel.strip()
        elif xlabel:  # Truthy but not str → True
            xlabel = "Time"
        else:  # None, False, "" → no label
            xlabel = ""

        # Set default ylabel only when None
        if isinstance(ylabel, str):
            ylabel = ylabel.strip()
        elif ylabel:  # Truthy but not str → True
            ylabel = f"{long_name} ({units})"
        else:  # None, False, "" → no label
            ylabel = ""

        # Set default title only when True
        if isinstance(title, str):
            title = title.strip()
        elif title:  # Truthy but not str → True
            # add rolling text if requested
            roll_txt = (
                f" with {rolling_window}-hour Rolling Mean" if rolling_window else ""
            )
            if spatial_dim:
                if layer_dim:
                    title = f"Time Series of {long_name} ({spatial_dim}={index}, {layer_dim}={k}){roll_txt}"
                else:
                    title = (
                        f"Time Series of {long_name} ({spatial_dim}={index}){roll_txt}"
                    )
            else:
                title = f"Time Series of {long_name}{roll_txt}"
        else:  # None, False, "" → no label
            title = ""

        return xlabel, ylabel, title

    def _apply_log_scale(
        self, ax: plt.Axes, data: xr.DataArray, log_flag: bool
    ) -> None:
        """
        Apply logarithmic scale to the y-axis if requested, using proper locator and formatter.
        """
        if not log_flag:
            return

        # Only positive values can be plotted on a log scale
        if data.min().item() <= 0:
            import warnings

            warnings.warn(
                "Log scale requested but data contains non-positive values; skipping log scale."
            )
            return

        # Set y-axis to log scale with base-10 locator/formatter
        ax.set_yscale("log", base=10)
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(LogFormatter())

    def _apply_rolling(
        self,
        da: xr.DataArray,
        window: int | None = None,
        min_periods: int | None = None,
    ) -> xr.DataArray:
        """
        Apply centered rolling mean on time axis with optional min_periods.
        """
        if window is None:
            return da
        mp = min_periods if min_periods is not None else window // 2 + 1
        return da.rolling(time=window, center=True, min_periods=mp).mean()

    def _interp_uv_to_centers(
        self, u: xr.DataArray, v: xr.DataArray
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Return velocities located at cell centers.

        * If u,v are defined on 'node', perform simple node-average.
        * If they are already on 'nele', return as-is.
        """
        # case 1: already cell-centered
        if "nele" in u.dims and "nele" in v.dims:
            return u, v

        # case 2: node-based -> average three surrounding nodes
        if "node" in u.dims and "node" in v.dims:
            idx = self.ds["nv_zero"].values  # (nele, 3)
            uc = (
                u.isel(node=xr.DataArray(idx[:, 0], dims="nele"))
                + u.isel(node=xr.DataArray(idx[:, 1], dims="nele"))
                + u.isel(node=xr.DataArray(idx[:, 2], dims="nele"))
            ) / 3.0
            vc = (
                v.isel(node=xr.DataArray(idx[:, 0], dims="nele"))
                + v.isel(node=xr.DataArray(idx[:, 1], dims="nele"))
                + v.isel(node=xr.DataArray(idx[:, 2], dims="nele"))
            ) / 3.0
            return uc, vc

        # unsupported dimension set
        raise ValueError(
            f"Unsupported dims for u,v: {u.dims} / {v.dims}. "
            "Expect 'node' or 'nele'."
        )

    def _select_and_reduce_uv(
        self,
        u3d: xr.DataArray,
        v3d: xr.DataArray,
        time_sel=None,
        siglay_sel=None,
        reduce: dict | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Slice 3-D velocity fields and apply optional averaging, then
        interpolate to cell centres.

        Parameters
        ----------
        u3d, v3d : DataArray
            3-D arrays with dims ('time', 'siglay', 'node' or 'nele').
        time_sel  : int | slice | list | tuple | None
            Selection for the 'time' dimension.  None → keep all.
        siglay_sel: int | slice | list | tuple | None
            Selection for the 'siglay' dimension. None → keep all.
        reduce    : dict | None
            Mapping of {"time": "mean"|"sum"|None, "siglay": "mean"|"sum"|"thickness"|None}.

        Returns
        -------
        uc, vc : DataArray
            1-D velocity components defined on 'nele' (cell centres).
        """
        # 1) slicing -----------------------------------------------------
        if time_sel is not None:
            u3d = self._apply_indexer(u3d, "time", time_sel)
            v3d = self._apply_indexer(v3d, "time", time_sel)

        if siglay_sel is not None:
            u3d = self._apply_indexer(u3d, "siglay", siglay_sel)
            v3d = self._apply_indexer(v3d, "siglay", siglay_sel)

        # 2) reduction ---------------------------------------------------
        reduce = reduce or {}

        # -- vertical reduction first ------------------------------------
        op_vert = reduce.get("siglay")
        if "siglay" in u3d.dims:  # remain only if not sliced away
            if op_vert == "thickness":
                # weighted average using physical layer thickness
                w = self._layer_thickness(siglay=siglay_sel, time_sel=time_sel)
                u3d = (u3d * w).sum("siglay") / w.sum("siglay")
                v3d = (v3d * w).sum("siglay") / w.sum("siglay")
            elif op_vert == "mean":
                u3d = u3d.mean("siglay")
                v3d = v3d.mean("siglay")
            elif op_vert == "sum":
                u3d = u3d.sum("siglay")
                v3d = v3d.sum("siglay")
            # if op_vert is None: no vertical reduction

        # -- temporal reduction second -----------------------------------
        op_time = reduce.get("time")
        if "time" in u3d.dims:  # remain only if not sliced away
            if op_time == "mean":
                u3d = u3d.mean("time")
                v3d = v3d.mean("time")
            elif op_time == "sum":
                u3d = u3d.sum("time")
                v3d = v3d.sum("time")
            # if op_time is None: no temporal reduction

        # 3) remove singleton dims, keep only spatial --------------------
        u1 = u3d.squeeze(drop=True)
        v1 = v3d.squeeze(drop=True)

        # 4) sanity check ------------------------------------------------
        if {"node", "nele"}.isdisjoint(u1.dims):
            raise ValueError(
                f"Unexpected dims after reduction: {u1.dims}. "
                "Expect 'node' or 'nele' to remain."
            )

        # 5) interpolate to cell centres --------------------------------
        uc, vc = self._interp_uv_to_centers(u1, v1)

        return uc, vc

    def _auto_skip(self, nele, base=3000):
        """
        Return suitable skip value so that plotted arrows ≈ nele/base.
        """
        import math

        return max(1, int(math.sqrt(nele / base)))

    def _apply_indexer(self, da: xr.DataArray, dim: str, idx):
        """
        Apply idx to dim, choosing isel (positional) or sel (label) automatically.
        * Positional types → .isel(drop=False)
        * Fallback to .sel() if positional fails (IndexError) or if idx is non-positional.
        * If .sel() fails (KeyError), raise a clear ValueError for the user.
        """

        # --- 1) decide whether idx looks positional ------------------
        is_positional = False
        if isinstance(idx, (int, np.integer)):
            is_positional = True
        elif isinstance(idx, slice):
            is_positional = (
                idx.start is None or isinstance(idx.start, (int, np.integer))
            ) and (idx.stop is None or isinstance(idx.stop, (int, np.integer)))
        elif isinstance(idx, Sequence):
            is_positional = all(isinstance(i, (int, np.integer)) for i in idx)

        # --- 2) try positional isel ---------------------------------
        if is_positional:
            try:
                return da.isel({dim: idx}, drop=False)
            except IndexError:
                # positional failed → fall back to label-based sel()
                pass

        # --- 3) label-based sel() with error capture ----------------
        try:
            return da.sel({dim: idx})
        except KeyError as e:
            raise ValueError(
                f"{dim!r} coordinate does not contain label/index {idx!r}. "
                "Specify a valid label or use opts.vec_time (positional index)."
            ) from e

    def _label_to_index(self, label) -> int | None:
        """
        Return the positional index of *label* along self.ds.time.
        Works for integer labels as well as numpy.datetime64 / pandas.Timestamp.
        Returns None if label not found or time coord missing.
        """
        if "time" not in self.ds.coords:
            return None

        arr = self.ds["time"].values  # 1-D array of coordinate values
        dtype_kind = arr.dtype.kind  # 'M' for datetime64, 'i'/'u' for int

        # --- A. numeric coordinate (integer or float) -----------------
        if dtype_kind in ("i", "u", "f"):
            # compare directly; no conversion needed
            idx = np.where(arr == label)[0]
            return int(idx[0]) if idx.size else None

        # --- B. datetime-like coordinate ------------------------------
        if dtype_kind == "M":
            # ensure label is datetime64 for safe comparison
            if np.asarray(label).dtype.kind == "M":
                label_dt = label
            else:
                # fallback: attempt nanosecond resolution
                label_dt = np.datetime64(label, "ns")
            idx = np.where(arr == label_dt)[0]
            return int(idx[0]) if idx.size else None

        # --- C. unsupported coordinate type ---------------------------
        return None

    def _layer_thickness(self, *, siglay=None, time_sel=None):
        """
        Return physical layer thickness (m) on cell centres.
        Dimensions → ('time', 'siglay', 'nele')
        """
        # 1) depth and sea level on node points
        H_node = self.ds["h"]  # (node)
        zeta_node = self.ds["zeta"]  # (time, node)
        if time_sel is not None:
            zeta_node = self._apply_indexer(zeta_node, "time", time_sel)

        # 2) sigma-layer width Δσ  (siglay)
        if "siglay_width" in self.ds:
            d_sigma = self.ds["siglay_width"]
        elif "siglev" in self.ds:
            d_sigma = self.ds["siglev"].diff("siglev").rename({"siglev": "siglay"})
        else:
            n = self.ds.dims["siglay"]
            d_sigma = xr.DataArray(np.ones(n) / n, dims="siglay")

        if siglay is not None:
            d_sigma = self._apply_indexer(d_sigma, "siglay", siglay)

        # 3) physical thickness on node points  → broadcast (time, siglay, node)
        thick_node = (H_node + zeta_node) * d_sigma

        # 4) convert node → cell-centre (nele) by arithmetic mean
        nv = self.ds["nv_zero"].values  # (nele, 3) node indices (0-based)
        thick_cell = (
            thick_node.isel(node=xr.DataArray(nv[:, 0], dims="nele"))
            + thick_node.isel(node=xr.DataArray(nv[:, 1], dims="nele"))
            + thick_node.isel(node=xr.DataArray(nv[:, 2], dims="nele"))
        ) / 3.0  # dims now ('time','siglay','nele')

        return thick_cell

    def _node2cell_mean(self, da_node):
        nv = self.ds["nv_zero"].values  # (nele, 3) node indices
        return (
            da_node.isel(node=xr.DataArray(nv[:, 0], dims="nele"))
            + da_node.isel(node=xr.DataArray(nv[:, 1], dims="nele"))
            + da_node.isel(node=xr.DataArray(nv[:, 2], dims="nele"))
        ) / 3.0


# ------------------------------------------------------------------
# Backward-compat API alias (old notebook examples expect this name)
# ------------------------------------------------------------------
FvcomPlotter.plot_time_series = FvcomPlotter.plot_timeseries  # type: ignore[attr-defined]
