"""Time-averaged dye concentration contour maps.

This module provides tools for creating horizontal contour maps showing
time-averaged dye concentration from individual sources at specified
vertical layers.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ..dye_timeseries import decode_fvcom_time
from ..ensemble_analysis import get_source_name
from ..plot_options import FvcomPlotOptions
from .config import FvcomPlotConfig
from .core import FvcomPlotter

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def load_dye_time_avg(
    output_dir: str | Path,
    year: int,
    basename: str,
    member: int,
    sigmas: list[int] | None = None,
    depth_average: bool = False,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    var_name: str = "DYE",
) -> xr.Dataset:
    """Load and compute time-averaged dye concentration for a source.

    Parameters
    ----------
    output_dir : str or Path
        Base output directory (e.g., 'goto2023/dye_run/output')
    year : int
        Simulation year (e.g., 2021)
    basename : str
        Case basename (e.g., 'tb_w18_r16')
    member : int
        Member ID (1-18 for individual sources)
    sigmas : list of int, optional
        Sigma layer indices (0-based). If None, uses all layers.
    depth_average : bool
        If True, compute depth-averaged concentration (ignores sigmas).
    start : str or pd.Timestamp, optional
        Start time for averaging window.
    end : str or pd.Timestamp, optional
        End time for averaging window.
    var_name : str
        Variable name in NetCDF files (default: 'DYE')

    Returns
    -------
    xr.Dataset
        Dataset containing:
        - 'DYE_time_avg': time-averaged dye (dims: siglay, node) or (node,) if depth_average
        - 'lon', 'lat', 'x', 'y': node coordinates
        - 'nv': triangle connectivity
        - Attributes with metadata (source name, time range, etc.)
    """
    output_dir = Path(output_dir)

    # Construct path to member output directory
    member_dir = output_dir / str(year) / str(member)

    # Find NetCDF files (exclude restart files)
    pattern = f"{basename}_{year}_{member}_*.nc"
    files = sorted([f for f in member_dir.glob(pattern) if "restart" not in f.name])

    if not files:
        raise FileNotFoundError(f"No files found for member {member} in {member_dir}")

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
    ds = decode_fvcom_time(ds, time_key="time")

    # Extract DYE variable
    if var_name not in ds:
        ds.close()
        raise ValueError(f"Variable '{var_name}' not found for member {member}")

    dye = ds[var_name]

    # Apply time window
    if start is not None or end is not None:
        time_vals = pd.DatetimeIndex(ds.time.values)
        if start is not None:
            start_ts = pd.Timestamp(start)
            mask_start = time_vals >= start_ts
        else:
            mask_start = np.ones(len(time_vals), dtype=bool)
            start_ts = time_vals.min()

        if end is not None:
            end_ts = pd.Timestamp(end)
            mask_end = time_vals <= end_ts
        else:
            mask_end = np.ones(len(time_vals), dtype=bool)
            end_ts = time_vals.max()

        time_mask = mask_start & mask_end
        time_indices = np.where(time_mask)[0]
        dye = dye.isel(time=time_indices)
    else:
        start_ts = pd.Timestamp(ds.time.values[0])
        end_ts = pd.Timestamp(ds.time.values[-1])

    # Compute time average
    if depth_average:
        # Depth-averaged: average over time and siglay
        dye_avg = dye.mean(dim=["time", "siglay"])
        dye_avg.attrs["averaging"] = "depth_average"
    else:
        # Time average, keep siglay dimension
        dye_avg = dye.mean(dim="time")

        # Select specific sigma layers if specified
        if sigmas is not None:
            dye_avg = dye_avg.isel(siglay=sigmas)
            if len(sigmas) == 1:
                # Squeeze single layer but track which layer
                dye_avg = dye_avg.squeeze("siglay", drop=False)
            dye_avg.attrs["siglay_indices"] = sigmas
        dye_avg.attrs["averaging"] = "time_average"

    # Get source name
    source_name = get_source_name(member, style="short")

    # Create result dataset with coordinates
    result = xr.Dataset(
        {
            "DYE_time_avg": dye_avg,
        },
        coords={
            "lon": ds["lon"],
            "lat": ds["lat"],
            "x": ds["x"],
            "y": ds["y"],
        },
    )

    # Add triangle connectivity
    result["nv"] = ds["nv"]

    # Add metadata
    result.attrs["source_name"] = source_name
    result.attrs["member_id"] = member
    result.attrs["year"] = year
    result.attrs["time_start"] = str(start_ts)
    result.attrs["time_end"] = str(end_ts)
    result.attrs["n_timesteps"] = len(dye.time)

    ds.close()

    return result


def plot_dye_contour(
    ds: xr.Dataset,
    siglay: int | None = None,
    output: str | Path | None = None,
    figsize: tuple[float, float] = (12, 10),
    vmin: float | None = None,
    vmax: float | None = None,
    levels: int | list[float] = 20,
    cmap: str = "viridis",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    fontsize_title: float = 14,
    fontsize_label: float = 12,
    fontsize_tick: float = 10,
    dpi: int = 300,
    with_mesh: bool = False,
    coastlines: bool = True,
    add_colorbar: bool = True,
    cbar_label: str | None = None,
    ax: Axes | None = None,
) -> dict:
    """Create contour map of time-averaged dye concentration.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from load_dye_time_avg() containing 'DYE_time_avg'
    siglay : int, optional
        Sigma layer index to plot (if DYE_time_avg has siglay dimension).
        If None and data has multiple layers, uses layer 0.
    output : str or Path, optional
        Output file path. If provided, saves figure.
    figsize : tuple
        Figure size (width, height) in inches.
    vmin, vmax : float, optional
        Value range for colorbar. If None, auto-determined from data.
    levels : int or list
        Number of contour levels or explicit level values.
    cmap : str
        Matplotlib colormap name.
    xlim, ylim : tuple, optional
        Longitude/latitude bounds (xmin, xmax), (ymin, ymax).
        Supports degree-minute notation (e.g., "139:30", "35:15").
    title : str, optional
        Plot title. If None, auto-generated from dataset metadata.
    fontsize_title : float
        Title font size.
    fontsize_label : float
        Axis label font size.
    fontsize_tick : float
        Tick label font size.
    dpi : int
        Output resolution.
    with_mesh : bool
        If True, overlay mesh lines.
    coastlines : bool
        If True, draw coastlines.
    add_colorbar : bool
        If True, add colorbar.
    cbar_label : str, optional
        Colorbar label. If None, uses "Dye Concentration".
    ax : Axes, optional
        Existing axes to plot on. If None, creates new figure.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'fig': matplotlib Figure object
        - 'ax': matplotlib Axes object
        - 'output_path': Path to saved file (if output provided)
    """
    # Get data array
    da = ds["DYE_time_avg"]

    # Handle siglay dimension if present
    if "siglay" in da.dims:
        if siglay is None:
            siglay = 0
            logger.info(f"Using default siglay={siglay} (surface)")
        da = da.isel(siglay=siglay)
        layer_info = f"Layer {siglay}"
    else:
        layer_info = ds.attrs.get("averaging", "")
        if layer_info == "depth_average":
            layer_info = "Depth-averaged"
        elif "siglay_indices" in ds.attrs:
            indices = ds.attrs["siglay_indices"]
            if len(indices) == 1:
                layer_info = f"Layer {indices[0]}"
            else:
                layer_info = f"Layers {indices[0]}-{indices[-1]}"

    # Generate title if not provided
    if title is None:
        source_name = ds.attrs.get("source_name", "Unknown")
        time_start = ds.attrs.get("time_start", "?")
        time_end = ds.attrs.get("time_end", "?")
        # Format dates
        try:
            t_start = pd.Timestamp(time_start).strftime("%Y-%m-%d")
            t_end = pd.Timestamp(time_end).strftime("%Y-%m-%d")
        except Exception:
            t_start = time_start
            t_end = time_end
        title = f"Time-Averaged Dye: {source_name} ({layer_info})\n{t_start} to {t_end}"

    # Prepare xr.Dataset for FvcomPlotter
    # Need to add nv_zero (0-based connectivity)
    ds_plot = ds.copy()
    nv = ds_plot["nv"].values
    if nv.min() >= 1:
        # Convert from 1-based to 0-based
        ds_plot["nv_zero"] = (("nele", "three"), nv.T - 1)
    else:
        ds_plot["nv_zero"] = (("nele", "three"), nv.T)

    # Also need nv_ccw for validation (counter-clockwise)
    ds_plot["nv_ccw"] = ds_plot["nv_zero"]

    # Create FvcomPlotOptions
    opts = FvcomPlotOptions(
        figsize=figsize,
        xlim=xlim,
        ylim=ylim,
        vmin=vmin,
        vmax=vmax,
        levels=levels,
        cmap=cmap,
        with_mesh=with_mesh,
        coastlines=coastlines,
        colorbar=add_colorbar,
        title=title,
        use_latlon=True,
    )
    opts.extra["cbar_label"] = cbar_label or "Dye Concentration"

    # Set scalar reduction - data is already 2D (node only)
    # We don't need to reduce further

    # Create plotter with proper config
    plot_cfg = FvcomPlotConfig(
        figsize=figsize,
        fontsize={
            "title": fontsize_title,
            "xlabel": fontsize_label,
            "ylabel": fontsize_label,
        },
    )
    plotter = FvcomPlotter(ds_plot, plot_config=plot_cfg)

    # Call plot_2d (pass ax via keyword so the @precedence decorator handles it)
    ax_out = plotter.plot_2d(
        da=da,
        opts=opts,
        ax=ax,
    )

    # Get figure
    if ax_out is not None:
        fig = ax_out.figure
    else:
        fig = plt.gcf()

    # Apply font sizes to ticks
    if ax_out is not None:
        ax_out.tick_params(axis="both", labelsize=fontsize_tick)

    # Save if output provided
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved: {output}")

    return {
        "fig": fig,
        "ax": ax_out,
        "output_path": output,
    }


def create_dye_contour_maps(
    output_dir: str | Path,
    year: int,
    basename: str,
    members: list[int],
    sigmas: list[int] | None = None,
    depth_average: bool = False,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    output_pattern: str | None = None,
    **kwargs,
) -> list[dict]:
    """Create contour maps for multiple sources.

    This is the main entry point for batch creation of dye contour maps.

    Parameters
    ----------
    output_dir : str or Path
        Base output directory (e.g., 'goto2023/dye_run/output')
    year : int
        Simulation year (e.g., 2021)
    basename : str
        Case basename (e.g., 'tb_w18_r16')
    members : list of int
        Member IDs to process (e.g., [1, 2, 3] or range(1, 19))
    sigmas : list of int, optional
        Sigma layer indices (0-based). If None, uses surface layer [0].
    depth_average : bool
        If True, compute depth-averaged concentration.
    start : str or pd.Timestamp, optional
        Start time for averaging window.
    end : str or pd.Timestamp, optional
        End time for averaging window.
    output_pattern : str, optional
        Output filename pattern. Can include placeholders:
        {member}, {source_name}, {year}, {siglay}.
        Default: 'dye_contour_{source_name}_k{siglay}_{year}.png'
    **kwargs
        Additional arguments passed to plot_dye_contour()

    Returns
    -------
    list of dict
        List of result dictionaries from plot_dye_contour()
    """
    print("=" * 70, file=sys.stdout)
    print("DYE CONCENTRATION CONTOUR MAPS", file=sys.stdout)
    print("=" * 70, file=sys.stdout)

    output_dir = Path(output_dir)

    # Default sigma layers
    if sigmas is None and not depth_average:
        sigmas = [0]  # Surface layer

    # Default output pattern
    if output_pattern is None:
        if depth_average:
            output_pattern = "dye_contour_{source_name}_depth_avg_{year}.png"
        else:
            output_pattern = "dye_contour_{source_name}_k{siglay}_{year}.png"

    results = []

    for member in members:
        source_name = get_source_name(member, style="short")
        print(f"\nProcessing member {member}: {source_name}", file=sys.stdout)

        try:
            # Load data
            ds = load_dye_time_avg(
                output_dir=output_dir,
                year=year,
                basename=basename,
                member=member,
                sigmas=sigmas,
                depth_average=depth_average,
                start=start,
                end=end,
            )

            # Determine layers to plot
            if depth_average:
                layers_to_plot = [None]
            elif sigmas is not None:
                layers_to_plot = sigmas
            else:
                layers_to_plot = [0]

            for siglay in layers_to_plot:
                # Generate output path
                output_file = output_pattern.format(
                    member=member,
                    source_name=source_name,
                    year=year,
                    siglay=siglay if siglay is not None else "avg",
                )

                # If output_file is not absolute, make it relative to plot dir
                output_path = Path(output_file)
                if not output_path.is_absolute():
                    # Assume we want to save in a 'png' subdirectory
                    output_path = Path(output_file)

                # Create plot
                result = plot_dye_contour(
                    ds=ds,
                    siglay=siglay if not depth_average else None,
                    output=output_path,
                    **kwargs,
                )
                result["member"] = member
                result["source_name"] = source_name
                result["siglay"] = siglay
                results.append(result)

                # Close figure to free memory
                plt.close(result["fig"])

            ds.close()

        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            continue

    print(f"\n{'=' * 70}", file=sys.stdout)
    print(f"Created {len(results)} contour maps", file=sys.stdout)
    print("=" * 70, file=sys.stdout)

    return results
