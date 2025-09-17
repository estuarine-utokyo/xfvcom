# ~/Github/xfvcom/xfvcom/plot/markers.py
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ..plot_options import FvcomPlotOptions
from .core import FvcomPlotter


def _resolve_nodes(
    data: Any, *, index_base: int, n_nodes: int
) -> tuple[NDArray[Any], NDArray[Any] | None, list[str]]:
    """
    Return (node_idx[0-based], label_str) resolved from various inputs.

    * If *data* has lon/lat columns ⇒ coords are taken as-is.
    * Otherwise treat values as node indices referring to the mesh.
    """
    # Case 1: DataFrame with lon/lat
    if isinstance(data, pd.DataFrame) and {"lon", "lat"}.issubset(data):
        lon = data["lon"].to_numpy(float)
        lat = data["lat"].to_numpy(float)
        label = (
            data["label"].astype(str).to_list()
            if "label" in data
            else [str(i + index_base) for i in range(len(lon))]
        )
        return lon, lat, label

    # Case 2: sequence of indices
    indices = np.asarray(data, dtype=int)
    if indices.ndim != 1:
        raise ValueError("indices must be 1-D")
    if indices.min() < 0 or indices.max() >= n_nodes:
        raise IndexError("node index out of range")
    label = [str(i + index_base) for i in indices]
    return indices, None, label  # coords are resolved later


def make_node_marker_post(
    nodes: Any,
    plotter: FvcomPlotter,
    *,
    marker_kwargs: Mapping[str, Any] | None = None,
    text_kwargs: Mapping[str, Any] | None = None,
    index_base: int = 0,
    use_latlon: bool | None = None,
    respect_bounds: bool = True,
    marker_clip_buffer: float = 0.0,
    text_clip_buffer: float = 0.0,
) -> Callable[[Axes], None]:
    """Return a post_process_func that plots node markers / labels.

    Parameters
    ----------
    nodes : array-like or DataFrame
        Node indices or coordinates to mark. When providing indices:
        - If index_base=0: use 0-based indices (e.g., [0, 1, 2])
        - If index_base=1: use 1-based indices (e.g., [1, 2, 3])
        The indices should match the numbering system you want displayed.
    plotter : FvcomPlotter
        The plotter instance to get coordinate data from
    marker_kwargs : dict, optional
        Keyword arguments for marker styling
    text_kwargs : dict, optional
        Keyword arguments for text labels
    index_base : int, default 0
        Base for node numbering (0 for Python, 1 for Fortran/FVCOM)
        This affects both input interpretation and label display.
    use_latlon : bool, optional
        Whether to use lat/lon coordinates. If None, will be determined
        from opts parameter when the function is called.
    respect_bounds : bool, default True
        Whether to filter markers to only show those within xlim/ylim bounds.
        When True, markers outside the specified bounds will not be displayed.
    marker_clip_buffer : float, default 0.0
        Buffer zone (in degrees) for marker clipping.
        Positive values include markers slightly outside bounds (show edge markers),
        negative values exclude markers near boundaries.
    text_clip_buffer : float, default 0.0
        Buffer zone (in degrees) for text clipping when using Cartopy.
        Positive values make the clipping area larger (show more text),
        negative values make it smaller (hide text near edges).

    Notes
    -----
    When using Cartopy with geographic coordinates and clip_on=True in text_kwargs,
    this function implements manual text clipping as a workaround for Cartopy's
    known issue with text clipping in geographic projections.

    The marker_clip_buffer and text_clip_buffer allow independent control over
    marker and text visibility near boundaries, useful for dense node displays.
    """

    mkw: dict[str, Any] = {  # marker defaults
        "marker": "o",
        "color": "red",
        "markersize": 3,
        "zorder": 4,
        "clip_on": True,
    } | dict(marker_kwargs or {})

    tkw: dict[str, Any] = {  # text defaults
        "fontsize": 8,
        "color": "yellow",
        "ha": "center",
        "va": "bottom",
        "zorder": 5,
        "clip_on": True,
    } | dict(text_kwargs or {})

    # Store references to coordinate arrays - we'll determine which to use later
    lon_arr = plotter.ds.lon.values
    lat_arr = plotter.ds.lat.values
    x_arr = plotter.ds.x.values if "x" in plotter.ds else lon_arr
    y_arr = plotter.ds.y.values if "y" in plotter.ds else lat_arr
    n_nodes = len(lon_arr)

    # -- resolve input --------------------------------------------------
    # 1) iterable of indices → ndarray[int]
    try:
        idx = np.asarray(nodes, dtype=int)
        if idx.ndim == 1 and idx.size > 0:
            mode = "index"
            # Validate input indices before conversion
            if idx.min() < index_base or idx.max() > n_nodes - 1 + index_base:
                if index_base == 1 and idx.min() == 0:
                    raise IndexError(
                        f"Node indices out of range. You provided 0-based indices "
                        f"(starting from {idx.min()}) but index_base=1 expects 1-based indices. "
                        f"Valid range is 1 to {n_nodes}. "
                        f"Use range(1, {n_nodes + 1}) or nodes=[1, 2, 3, ...]"
                    )
                else:
                    raise IndexError(
                        f"Node indices out of range. With index_base={index_base}, "
                        f"valid range is {index_base} to {n_nodes - 1 + index_base}. "
                        f"You provided indices from {idx.min()} to {idx.max()}."
                    )
            # Convert from user's index_base to 0-based for internal use
            idx = idx - index_base
        else:  # fallback to treat as coordinates
            raise ValueError
    except (TypeError, ValueError):
        # 2) DataFrame or alike with coordinates
        df = pd.DataFrame(nodes)
        # We'll check for both coordinate types and use what's available
        if {"lon", "lat"}.issubset(df):
            lon_direct = df["lon"].to_numpy(float)
            lat_direct = df["lat"].to_numpy(float)
            mode = "coord_lonlat"
        elif {"x", "y"}.issubset(df):
            x_direct = df["x"].to_numpy(float)
            y_direct = df["y"].to_numpy(float)
            mode = "coord_xy"
        else:
            raise ValueError(
                "nodes must be 1-D indices or DataFrame with lon/lat or x/y."
            )
        labels = df.get("label", pd.Series(range(len(df)))).astype(str).to_list()

    # Store the use_latlon preference
    stored_use_latlon = use_latlon
    stored_respect_bounds = respect_bounds
    stored_marker_clip_buffer = marker_clip_buffer
    stored_text_clip_buffer = text_clip_buffer

    # -- post-processor -------------------------------------------------
    def _post(ax: Axes, *, opts: FvcomPlotOptions | None = None, **__) -> None:
        """Executed by `FvcomPlotter.plot_2d`."""
        # Determine which coordinates to use based on opts or stored preference
        if stored_use_latlon is not None:
            use_latlon = stored_use_latlon
        else:
            use_latlon = opts.use_latlon if opts else True

        # Get bounds for filtering if respect_bounds is enabled
        if stored_respect_bounds and opts and (opts.xlim or opts.ylim):
            # Use the bounds from opts (these are always in lat/lon for geographic plots)
            if opts.xlim:
                base_lon_min, base_lon_max = opts.xlim
            else:
                base_lon_min, base_lon_max = float(lon_arr.min()), float(lon_arr.max())

            if opts.ylim:
                base_lat_min, base_lat_max = opts.ylim
            else:
                base_lat_min, base_lat_max = float(lat_arr.min()), float(lat_arr.max())

            # Apply marker buffer for marker bounds
            marker_lon_min = base_lon_min - stored_marker_clip_buffer
            marker_lon_max = base_lon_max + stored_marker_clip_buffer
            marker_lat_min = base_lat_min - stored_marker_clip_buffer
            marker_lat_max = base_lat_max + stored_marker_clip_buffer
        else:
            # No bounds checking
            marker_lon_min, marker_lon_max = -np.inf, np.inf
            marker_lat_min, marker_lat_max = -np.inf, np.inf

        # Check if we need enhanced text clipping for Cartopy
        # This is triggered when:
        # 1. We have text to plot (text_kwargs is not None)
        # 2. clip_on=True is set in text_kwargs
        # 3. We're using geographic coordinates (use_latlon=True)
        # 4. The axes is a Cartopy GeoAxes
        enhanced_text_clip = False
        text_lon_min, text_lon_max = marker_lon_min, marker_lon_max
        text_lat_min, text_lat_max = marker_lat_min, marker_lat_max

        if text_kwargs is not None and tkw.get("clip_on", True) and use_latlon:
            try:
                proj = ax.projection  # Cartopy's GeoAxes has this attr
                if isinstance(proj, ccrs.Projection):
                    enhanced_text_clip = True
                    # Apply buffer for text clipping (independent of marker buffer)
                    if stored_respect_bounds and opts and opts.xlim:
                        text_lon_min = base_lon_min - stored_text_clip_buffer
                        text_lon_max = base_lon_max + stored_text_clip_buffer
                    if stored_respect_bounds and opts and opts.ylim:
                        text_lat_min = base_lat_min - stored_text_clip_buffer
                        text_lat_max = base_lat_max + stored_text_clip_buffer
            except AttributeError:
                pass  # Normal Matplotlib Axes

        if mode == "coord_lonlat":  # DataFrame with lon/lat
            for x, y, lbl in zip(lon_direct, lat_direct, labels, strict=False):
                # Check marker bounds if enabled
                if stored_respect_bounds and not (
                    marker_lon_min <= x <= marker_lon_max
                    and marker_lat_min <= y <= marker_lat_max
                ):
                    continue
                ax.plot(x, y, **_inject_transform(ax, mkw, use_latlon))
                if text_kwargs is not None:
                    # Apply enhanced text clipping if needed
                    if enhanced_text_clip:
                        if (
                            text_lon_min <= x <= text_lon_max
                            and text_lat_min <= y <= text_lat_max
                        ):
                            ax.text(x, y, lbl, **_inject_transform(ax, tkw, use_latlon))
                    else:
                        ax.text(x, y, lbl, **_inject_transform(ax, tkw, use_latlon))
        elif mode == "coord_xy":  # DataFrame with x/y
            for x, y, lbl in zip(x_direct, y_direct, labels, strict=False):
                # For x/y coordinates, we need to check against the corresponding lat/lon
                # This is more complex and would require reverse transformation
                # For now, we'll plot without bounds checking for x/y mode
                ax.plot(x, y, **_inject_transform(ax, mkw, use_latlon))
                if text_kwargs is not None:
                    ax.text(x, y, lbl, **_inject_transform(ax, tkw, use_latlon))
        else:  # indices → lookup coords based on use_latlon
            # Pre-compute coordinates for performance
            coord_x = lon_arr if use_latlon else x_arr
            coord_y = lat_arr if use_latlon else y_arr

            # Vectorized bounds checking for performance
            lon_vals = lon_arr[idx]
            lat_vals = lat_arr[idx]

            # Pre-compute marker visibility mask
            if stored_respect_bounds:
                marker_mask = (
                    (lon_vals >= marker_lon_min)
                    & (lon_vals <= marker_lon_max)
                    & (lat_vals >= marker_lat_min)
                    & (lat_vals <= marker_lat_max)
                )
            else:
                marker_mask = np.ones(len(idx), dtype=bool)

            # Pre-compute text visibility mask for enhanced clipping
            if enhanced_text_clip and text_kwargs is not None:
                text_mask = (
                    (lon_vals >= text_lon_min)
                    & (lon_vals <= text_lon_max)
                    & (lat_vals >= text_lat_min)
                    & (lat_vals <= text_lat_max)
                )
            else:
                text_mask = (
                    marker_mask  # Use same as marker mask if no enhanced clipping
                )

            for j, i in enumerate(idx):
                # Check marker visibility using pre-computed mask
                if not marker_mask[j]:
                    continue

                x, y = coord_x[i], coord_y[i]
                lbl = str(i + index_base)
                ax.plot(x, y, **_inject_transform(ax, mkw, use_latlon))

                if text_kwargs is not None:
                    # Apply text clipping based on the appropriate mask
                    if enhanced_text_clip:
                        if text_mask[j]:  # Use enhanced text clipping mask
                            ax.text(x, y, lbl, **_inject_transform(ax, tkw, use_latlon))
                    elif marker_mask[j]:  # Text follows marker visibility
                        ax.text(x, y, lbl, **_inject_transform(ax, tkw, use_latlon))

    return _post


def _inject_transform(
    ax: Axes, kw: dict[str, Any], use_latlon: bool = True
) -> dict[str, Any]:
    """Add PlateCarree transform only when `ax` is a Cartopy GeoAxes and using lat/lon coordinates."""
    if "transform" in kw:
        return kw  # user already supplied

    # If we're using Cartesian coordinates, don't inject any transform
    if not use_latlon:
        return kw

    try:
        proj = ax.projection  # Cartopy's GeoAxes has this attr
    except AttributeError:
        return kw  # Normal Matplotlib Axes
    else:
        import cartopy.crs as ccrs

        if isinstance(proj, ccrs.Projection):
            # When use_latlon=True, we're always using geographic coordinates
            # which need PlateCarree transform for any projection
            return {**kw, "transform": ccrs.PlateCarree()}
        return kw


__all__ = ["make_node_marker_post"]
