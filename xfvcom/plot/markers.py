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
) -> Callable[[Axes], None]:
    """Return a post_process_func that plots node markers / labels."""

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

    lon_arr = plotter.ds.lon.values
    lat_arr = plotter.ds.lat.values

    # -- resolve input --------------------------------------------------
    # 1) iterable of indices → ndarray[int]
    try:
        idx = np.asarray(nodes, dtype=int)
        if idx.ndim == 1 and idx.size > 0:
            mode = "index"
        else:  # fallback to treat as lon/lat
            raise ValueError
    except (TypeError, ValueError):
        # 2) DataFrame or alike with lon/lat
        df = pd.DataFrame(nodes)
        if not {"lon", "lat"}.issubset(df):
            raise ValueError("nodes must be 1-D indices or DataFrame with lon/lat.")
        mode = "coord"
        lon_direct = df["lon"].to_numpy(float)
        lat_direct = df["lat"].to_numpy(float)
        labels = df.get("label", pd.Series(range(len(df)))).astype(str).to_list()

    # -- post-processor -------------------------------------------------
    def _post(ax: Axes, *, opts: FvcomPlotOptions | None = None, **__) -> None:
        """Executed by `FvcomPlotter.plot_2d`."""
        if mode == "coord":  # A) lon/lat already resolved
            for x, y, lbl in zip(lon_direct, lat_direct, labels, strict=False):
                ax.plot(x, y, **_inject_transform(ax, mkw))
                ax.text(x, y, lbl, **_inject_transform(ax, tkw))
        else:  # B) indices → lookup coords
            for i in idx:
                x, y = lon_arr[i], lat_arr[i]
                lbl = str(i + index_base)
                ax.plot(x, y, **_inject_transform(ax, mkw))
                ax.text(x, y, lbl, **_inject_transform(ax, tkw))

    return _post


def _inject_transform(ax: Axes, kw: dict[str, Any]) -> dict[str, Any]:
    """Add PlateCarree transform only when `ax` is a Cartopy GeoAxes."""
    if "transform" in kw:
        return kw  # user already supplied
    try:
        proj = ax.projection  # Cartopy's GeoAxes has this attr
    except AttributeError:
        return kw  # Normal Matplotlib Axes
    else:
        import cartopy.crs as ccrs

        if isinstance(proj, ccrs.Projection):
            return {**kw, "transform": ccrs.PlateCarree()}
        return kw


__all__ = ["make_node_marker_post"]
