# ~/Github/xfvcom/xfvcom/plot/markers.py
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from ..plot_options import FvcomPlotOptions
from .core import FvcomPlotter


def _resolve_nodes(
    data: Any, *, index_base: int, n_nodes: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
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
    label_fmt: str = "{idx}",
    index_base: int = 0,
) -> Callable[[Axes], None]:
    """
    Factory: return post_process_func that plots node markers / labels.

    Parameters
    ----------
    nodes : DataFrame | sequence[int] | DataArray | ndarray
        * DataFrame ⇒ must include ``lon`` / ``lat`` (deg) and optional ``label``.
        * Otherwise ⇒ treated as node indices referring to plotter.ds.*
    plotter : FvcomPlotter
        Provides lon/lat arrays for mesh look-up.
    marker_kwargs, text_kwargs : mapping, optional
        Additional kwargs passed to ``ax.plot`` / ``ax.text``.
    label_fmt : str, default "{idx}"
        Format string.  Available fields: ``idx`` (0-based int).
    index_base : int, default 0
        Added to ``idx`` inside ``label_fmt`` when formatting.

    Returns
    -------
    Callable[[Axes, ...], None]
        Compatible with ``post_process_func`` in plot_2d.
    """
    transform = ccrs.PlateCarree()
    mkw_base: dict[str, Any] = {
        "marker": "o",
        "color": "red",
        "markersize": 3,
        "transform": transform,
        "zorder": 4,
        "clip_on": True,
    }
    mkw = mkw_base | dict(marker_kwargs or {})

    tkw_base: dict[str, Any] = {
        "fontsize": 7,
        "color": "yellow",
        "ha": "center",
        "va": "bottom",
        "transform": transform,
        "zorder": 5,
        "clip_on": True,
    }
    tkw = tkw_base | dict(text_kwargs or {})

    # mesh coordinates
    lon_arr = plotter.ds.lon.values
    lat_arr = plotter.ds.lat.values

    # resolve inputs to indices & labels
    lon_direct, lat_direct, label = _resolve_nodes(
        nodes, index_base=index_base, n_nodes=len(lon_arr)
    )

    def _post(ax: Axes, *, opts: FvcomPlotOptions | None = None, **__) -> None:
        """Post-processor executed by plot_2d."""
        if lon_direct is not None:  # lon/lat already resolved
            for x, y, txt in zip(lon_direct, lat_direct, label, strict=False):
                ax.plot(x, y, **mkw)
                ax.text(x, y, txt, **tkw)
        else:  # indices ⇒ lookup coordinates
            for idx, txt in zip(lon_direct, label, strict=False):
                ax.plot(lon_arr[idx], lat_arr[idx], **mkw)
                ax.text(lon_arr[idx], lat_arr[idx], txt, **tkw)

    return _post


__all__ = ["make_node_marker_post"]
