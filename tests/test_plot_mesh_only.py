"""
Unit test: ensure FvcomPlotter.plot_2d(da=None, with_mesh=True)
works without errors (used internally by plot_vector2d).
"""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in linestrings",
    category=RuntimeWarning,
    module=r"shapely\..*",
)
import numpy as np
import xarray as xr

from xfvcom.plot.config import FvcomPlotConfig
from xfvcom.plot.core import FvcomPlotter
from xfvcom.plot_options import FvcomPlotOptions


def _dummy_mesh_ds() -> xr.Dataset:
    node = np.arange(6)

    lon = np.random.rand(len(node)) * 1.0e-3 + node.astype(float)
    lat = np.random.rand(len(node)) * 1.0e-3 + node.astype(float)

    ds = xr.Dataset()
    ds["lon"] = ("node", lon)
    ds["lat"] = ("node", lat)

    # non-degenerate triangles (i, i+1, i+2)
    nv = np.vstack([node[:-2], node[1:-1], node[2:]]).T  # shape (nele, 3)
    ds["nv_zero"] = (("nele", "three"), nv)
    return ds


def test_plot_mesh_only() -> None:
    """
    plot_2d(da=None, with_mesh=True) should run without raising
    and return a Matplotlib Axes instance.
    """
    ds = _dummy_mesh_ds()
    plotter = FvcomPlotter(ds, FvcomPlotConfig())
    opts = FvcomPlotOptions(with_mesh=True)

    ax = plotter.plot_2d(da=None, opts=opts)
    assert ax is not None
