from __future__ import annotations

import numpy as np
import xarray as xr

from xfvcom import make_node_marker_post
from xfvcom.plot.core import FvcomPlotConfig, FvcomPlotter
from xfvcom.plot_options import FvcomPlotOptions


def _dummy_mesh_ds() -> xr.Dataset:
    node = np.arange(4)
    x = node.astype(float)
    y = node.astype(float)
    nv = np.vstack([node[:-2], node[1:-1], node[2:]]).T

    ds = xr.Dataset()
    ds["x"] = ("node", x)
    ds["y"] = ("node", y)
    ds["lon"] = ("node", x)
    ds["lat"] = ("node", y)
    ds["nv_zero"] = (("nele", "three"), nv)
    return ds


def test_node_marker_cartesian() -> None:
    ds = _dummy_mesh_ds()
    plotter = FvcomPlotter(ds, FvcomPlotConfig())
    opts = FvcomPlotOptions(use_latlon=False, with_mesh=True)
    pp = make_node_marker_post(range(ds.dims["node"]), plotter)

    ax = plotter.plot_2d(da=None, opts=opts, post_process_func=pp)
    assert ax is not None
