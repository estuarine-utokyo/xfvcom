import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xfvcom.plot.core import FvcomPlotConfig, FvcomPlotter
from xfvcom.plot_options import FvcomPlotOptions


def _dummy_3d_ds():
    """Minimal 3-D dataset with time, siglay, node for unit test."""
    time = pd.date_range("2020-01-01", periods=4, freq="h")
    siglay = np.arange(3)
    node = np.arange(5)
    temp = xr.DataArray(
        np.random.rand(len(time), len(siglay), len(node)),
        coords={"time": time, "siglay": siglay, "node": node},
        dims=("time", "siglay", "node"),
        name="temp",
    )
    ds = xr.Dataset({"temp": temp})
    ds["lon"] = ("node", np.random.rand(len(node)))
    ds["lat"] = ("node", np.random.rand(len(node)))
    # simple connectivity (nele = node-1)
    nv = np.vstack([node[:-1], node[1:], node[1:]]).T
    ds["nv_zero"] = (("nele", "three"), nv)
    return ds


def test_plot_2d_error_when_siglay_left():
    ds = _dummy_3d_ds()
    plotter = FvcomPlotter(ds, FvcomPlotConfig())
    opts = FvcomPlotOptions(scalar_reduce={"time": "mean"})  # siglay 未縮約
    with pytest.raises(ValueError, match=r"siglay.*dimension"):
        plotter.plot_2d(da=ds["temp"], opts=opts)
