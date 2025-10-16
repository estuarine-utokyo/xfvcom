import numpy as np
import pandas as pd
import xarray as xr

from xfvcom.plot.config import FvcomPlotConfig
from xfvcom.plot.core import FvcomPlotter
from xfvcom.plot_options import FvcomPlotOptions


def _dummy_ds():
    time = pd.date_range("2020-01-01", periods=24, freq="h")
    siglay = np.arange(3)
    node = np.arange(4)
    temp = xr.DataArray(
        np.random.rand(len(time), len(siglay), len(node)),
        coords={"time": time, "siglay": siglay, "node": node},
        dims=("time", "siglay", "node"),
        name="temp",
        attrs={"long_name": "Temperature", "units": "°C"},
    )
    ds = xr.Dataset({"temp": temp})
    # 必要最小限のメッシュ変数を追加
    ds["lon"] = xr.DataArray(np.random.rand(len(node)), dims="node")
    ds["lat"] = xr.DataArray(np.random.rand(len(node)), dims="node")
    ds["nv_zero"] = xr.DataArray(
        np.vstack([node[:-1], node[1:], node[1:]]).T, dims=("nele", "three")
    )
    return ds


def test_scalar_time_mean():
    ds = _dummy_ds()
    plotter = FvcomPlotter(ds, FvcomPlotConfig())
    da = ds["temp"]
    opts = FvcomPlotOptions(
        scalar_time=slice("2020-01-01", "2020-01-01T11:00"),
        scalar_reduce={"time": "mean"},
    )
    reduced = plotter._reduce_scalar(
        da, time_sel=opts.scalar_time, reduce=opts.scalar_reduce
    )
    assert "time" not in reduced.dims
    # 平均後の値は区間データの平均になっていること
    manual = da.sel(time=opts.scalar_time).mean("time")
    np.testing.assert_allclose(reduced.values, manual.values)


def test_scalar_siglay_mean():
    ds = _dummy_ds()
    plotter = FvcomPlotter(ds, FvcomPlotConfig())
    da = ds["temp"]
    opts = FvcomPlotOptions(
        scalar_siglay=slice(0, 1),
        scalar_reduce={"siglay": "mean"},
    )
    reduced = plotter._reduce_scalar(
        da, siglay_sel=opts.scalar_siglay, reduce=opts.scalar_reduce
    )
    assert "siglay" not in reduced.dims
    manual = da.sel(siglay=opts.scalar_siglay).mean("siglay")
    np.testing.assert_allclose(reduced.values, manual.values)
