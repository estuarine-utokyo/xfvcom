import numpy as np
import xarray as xr

from xfvcom.plot.config import FvcomPlotConfig
from xfvcom.plot.core import FvcomPlotter
from xfvcom.plot_options import FvcomPlotOptions


def test_plot_2d_without_long_name():
    # dummy 2-D field (node)
    node = np.arange(10)
    da = xr.DataArray(
        np.random.rand(len(node)),
        dims="node",
        coords={"node": node},
        name="temp",  # name だけ
        # attrs intentionally empty
    )
    ds = xr.Dataset({"temp": da})
    ds["lon"] = ("node", np.random.rand(len(node)))
    ds["lat"] = ("node", np.random.rand(len(node)))
    nv = np.vstack([node[:-1], node[1:], node[1:]]).T  # (nele, three)
    ds["nv_zero"] = (("nele", "three"), nv)

    plotter = FvcomPlotter(ds, FvcomPlotConfig())
    opts = FvcomPlotOptions()  # no reduce necessary

    # Should not raise AttributeError
    ax = plotter.plot_2d(da=ds["temp"], opts=opts)
    assert ax is not None
