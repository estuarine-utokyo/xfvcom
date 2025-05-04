# tests/test_plot2d_vec.py
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import xarray as xr

from xfvcom.plot.core import FvcomPlotConfig, FvcomPlotOptions, FvcomPlotter


def _tiny_ds() -> xr.Dataset:
    """Return a 3-time-step, 1-layer, 2-node toy dataset."""
    time = pd.date_range("2020-01-01 01:00", periods=3, freq="6h")
    da = xr.DataArray(
        np.arange(3 * 1 * 2).reshape(3, 1, 2),  # temp
        coords={"time": time, "siglay": [0], "node": [0, 1]},
        dims=("time", "siglay", "node"),
        name="temp",
        attrs={"units": "°C", "long_name": "temperature"},
    )
    ds = da.to_dataset()

    # --- minimal lon/lat coords required by plot_2d -----------------
    ds["lon"] = xr.DataArray([0.0, 1.0], dims=("node",))
    ds["lat"] = xr.DataArray([35.0, 35.5], dims=("node",))

    # Velocity components
    ds["u"] = xr.zeros_like(da) + 0.1
    ds["v"] = xr.zeros_like(da) + 0.2
    return ds


def test_plot2d_with_vectors(tmp_path):
    ds = _tiny_ds()
    cfg = FvcomPlotConfig()
    plotter = FvcomPlotter(ds, cfg)

    # scalar DA (time=1)
    da_scalar = ds["temp"].isel(time=1, siglay=0)
    opts = FvcomPlotOptions(plot_vec2d=True, vec_siglay=0)

    fig, ax, _ = plotter.plot_2d(da=da_scalar, opts=opts)
    # Simple verification
    assert ax.get_title() != ""

    # For debug purpose
    fig.savefig(tmp_path / "plot2d_vec.png", dpi=80)
    matplotlib.pyplot.close(fig)
