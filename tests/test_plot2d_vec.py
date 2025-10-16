# tests/test_plot2d_vec.py
import matplotlib

matplotlib.use("Agg")  # headless backend

import numpy as np
import pandas as pd
import xarray as xr

from xfvcom.plot.config import FvcomPlotConfig
from xfvcom.plot.core import FvcomPlotter
from xfvcom.plot_options import FvcomPlotOptions


def _tiny_ds() -> xr.Dataset:
    """Return the smallest FVCOM-like dataset that plot_2d() can handle."""
    # --- base axes ----------------------------------------------------
    time = pd.date_range("2020-01-01 01:00", periods=3, freq="6h")
    siglay = [0]  # single layer
    node = [0, 1, 2]  # three nodes → one triangle (0,1,2)

    # --- coordinates --------------------------------------------------
    lon = xr.DataArray([0.0, 1.0, 0.0], dims=("node",))
    lat = xr.DataArray([35.0, 35.0, 36.0], dims=("node",))
    # cell centre = simple mean of vertices
    lonc = xr.DataArray([lon.mean().item()], dims=("nele",))
    latc = xr.DataArray([lat.mean().item()], dims=("nele",))

    # --- connectivity: 1-based (nv) & 0-based (nv_zero/nv_ccw) -------
    nv_zero = np.array([[0, 1, 2]], dtype="i4")  # (nele,3)
    nv = xr.DataArray(nv_zero + 1, dims=("nele", "three"))
    nv_ccw = nv  # for verbose check

    # --- variables ----------------------------------------------------
    temp = xr.DataArray(
        np.arange(len(time) * len(siglay) * len(node)).reshape(
            len(time), len(siglay), len(node)
        ),
        dims=("time", "siglay", "node"),
        coords={"time": time, "siglay": siglay, "node": node},
        name="temp",
        attrs={"long_name": "temperature", "units": "°C"},
    )
    u = xr.zeros_like(temp) + 0.1
    v = xr.zeros_like(temp) + 0.2

    ds = xr.Dataset(
        {
            "temp": temp,
            "u": u,
            "v": v,
            "lon": lon,
            "lat": lat,
            "lonc": lonc,
            "latc": latc,
            "nv_zero": (("nele", "three"), nv_zero),
            "nv": nv,
            "nv_ccw": nv_ccw,
        }
    )
    return ds


def test_plot2d_with_vectors(tmp_path):
    ds = _tiny_ds()
    cfg = FvcomPlotConfig()
    plotter = FvcomPlotter(ds, cfg)

    # scalar DataArray (time=1) with vector overlay
    da_scalar = ds["temp"].isel(time=1, siglay=0)
    opts = FvcomPlotOptions(plot_vec2d=True, vec_siglay=0)

    ax = plotter.plot_2d(da=da_scalar, opts=opts)
    assert ax is not None

    # basic sanity: at least one contour/quad collection or a title
    assert ax.collections or ax.get_title()

    # save for manual inspection (optional)
    ax.figure.savefig(tmp_path / "plot2d_vec.png", dpi=80)
    matplotlib.pyplot.close(ax.figure)
