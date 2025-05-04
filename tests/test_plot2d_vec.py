# tests/test_plot2d_vec.py
import matplotlib  # ← GUI を使わないように

matplotlib.use("Agg")  # (バックエンドは一度だけ設定)

import numpy as np
import pandas as pd
import xarray as xr

from xfvcom.plot.core import FvcomPlotOptions, FvcomPlotter


def _tiny_ds() -> xr.Dataset:
    """Return a 3-time-step, 1-layer, 2-node toy dataset."""
    time = pd.date_range("2020-01-01 01:00", periods=3, freq="6H")
    da = xr.DataArray(
        np.arange(3 * 1 * 2).reshape(3, 1, 2),  # temp
        coords={"time": time, "siglay": [0], "node": [0, 1]},
        dims=("time", "siglay", "node"),
        name="temp",
        attrs={"units": "°C", "long_name": "temperature"},
    )
    ds = da.to_dataset()
    # 速度成分（同形状でゼロ）
    ds["u"] = xr.zeros_like(da) + 0.1
    ds["v"] = xr.zeros_like(da) + 0.2
    return ds


def test_plot2d_with_vectors(tmp_path):
    ds = _tiny_ds()
    plotter = FvcomPlotter(ds)

    # scalar DA (time=1) にベクトル重畳を要求
    da_scalar = ds["temp"].isel(time=1, siglay=0)
    opts = FvcomPlotOptions(plot_vec2d=True, vec_siglay=0)

    fig, ax, _ = plotter.plot_2d(da=da_scalar, opts=opts)
    # タイトルが設定されていれば「描画に成功した」程度の簡易検証
    assert ax.get_title() != ""

    # 保存して目視デバッグ用 artefact を残せるようにしておく（任意）
    fig.savefig(tmp_path / "plot2d_vec.png", dpi=80)
    matplotlib.pyplot.close(fig)
