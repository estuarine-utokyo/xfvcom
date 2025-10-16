from __future__ import annotations

import shutil
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pytest
from matplotlib.testing.compare import compare_images

pytestmark = pytest.mark.png

from xfvcom.plot.core import FvcomPlotOptions, FvcomPlotter

BASELINE = Path(__file__).parent / "baseline"
BASELINE.mkdir(exist_ok=True)


def _assert_or_regen(tmp_png: Path, baseline_name: str, regen: bool) -> None:
    ref = BASELINE / baseline_name
    if regen or not ref.exists():
        shutil.copy(tmp_png, ref)
    else:
        diff = compare_images(str(ref), str(tmp_png), tol=1.0)  # type: ignore[call-overload]
        assert diff is None, diff


@pytest.mark.parametrize("use_latlon", [True])  # Cartesian は除外
def test_prepare_axis_regression(
    tmp_path: Path,
    fvcom_ds,
    plotter: FvcomPlotter,
    regen_baseline: bool,
    use_latlon: bool,
) -> None:
    # --------------------------------------------------
    # 1) ダミー DS に x, y を補完（安全のため）
    # --------------------------------------------------
    if "x" not in fvcom_ds:
        fvcom_ds["x"] = ("node", np.arange(len(fvcom_ds["lon"])))
        fvcom_ds["y"] = ("node", np.arange(len(fvcom_ds["lat"])))

    # --------------------------------------------------
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection=(ccrs.PlateCarree() if use_latlon else None))

    mtri.Triangulation(fvcom_ds["lon"], fvcom_ds["lat"], fvcom_ds["nv_zero"])
    da = fvcom_ds["temp"].isel(time=0, siglay=0)

    opts = FvcomPlotOptions(
        use_latlon=use_latlon,
        with_mesh=True,
        plot_vec2d=False,
        coastlines=False,
        obclines=False,
    )

    plotter.plot_2d(da=da, ax=ax, opts=opts)

    case = "scalar_geo"  # いまは地理座標のみ
    out_png = tmp_path / f"{case}.png"
    fig.savefig(out_png, dpi=100)
    _assert_or_regen(out_png, f"{case}.png", regen_baseline)
