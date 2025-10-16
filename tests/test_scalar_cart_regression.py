"""Cartesian mesh + scalar regression."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pytest
from matplotlib.testing.compare import compare_images

pytestmark = pytest.mark.png

from xfvcom.plot.core import FvcomPlotOptions

BASELINE = Path(__file__).parent / "baseline"
BASELINE.mkdir(exist_ok=True)


def _cmp_or_copy(src: Path, name: str, regen: bool) -> None:
    ref = BASELINE / name
    if regen or not ref.exists():
        shutil.copy(src, ref)
    else:
        diff = compare_images(str(ref), str(src), tol=1.0)  # type: ignore[call-overload]
        assert diff is None, diff


def test_scalar_cart_regression(tmp_path, fvcom_ds, plotter, regen_baseline):
    # x, y を補完
    if "x" not in fvcom_ds:
        fvcom_ds["x"] = ("node", np.arange(len(fvcom_ds["lon"])))
        fvcom_ds["y"] = ("node", np.arange(len(fvcom_ds["lat"])))

    fig, ax = plt.subplots(figsize=(4, 4))  # Cartesian → projection=None
    mtri.Triangulation(fvcom_ds["x"], fvcom_ds["y"], fvcom_ds["nv_zero"])
    da = fvcom_ds["temp"].isel(time=0, siglay=0)

    opts = FvcomPlotOptions(
        use_latlon=False,
        with_mesh=True,
        plot_vec2d=False,
        coastlines=False,
        obclines=False,
    )

    plotter.plot_2d(da=da, ax=ax, opts=opts)

    out = tmp_path / "scalar_cart.png"
    fig.savefig(out, dpi=100)
    _cmp_or_copy(out, "scalar_cart.png", regen_baseline)
