"""Scalar + depth-average vectors regression (lat/lon)."""

from __future__ import annotations

import shutil
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pytest
from matplotlib.testing.compare import compare_images

from xfvcom.plot.core import FvcomPlotOptions, FvcomPlotter

BASELINE = Path(__file__).parent / "baseline"
BASELINE.mkdir(exist_ok=True)


def _cmp_or_copy(src: Path, name: str, regen: bool) -> None:
    ref = BASELINE / name
    if regen or not ref.exists():
        shutil.copy(src, ref)
    else:
        diff = compare_images(str(ref), str(src), tol=1.0)  # type: ignore[call-overload]
        assert diff is None, diff


def test_scalar_vector_regression(tmp_path, fvcom_ds, plotter, regen_baseline):
    fig, ax = plt.subplots(
        figsize=(4, 4),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    # --------------------------------------------------
    # 1) Add cell-center coordinates if absent
    # --------------------------------------------------
    if "lonc" not in fvcom_ds or "latc" not in fvcom_ds:
        nv = fvcom_ds["nv_zero"].values  # (nele, 3)
        lonc = fvcom_ds["lon"].values[nv].mean(axis=1)
        latc = fvcom_ds["lat"].values[nv].mean(axis=1)
        fvcom_ds["lonc"] = ("nele", lonc)  # type: ignore[assignment]
        fvcom_ds["latc"] = ("nele", latc)  # type: ignore[assignment]

    da = fvcom_ds["temp"].isel(time=1, siglay=0)
    opts = FvcomPlotOptions(
        use_latlon=True,
        plot_vec2d=True,
        vec_siglay=0,
        vec_time=1,
        arrow_color="k",
        with_mesh=True,
        coastlines=False,
        obclines=False,
    )
    plotter.plot_2d(da=da, ax=ax, opts=opts)

    out = tmp_path / "scalar_vec.png"
    fig.savefig(out, dpi=100)
    _cmp_or_copy(out, "scalar_vec.png", regen_baseline)
