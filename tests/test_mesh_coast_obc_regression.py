"""Mesh + coastline + open-boundary line regression."""

from __future__ import annotations

import shutil
from pathlib import Path

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
        diff = compare_images(ref, src, tol=1.0)
        assert diff is None, diff


def test_mesh_coast_obc_regression(tmp_path, fvcom_ds, plotter, regen_baseline):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection="platecarree")

    opts = FvcomPlotOptions(
        use_latlon=True,
        with_mesh=True,
        coastlines=True,
        obclines=True,
        plot_vec2d=False,
    )
    # da=None → mesh-only
    plotter.plot_2d(da=None, ax=ax, opts=opts)

    out = tmp_path / "mesh_coast.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    _cmp_or_copy(out, "mesh_coast.png", regen_baseline)
