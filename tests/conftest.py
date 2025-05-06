"""
Global pytest fixtures for xfvcom unit tests.
Provide a minimal FVCOM-like Dataset (fvcom_ds) and a ready Plotter (plotter).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xfvcom.plot.core import FvcomPlotConfig, FvcomPlotter


@pytest.fixture
def fvcom_ds() -> xr.Dataset:
    """Return a minimal 3-D FVCOM-like dataset suitable for plotting tests."""
    time = pd.date_range("2020-01-01", periods=2, freq="h")
    siglay = np.arange(3)
    node = np.arange(5)
    nele = len(node) - 2

    # coordinates
    lon = np.linspace(140.0, 140.1, len(node))
    lat = np.linspace(35.0, 35.1, len(node))

    ds = xr.Dataset()
    ds["lon"] = ("node", lon)
    ds["lat"] = ("node", lat)

    # connectivity: simple fan
    nv_zero = np.vstack([node[:-2], node[1:-1], node[2:]]).T
    ds["nv_zero"] = (("nele", "three"), nv_zero)

    # depth & zeta for layer thickness
    ds["h"] = ("node", np.full(len(node), 10.0))
    ds["zeta"] = (("time", "node"), np.zeros((len(time), len(node))))

    # sigma coordinates
    siglev = np.linspace(-1.0, 0.0, len(siglay) + 1)
    ds["siglev"] = (("siglev", "node"), np.tile(siglev[:, None], (1, len(node))))

    # 3-D velocity (cell‐centred) and scalar (node)
    rng = np.random.default_rng(seed=0)
    ds["u"] = (("time", "siglay", "nele"), rng.random((len(time), len(siglay), nele)))
    ds["v"] = (("time", "siglay", "nele"), rng.random((len(time), len(siglay), nele)))
    ds["temp"] = (
        ("time", "siglay", "node"),
        rng.random((len(time), len(siglay), len(node))),
        {"long_name": "Temperature", "units": "°C"},
    )

    return ds


def pytest_addoption(parser):
    parser.addoption(
        "--regenerate-baseline",
        action="store_true",
        help="Overwrite baseline PNGs with current output",
    )


@pytest.fixture
def plotter(fvcom_ds: xr.Dataset) -> FvcomPlotter:
    """Return a FvcomPlotter instance bound to *fvcom_ds*."""
    return FvcomPlotter(fvcom_ds, FvcomPlotConfig())


@pytest.fixture
def regen_baseline(request) -> bool:  # 型ヒントは任意
    """True if pytest was run with --regenerate-baseline."""
    return bool(request.config.getoption("--regenerate-baseline"))
