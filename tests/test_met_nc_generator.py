# -*- coding: utf-8 -*-
"""Unit-test for constant meteorology NetCDF writer (grid.dat 入口)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

from xfvcom.io.grid_reader import GridASCII
from xfvcom.io.met_nc_generator import MetNetCDFGenerator

# ------------------------------------------------------------------
# Helper: create a tiny dummy grid.dat in memory
# ------------------------------------------------------------------
GRD_TXT = """\
Node Number = 4
Cell Number = 2
1  3  1  2  3
2  3  1  3  4
1  0.0  0.0  -5
2  1.0  0.0  -5
3  1.0  1.0  -5
4  0.0  1.0  -5
"""


@pytest.fixture(scope="module")
def tiny_grid(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return path to a temporary tiny grid.dat file."""
    p = tmp_path_factory.mktemp("data") / "tiny_grid.dat"
    p.write_text(GRD_TXT, encoding="utf-8")
    return p


# ------------------------------------------------------------------
# 1. GridASCII parser sanity
# ------------------------------------------------------------------
def test_gridascii_parse(tiny_grid: Path) -> None:
    grid = GridASCII(tiny_grid)
    assert grid.nnode == 4
    assert grid.nele == 2
    assert grid.nv.shape == (2, 3)  # (nele, 3)
    ds = grid.to_xarray()
    assert {"x", "y", "nv"}.issubset(ds.variables)


# ------------------------------------------------------------------
# 2. End-to-end: grid.dat → met.nc
# ------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore:Ambiguous reference date")
def test_constant_met_nc(tmp_path: Path, tiny_grid: Path) -> None:
    out_nc = tmp_path / "met.nc"
    gen = MetNetCDFGenerator(
        grid_nc=tiny_grid,
        start="2025-01-01T00:00:00Z",
        end="2025-01-01T01:00:00Z",  # 2 timesteps
        dt_seconds=3600,
        utm_zone=54,
        northern=True,  # ← パラメータ名を現行実装に合わせる
        uwind=2.0,
        vwind=-1.0,
    )
    gen.write(out_nc)
    assert out_nc.exists()

    with nc.Dataset(out_nc) as ds:
        # dimensions
        assert ds.dimensions["time"].size == 2
        assert ds.dimensions["nele"].size == 2
        # variables present
        for v in (
            "time",
            "uwind_speed",
            "vwind_speed",
            "air_temperature",
            "relative_humidity",
            "air_pressure",
            "short_wave",
            "long_wave",
            "Precipitation",
        ):
            assert v in ds.variables
        # constant values
        assert np.all(ds.variables["uwind_speed"][:] == 2.0)
        assert np.all(ds.variables["vwind_speed"][:] == -1.0)
        assert np.all(ds.variables["air_temperature"][:] == 20.0)  # default


@pytest.mark.filterwarnings("ignore:Ambiguous reference date")
def test_timeseries_met_nc(tmp_path: Path, tiny_grid: Path) -> None:
    csv = tmp_path / "met.csv"
    csv.write_text(
        """time,uwind,air_temp
2025-01-01 09:00,1.0,14.0
2025-01-01 10:00,2.0,15.0
""",
        encoding="utf-8",
    )

    gen = MetNetCDFGenerator(
        grid_nc=tiny_grid,
        start="2025-01-01T00:00Z",
        end="2025-01-01T01:00Z",
        dt_seconds=3600,
        utm_zone=54,
        northern=True,
        ts_specs=[str(csv)],
    )
    nc_path = tmp_path / "met_ts.nc"
    gen.write(nc_path)

    with nc.Dataset(nc_path) as ds:
        assert np.all(ds["uwind_speed"][:, 0] == np.array([1.0, 2.0], dtype=float))
        assert np.all(ds["air_temperature"][:, 0] == np.array([14.0, 15.0], dtype=float))


def test_met_data_tz(tmp_path: Path, tiny_grid: Path) -> None:
    csv = tmp_path / "met_tz.csv"
    csv.write_text(
        """time,uwind
2025-01-01 00:00,1.0
2025-01-01 01:00,2.0
""",
        encoding="utf-8",
    )

    gen_bad = MetNetCDFGenerator(
        grid_nc=tiny_grid,
        start="2025-01-01T00:00Z",
        end="2025-01-01T01:00Z",
        dt_seconds=3600,
        utm_zone=54,
        northern=True,
        ts_specs=[str(csv)],
    )

    with pytest.raises(ValueError, match="outside the available data range"):
        gen_bad.render()

    gen_ok = MetNetCDFGenerator(
        grid_nc=tiny_grid,
        start="2025-01-01T00:00Z",
        end="2025-01-01T01:00Z",
        dt_seconds=3600,
        utm_zone=54,
        northern=True,
        ts_specs=[str(csv)],
        data_tz="UTC",
    )

    out_nc = tmp_path / "met_tz.nc"
    gen_ok.write(out_nc)
    with nc.Dataset(out_nc) as ds:
        assert np.all(
            ds["uwind_speed"][:, 0] == np.array([1.0, 2.0], dtype=float)
        )

