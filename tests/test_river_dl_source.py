# -*- coding: utf-8 -*-
"""Unit tests for RiverDLNetCDFSource and river_dl mode of RiverNetCDFGenerator."""

from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xfvcom.io.river_nc_generator import RiverNetCDFGenerator
from xfvcom.io.sources.river_dl import RiverDLNetCDFSource


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _make_river_dl_nc(
    path: Path,
    *,
    times: pd.DatetimeIndex,
    discharge: np.ndarray,
    river: str = "TestRiver",
    station: str = "TestStation",
) -> None:
    """Write a river_dl-style discharge_hourly.nc.

    Schema mirrors ``$DATA_DIR/river_dl/discharge/<R>/<S>/discharge_hourly.nc``:
    dim ``time``; vars ``discharge(time)`` [m3/s], ``time(time)`` int64.
    """
    ds = xr.Dataset(
        {
            "discharge": (
                ("time",),
                discharge.astype(np.float32),
                {"long_name": "river discharge", "units": "m3/s"},
            ),
        },
        coords={"time": times},
        attrs={"river": river, "station": station, "Conventions": "CF-1.8"},
    )
    ds.to_netcdf(path)


@pytest.fixture
def synth_nc(tmp_path: Path) -> Path:
    times = pd.date_range("2020-01-01", periods=48, freq="1h")
    q = np.arange(48.0, dtype=np.float32) + 10.0  # 10, 11, 12, ...
    p = tmp_path / "discharge_hourly.nc"
    _make_river_dl_nc(p, times=times, discharge=q)
    return p


# ------------------------------------------------------------------
# 1. Source: pass-through on matching timeline
# ------------------------------------------------------------------
def test_river_dl_source_pass_through(synth_nc: Path) -> None:
    times = pd.date_range("2020-01-01", periods=48, freq="1h")
    src = RiverDLNetCDFSource(synth_nc)

    flux = src.get_series("flux", times)
    assert flux.shape == (48,)
    np.testing.assert_allclose(flux, np.arange(48.0) + 10.0, rtol=1e-5)


def test_river_dl_source_temp_salt_constants(synth_nc: Path) -> None:
    times = pd.date_range("2020-01-01", periods=12, freq="1h")
    src = RiverDLNetCDFSource(synth_nc, temp_const=18.0, salt_const=0.5)
    np.testing.assert_array_equal(
        src.get_series("temp", times), np.full(12, 18.0, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        src.get_series("salt", times), np.full(12, 0.5, dtype=np.float32)
    )


def test_river_dl_source_scale(synth_nc: Path) -> None:
    times = pd.date_range("2020-01-01", periods=24, freq="1h")
    src = RiverDLNetCDFSource(synth_nc, scale=2.5)
    flux = src.get_series("flux", times)
    np.testing.assert_allclose(flux, (np.arange(24.0) + 10.0) * 2.5, rtol=1e-5)


def test_river_dl_source_time_interp(synth_nc: Path) -> None:
    # Request half-hourly times → linear interp between source hours
    src = RiverDLNetCDFSource(synth_nc)
    times = pd.date_range("2020-01-01 00:30:00", periods=3, freq="1h")
    flux = src.get_series("flux", times)
    # Source: t=0→10, t=1→11, t=2→12, t=3→13
    # Target: 0.5→10.5, 1.5→11.5, 2.5→12.5
    np.testing.assert_allclose(flux, [10.5, 11.5, 12.5], rtol=1e-5)


def test_river_dl_source_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RiverDLNetCDFSource(tmp_path / "does_not_exist.nc")


def test_river_dl_source_missing_variable_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.nc"
    ds = xr.Dataset({"foo": ("time", np.zeros(3))}, coords={"time": [0, 1, 2]})
    ds.to_netcdf(bad)
    with pytest.raises(KeyError, match="discharge"):
        RiverDLNetCDFSource(bad)


def test_river_dl_source_fills_internal_nan_by_default(tmp_path: Path) -> None:
    """river_dl Phase AN/AO leaves qc_flag=2 NaN holes; the source bridges
    them by default so FVCOM never sees NaN flux."""
    times = pd.date_range("2020-01-01", periods=8, freq="1h")
    q = np.array([10.0, 20.0, np.nan, np.nan, 50.0, 60.0, 70.0, 80.0], dtype=np.float32)
    p = tmp_path / "gap.nc"
    _make_river_dl_nc(p, times=times, discharge=q)
    src = RiverDLNetCDFSource(p)
    out = src.get_series("flux", times)
    # Endpoints unchanged; the two NaN samples are linearly bridged
    # between (10, 20) at t=1 and (50, 60) at t=4 → values 30, 40.
    np.testing.assert_allclose(
        out, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], rtol=1e-5
    )
    assert np.isfinite(out).all()


def test_river_dl_source_preserves_nan_when_fill_disabled(tmp_path: Path) -> None:
    times = pd.date_range("2020-01-01", periods=4, freq="1h")
    q = np.array([10.0, np.nan, np.nan, 40.0], dtype=np.float32)
    p = tmp_path / "gap_keep.nc"
    _make_river_dl_nc(p, times=times, discharge=q)
    src = RiverDLNetCDFSource(p, fill_nan=False)
    out = src.get_series("flux", times)
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_river_dl_source_unknown_variable_raises(synth_nc: Path) -> None:
    src = RiverDLNetCDFSource(synth_nc)
    times = pd.date_range("2020-01-01", periods=2, freq="1h")
    with pytest.raises(KeyError, match="Unsupported variable"):
        src.get_series("temperature", times)  # wrong name


# ------------------------------------------------------------------
# 2. Generator integration: NML + river_dl_map → forcing NC
# ------------------------------------------------------------------
NML_TXT = """\
&NML_RIVER
 RIVER_NAME = 'RiverA',
 RIVER_FILE = 'dummy.nc',
 RIVER_GRID_LOCATION = 1,
 RIVER_VERTICAL_DISTRIBUTION = 'uniform'
/
&NML_RIVER
 RIVER_NAME = 'RiverB',
 RIVER_FILE = 'dummy.nc',
 RIVER_GRID_LOCATION = 2,
 RIVER_VERTICAL_DISTRIBUTION = 'uniform'
/
"""


@pytest.fixture
def tiny_nml(tmp_path: Path) -> Path:
    p = tmp_path / "rivers.nml"
    p.write_text(NML_TXT, encoding="utf-8")
    return p


def test_river_dl_generator_smoke(tmp_path: Path, tiny_nml: Path) -> None:
    # Two synthetic discharge files
    t_full = pd.date_range("2020-01-01", periods=49, freq="1h")  # 0..48
    f_a = tmp_path / "A.nc"
    f_b = tmp_path / "B.nc"
    _make_river_dl_nc(f_a, times=t_full, discharge=np.full(49, 25.0, dtype=np.float32))
    _make_river_dl_nc(f_b, times=t_full, discharge=np.full(49, 50.0, dtype=np.float32))

    out_nc = tmp_path / "river_out.nc"
    gen = RiverNetCDFGenerator(
        tiny_nml,
        "2020-01-01T00:00:00Z",
        "2020-01-01T03:00:00Z",  # 4 timesteps at dt=3600
        3600,
        river_dl_map={
            "RiverA": {"source": f_a, "temp": 12.0, "salt": 0.0},
            "RiverB": {"source": f_b, "scale": 1.5},
        },
    )
    gen.write(out_nc)

    with nc.Dataset(out_nc, "r") as ds:
        assert ds.dimensions["rivers"].size == 2
        rnames = ds.variables["river_names"][:, :].tobytes().decode().split()
        # river_names are 80-char left-padded; reconstruct via rows
        raw_names = ds.variables["river_names"][:, :]
        cleaned = ["".join(c.decode() for c in row).strip() for row in raw_names]
        assert cleaned == ["RiverA", "RiverB"]

        flux = ds.variables["river_flux"][:, :]
        # RiverA: 25.0 unscaled; RiverB: 50.0 * 1.5 = 75.0
        np.testing.assert_allclose(flux[:, 0], np.full(4, 25.0), rtol=1e-5)
        np.testing.assert_allclose(flux[:, 1], np.full(4, 75.0), rtol=1e-5)

        temp = ds.variables["river_temp"][:, :]
        np.testing.assert_allclose(temp[:, 0], np.full(4, 12.0), rtol=1e-5)

        salt = ds.variables["river_salt"][:, :]
        np.testing.assert_array_equal(salt, np.zeros_like(salt))


def test_river_dl_generator_missing_source_raises(
    tmp_path: Path, tiny_nml: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        RiverNetCDFGenerator(
            tiny_nml,
            "2020-01-01T00:00:00Z",
            "2020-01-01T01:00:00Z",
            3600,
            river_dl_map={
                "RiverA": {"source": tmp_path / "missing.nc"},
            },
        )


def test_river_dl_generator_missing_source_key_raises(
    tmp_path: Path, tiny_nml: Path
) -> None:
    with pytest.raises(ValueError, match="source"):
        RiverNetCDFGenerator(
            tiny_nml,
            "2020-01-01T00:00:00Z",
            "2020-01-01T01:00:00Z",
            3600,
            river_dl_map={
                "RiverA": {"scale": 1.5},  # 'source' missing
            },
        )
