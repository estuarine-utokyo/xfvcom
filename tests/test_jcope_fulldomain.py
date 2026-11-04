# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Tests for :class:`xfvcom.io.FullDomainSource` and the generator's full-domain path.

Synthetic day files (int16 packed, 24 hourly steps, 46 levels) are written from the same fields as
the synthetic ``region.nc`` of test_jcope_obc_generator, so the two source kinds must give the same
OBC arrays up to the packing step.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_jcope_obc_generator import (  # isort: skip  # noqa: E402
    DX,
    IM,
    JM,
    LAT_ORIGIN,
    LON_ORIGIN,
    _build_basic_nc,
    _build_region_nc,
)

from xfvcom.io import FullDomainSource, JcopeGrid, JcopeObcGenerator
from xfvcom.io.jcope_obc_generator import N_JCOPE_ACTIVE_LEVELS

FILL = -32768
SCALE = {"TT": 0.01, "ST": 0.01, "UT": 0.001, "VT": 0.001, "EGT": 0.001}


def _write_days(
    root: Path,
    region_path: Path,
    n_days: int = 2,
    missing: str = "",
    chunks=(1, N_JCOPE_ACTIVE_LEVELS, 3, 4),
) -> None:
    """Day files 2020-01-01.. holding the region file's fields on the full (JM x IM) grid."""
    with nc.Dataset(region_path) as r:
        r_lon = np.asarray(r["lon"][:])
        i0 = int(round((r_lon[0] - LON_ORIGIN) / DX))
        data = {
            v: np.ma.filled(r[v][:], np.nan) for v in ("TT", "ST", "UT", "VT", "EGT")
        }
    lat = LAT_ORIGIN + np.arange(JM) * DX
    lon = LON_ORIGIN + np.arange(IM) * DX
    for d in range(n_days):
        path = root / "2020" / f"jcopetda_202001{d + 1:02d}.nc"
        path.parent.mkdir(parents=True, exist_ok=True)
        with nc.Dataset(path, "w") as ds:
            ds.product = "full_domain"
            ds.missing_fields = missing if d == 0 else ""
            ds.createDimension("time", 24)
            ds.createDimension("level", N_JCOPE_ACTIVE_LEVELS)
            ds.createDimension("lat", JM)
            ds.createDimension("lon", IM)
            tv = ds.createVariable("time", "f8", ("time",))
            tv.units, tv.calendar = "hours since 2000-01-01 00:00:00", "standard"
            t0 = (
                datetime(2020, 1, 1 + d) - datetime(2000, 1, 1)
            ).total_seconds() / 3600
            tv[:] = t0 + np.arange(24)
            ds.createVariable("lat", "f8", ("lat",))[:] = lat
            ds.createVariable("lon", "f8", ("lon",))[:] = lon
            for name, arr in data.items():
                dims = (
                    ("time", "level", "lat", "lon")
                    if arr.ndim == 4
                    else ("time", "lat", "lon")
                )
                full = np.full(arr.shape[:-1] + (IM,), np.nan, dtype=np.float64)
                full[..., i0:] = arr
                q = np.where(
                    np.isfinite(full), np.rint(full / SCALE[name]), FILL
                ).astype("i2")
                ch = chunks if arr.ndim == 4 else (chunks[0],) + chunks[2:]
                v = ds.createVariable(
                    name,
                    "i2",
                    dims,
                    fill_value=np.int16(FILL),
                    zlib=True,
                    chunksizes=ch,
                )
                v.scale_factor, v.add_offset = SCALE[name], 0.0
                v.set_auto_maskandscale(False)
                v[:] = q[24 * d : 24 * d + 24]


@pytest.fixture
def archives(tmp_path):
    basic = tmp_path / "basic.nc"
    region = tmp_path / "region.nc"
    _build_basic_nc(basic)
    _build_region_nc(region, n_time=48)
    root = tmp_path / "archive"
    _write_days(root, region)
    return basic, region, root


def _gen(basic, source):
    grid = JcopeGrid(basic)
    lats = [LAT_ORIGIN + 3 * DX] * 3
    lons = [
        LON_ORIGIN + 6 * DX,
        LON_ORIGIN + 8 * DX,
        LON_ORIGIN + 4 * DX,
    ]  # last one snaps from land
    return JcopeObcGenerator(
        grid=grid,
        region_nc=source,
        obc_nodes=[101, 102, 103],
        obc_lat=lats,
        obc_lon=lons,
        obc_h_fvcom=np.array([100.0, 200.0, 30.0], dtype=np.float32),
        n_siglay=10,
    )


def test_archive_equals_region(archives):
    basic, region, root = archives
    g_reg = _gen(basic, region)
    g_arc = _gen(basic, FullDomainSource(root, "2020-01-01", "2020-01-02"))
    assert np.array_equal(g_reg.mjd, g_arc.mjd)
    a, b = g_reg.build_tsobc_arrays(), g_arc.build_tsobc_arrays()
    for v in a:
        assert b[v].shape == a[v].shape
        np.testing.assert_allclose(b[v], a[v], atol=0.011, equal_nan=True)
    np.testing.assert_allclose(
        g_arc.build_elevation_array(), g_reg.build_elevation_array(), atol=1.1e-3
    )


def test_writes_with_source_label(archives, tmp_path):
    basic, _region, root = archives
    g = _gen(basic, FullDomainSource(root, "2020-01-01", "2020-01-02"))
    out = g.write_tsobc(tmp_path / "ts.nc")
    with nc.Dataset(out) as ds:
        assert "archive[20200101-20200102]" in ds.history


def test_missing_day_file_is_an_error(archives):
    _basic, _region, root = archives
    with pytest.raises(FileNotFoundError):
        FullDomainSource(root, "2020-01-01", "2020-01-03")


def test_declared_missing_hour(archives, tmp_path):
    basic, region, root = archives
    miss_root = tmp_path / "gap"
    _write_days(miss_root, region, missing="TT_2020010105")
    # blank that hour of TT in the first day file, as the archive does
    p = miss_root / "2020" / "jcopetda_20200101.nc"
    with nc.Dataset(p, "r+") as ds:
        v = ds["TT"]
        v.set_auto_maskandscale(False)
        v[5] = np.full(v.shape[1:], FILL, dtype="i2")
    with pytest.raises(ValueError, match="declared missing"):
        _gen(
            basic, FullDomainSource(miss_root, "2020-01-01", "2020-01-02")
        ).build_tsobc_arrays()
    g_ok = _gen(basic, FullDomainSource(root, "2020-01-01", "2020-01-02"))
    g_fix = _gen(
        basic,
        FullDomainSource(miss_root, "2020-01-01", "2020-01-02", missing="interpolate"),
    )
    # the synthetic TT is linear in time, so interpolation restores the hour (up to packing)
    np.testing.assert_allclose(
        g_fix.build_tsobc_arrays(["temp"])["temp"],
        g_ok.build_tsobc_arrays(["temp"])["temp"],
        atol=0.011,
    )


def test_picklable(archives):
    import pickle

    _basic, _region, root = archives
    src = FullDomainSource(root, "2020-01-01", "2020-01-02")
    again = pickle.loads(pickle.dumps(src))
    assert again.n_time == 48 and again.name == src.name
