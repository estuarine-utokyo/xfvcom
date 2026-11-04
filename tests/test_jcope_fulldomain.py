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
            ds.format_version = 1
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
            ds.createVariable("level", "i4", ("level",))[:] = np.arange(
                N_JCOPE_ACTIVE_LEVELS
            )
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
        assert "missing-record policy: error" in ds.history


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


# ---------------------------------------------------------------------------
# review round 1 (gpt-6.1-sol): validation
# ---------------------------------------------------------------------------


def _edit(path, fn):
    with nc.Dataset(path, "r+") as ds:
        fn(ds)


def test_time_axis_must_be_the_exact_hours(archives):
    _basic, _region, root = archives
    p = root / "2020" / "jcopetda_20200101.nc"

    def dup(ds):
        t = ds["time"][:]
        t[5] = t[4]
        ds["time"][:] = t

    _edit(p, dup)
    with pytest.raises(ValueError, match="24 hourly UTC steps"):
        FullDomainSource(root, "2020-01-01", "2020-01-02")


def test_axes_must_match_across_days(archives):
    _basic, _region, root = archives
    _edit(
        root / "2020" / "jcopetda_20200102.nc",
        lambda ds: ds["lat"].__setitem__(slice(None), ds["lat"][:] + 1),
    )
    with pytest.raises(ValueError, match="horizontal axes differ"):
        FullDomainSource(root, "2020-01-01", "2020-01-02")


def test_unsupported_packing_is_refused(archives):
    _basic, _region, root = archives
    _edit(
        root / "2020" / "jcopetda_20200101.nc",
        lambda ds: setattr(ds["TT"], "add_offset", 5.0),
    )
    with pytest.raises(ValueError, match="add_offset"):
        FullDomainSource(root, "2020-01-01", "2020-01-02")


def test_declaration_must_name_an_hour_of_its_file(archives, tmp_path):
    _basic, region, _root = archives
    bad = tmp_path / "bad"
    _write_days(bad, region, missing="TT_2019123105")
    with pytest.raises(ValueError, match="not an hour of this file"):
        FullDomainSource(bad, "2020-01-01", "2020-01-02")


def test_timezone_aware_dates_are_utc_days(archives):
    _basic, _region, root = archives
    src = FullDomainSource(root, "2020-01-01T00:00:00Z", "2020-01-02T00:00:00+00:00")
    assert src.n_time == 48


def test_cell_indices_are_validated(archives):
    _basic, _region, root = archives
    src = FullDomainSource(root, "2020-01-01", "2020-01-02")
    with pytest.raises(ValueError, match="integers"):
        src.read_profile("TT", [0.9], [6.9])
    with pytest.raises(ValueError, match="outside"):
        src.read_profile("TT", [JM], [6])
    with pytest.raises(ValueError, match="equal length"):
        src.read_profile("TT", [1, 2], [6])


def test_land_in_the_archive_is_an_error_not_nan_forcing(archives):
    basic, _region, root = archives
    src = FullDomainSource(root, "2020-01-01", "2020-01-02")
    with pytest.raises(ValueError, match="no data at 1 requested cell"):
        src.read_profile("TT", [3], [4])  # global i=4: land (NaN) in the synthetic TT


def test_interpolation_uses_the_day_outside_the_period(archives, tmp_path):
    basic, region, _root = archives
    gap = tmp_path / "gap2"
    _write_days(gap, region, n_days=2, missing="")
    # declare and blank 2020-01-01 23Z; the period asked for is 2020-01-01 only (edge of the period)
    _edit(
        gap / "2020" / "jcopetda_20200101.nc",
        lambda ds: setattr(ds, "missing_fields", "TT_2020010123"),
    )
    with nc.Dataset(gap / "2020" / "jcopetda_20200101.nc", "r+") as ds:
        v = ds["TT"]
        v.set_auto_maskandscale(False)
        v[23] = np.full(v.shape[1:], FILL, dtype="i2")
    src = FullDomainSource(gap, "2020-01-01", "2020-01-01", missing="interpolate")
    got = src.read_profile("TT", [3], [6])
    assert np.isfinite(got).all() and src.n_time == 24
    assert any("interpolated in time" in x for x in src.provenance())


def test_writers_do_not_destroy_an_existing_file_on_failure(archives, tmp_path):
    basic, _region, root = archives
    g = _gen(basic, FullDomainSource(root, "2020-01-01", "2020-01-02"))
    out = tmp_path / "ts.nc"
    out.write_bytes(b"precious")
    with pytest.raises(Exception):
        from xfvcom.io.jcope_obc_generator import write_tsobc_nc

        write_tsobc_nc(
            out,
            time_mjd=g.mjd,
            obc_nodes=g.obc_nodes,
            obc_h=g.obc_h_fvcom,
            siglev=g.siglev,
            siglay=g.siglay,
            fields={"temp": np.zeros((3, 3))},
        )
    assert out.read_bytes() == b"precious"
    assert not list(tmp_path.glob("*.tmp"))


def test_invalid_depths_and_sigma_are_refused(archives):
    basic, _region, root = archives
    grid = JcopeGrid(basic)
    src = FullDomainSource(root, "2020-01-01", "2020-01-02")
    kw = dict(
        grid=grid,
        region_nc=src,
        obc_nodes=[1],
        obc_lat=[LAT_ORIGIN + 3 * DX],
        obc_lon=[LON_ORIGIN + 6 * DX],
        n_siglay=5,
    )
    with pytest.raises(ValueError, match="positive"):
        JcopeObcGenerator(obc_h_fvcom=[-10.0], **kw)
    with pytest.raises(ValueError, match="outside the JCOPE grid"):
        JcopeObcGenerator(
            obc_h_fvcom=[10.0], **{**kw, "obc_lat": [0.0], "obc_lon": [0.0]}
        )
    with pytest.raises(ValueError, match="finite"):
        JcopeObcGenerator(obc_h_fvcom=[10.0], **{**kw, "obc_lat": [np.nan]})
