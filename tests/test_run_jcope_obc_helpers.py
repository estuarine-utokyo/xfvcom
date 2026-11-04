# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Input checks of scripts/run_jcope_obc.py (year list, OBC node file)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_jcope_obc as r  # noqa: E402  # isort: skip


@pytest.mark.parametrize("years", [[2021, 2020], [2020, 2020], [2020, 2022]])
def test_bad_year_lists_are_refused(years):
    with pytest.raises(SystemExit):
        r.check_year_list(years)


def test_year_gaps_can_be_allowed_but_order_cannot():
    r.check_year_list([2020, 2022], allow_gaps=True)
    with pytest.raises(SystemExit):
        r.check_year_list([2022, 2020], allow_gaps=True)
    r.check_year_list(r.parse_year_spec("2020-2025"))


def test_obc_file_is_parsed_strictly(tmp_path):
    ok = tmp_path / "ok.dat"
    ok.write_text("OBC Node Number = 3\n1 101 1\n2 102 1\n3 103 1\n")
    assert r.parse_obc_dat(ok).tolist() == [101, 102, 103]
    for body in (
        "OBC Node Number = 3\n1 101 1\n2 bad 1\n",
        "OBC Node Number = 2\n1 0 1\n",
        "x\n1 5 1\n2 5 1\n",
    ):
        f = tmp_path / "bad.dat"
        f.write_text(body)
        with pytest.raises(ValueError):
            r.parse_obc_dat(f)


# ---------------------------------------------------------------------------
# review round 2 (gpt-6.1-sol)
# ---------------------------------------------------------------------------
import os  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402


def _mjd(start, n):
    t = pd.date_range(start, periods=n, freq="h")
    return np.array((t - pd.Timestamp("1858-11-17")) / pd.Timedelta(days=1))


def test_round2_time_axis_of_the_sources_is_checked():
    # #2: the same file read for two years (duplicated times) is refused
    same = _mjd("2020-12-31 22:00", 2)
    with pytest.raises(SystemExit, match="outside the year 2021"):
        r.check_time_axis([2020, 2021], {2020: same, 2021: same})
    ok = {2020: _mjd("2020-12-31 22:00", 2), 2021: _mjd("2021-01-01", 3)}
    r.check_time_axis([2020, 2021], ok)
    # a gap at the year boundary needs --allow-year-gaps and non-consecutive years
    gap = {2020: _mjd("2020-12-30", 2), 2021: _mjd("2021-01-01", 3)}
    with pytest.raises(SystemExit, match="jumps"):
        r.check_time_axis([2020, 2021], gap)
    with pytest.raises(SystemExit, match="jumps"):
        r.check_time_axis([2020, 2021], gap, allow_gaps=True)
    r.check_time_axis(
        [2020, 2022], {2020: _mjd("2020-06-01", 2), 2022: _mjd("2022-01-01", 2)}, True
    )
    # within a year: a duplicate or a gap is refused
    dup = _mjd("2020-01-01", 3)
    dup[2] = dup[1]
    with pytest.raises(SystemExit, match="hourly"):
        r.check_time_axis([2020], {2020: dup})


def test_round2_region_pattern_must_name_the_year(monkeypatch, tmp_path):
    # #2
    argv = [
        "run_jcope_obc.py",
        "--years",
        "2020-2021",
        "--region-pattern",
        str(tmp_path / "region.nc"),
        "--fvcom-grid",
        "g",
        "--fvcom-dep",
        "d",
        "--fvcom-obc",
        "o",
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match=r"\{year\}"):
        r.main()


def _fp(tmp_path, **over):
    src = tmp_path / "region.nc"
    grid = tmp_path / "basic.nc"
    for p in (src, grid):
        if not p.exists():
            p.write_bytes(b"x" * 10)
    kw = dict(
        region_nc=src,
        basic_nc=grid,
        obc_ids=[1, 2],
        obc_lat=[35.0, 35.1],
        obc_lon=[139.0, 139.1],
        cell_j=[5, 6],
        cell_i=[7, 8],
        time_mjd=_mjd("2020-01-01", 3),
    )
    kw.update(over)
    return r._profile_fingerprint(**kw)


def test_round2_profile_cache_is_bound_to_its_inputs(tmp_path):
    # #4
    raw = {
        "temp": np.ones((3, 46, 2), np.float32),
        "salt": np.ones((3, 46, 2), np.float32),
    }
    cache = tmp_path / "cache"
    r._save_profile_cache(cache, 2020, _fp(tmp_path), raw)
    got = r._load_profile_cache(cache, 2020, _fp(tmp_path))
    assert got is not None and np.array_equal(got["temp"], raw["temp"])
    assert not [p for p in cache.iterdir() if p.suffix == ".tmp"]
    for over in (
        dict(obc_lat=[35.0, 35.2]),
        dict(cell_i=[7, 9]),
        dict(time_mjd=_mjd("2020-01-02", 3)),
    ):
        assert r._load_profile_cache(cache, 2020, _fp(tmp_path, **over)) is None
    # same size, other modification time: stale
    src = tmp_path / "region.nc"
    st = src.stat()
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert r._load_profile_cache(cache, 2020, _fp(tmp_path)) is None
    # another grid file
    other = tmp_path / "other_basic.nc"
    other.write_bytes(b"x" * 10)
    assert r._load_profile_cache(cache, 2020, _fp(tmp_path, basic_nc=other)) is None
    # a corrupt cache file means rebuild, not a crash
    r._save_profile_cache(cache, 2021, _fp(tmp_path), raw)
    p = r._profile_cache_path(cache, 2021)
    p.write_bytes(p.read_bytes()[:100])
    assert r._load_profile_cache(cache, 2021, _fp(tmp_path)) is None


@pytest.mark.parametrize(
    "body",
    [
        "OBC Node Number = 3\n1 101 1\n2 102 1\n",  # fewer records than declared
        "OBC Node Number = 2\n1 bad 1\n2 102 1\n",  # malformed first record
        "OBC Node Number = 2\nOBC Node Number = 2\n1 101 1\n2 102 1\n",  # two headers
        "OBC Node Number = 2\n1 101 1\n2 102\n",  # short record
        "1 101 1\n2 102 1\n",  # no header
    ],
)
def test_round2_obc_file_count_and_records(tmp_path, body):
    # #11
    f = tmp_path / "o.dat"
    f.write_text(body)
    with pytest.raises(ValueError):
        r.parse_obc_dat(f)


def test_round2_depth_file_must_match_the_mesh(tmp_path):
    # #12
    xy = np.array([[100.0, 200.0], [110.0, 210.0], [120.0, 220.0]])
    f = tmp_path / "d.dat"
    f.write_text("Node Number = 3\n100.0 200.0 5\n110.0 210.0 6\n120.0 220.0 7\n")
    assert r.parse_dep_dat(f, xy).tolist() == [5.0, 6.0, 7.0]
    for body in (
        "Node Number = 3\n110.0 210.0 6\n100.0 200.0 5\n120.0 220.0 7\n",  # reordered
        "Node Number = 3\n100.0 200.0 5\n110.0 210.0 6\n",  # short block
        "Node Number = 3\n100.0 200.0 5\n110.0 210.0 x\n120.0 220.0 7\n",  # malformed
        "Node Number = 2\n100.0 200.0 5\n110.0 210.0 6\n",  # another mesh's file
    ):
        f.write_text(body)
        with pytest.raises(ValueError):
            r.parse_dep_dat(f, xy)


def test_round2_compare_checks_alignment_before_values():
    # #19
    import compare_jcope_sources as c

    lat = 17.0 + np.arange(4) / 36.0
    lon = 117.0 + np.arange(5) / 36.0
    times = pd.date_range("2020-01-01", periods=3, freq="h")
    grid = types.SimpleNamespace(lat=lat, lon=lon)
    src = types.SimpleNamespace(lat=lat, lon=lon, time_index=times)
    # jsasaki 2026-11-05: review round 5 #11: the region schema (level/lat/lon) is now required
    reg = xr.Dataset(
        coords={"time": times, "level": np.arange(46), "lat": lat[:2], "lon": lon[:2]}
    )
    assert c.check_alignment(src, grid, reg).tolist() == [0, 1, 2]
    with pytest.raises(SystemExit, match="lat/lon"):
        c.check_alignment(
            types.SimpleNamespace(lat=lat, lon=lon + 5.0, time_index=times), grid, reg
        )
    bad_t = pd.DatetimeIndex([times[0], times[0], times[2]])
    with pytest.raises(SystemExit, match="duplicate"):
        c.check_alignment(src, grid, reg.assign_coords(time=bad_t))
    with pytest.raises(SystemExit, match="absent"):
        c.check_alignment(src, grid, reg.isel(time=slice(0, 2)))


# ---------------------------------------------------------------------------
# review round 3 (gpt-6.1-sol)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "t",
    [
        np.array([np.nan, np.nan]),
        np.array([]),
        np.zeros((2, 2)),
        np.ma.masked_array(_mjd("2020-01-01", 3), mask=[False, True, False]),
    ],
)
def test_round3_time_axis_must_be_finite_nonempty_1d(t):
    # #18: NaN compared false with every bound and passed
    with pytest.raises(SystemExit, match="non-empty, finite 1-D"):
        r.check_time_axis([2020], {2020: t})


def test_round3_compare_region_axes_have_no_relative_tolerance():
    # #16: 0.001 degree in longitude passed np.allclose's default rtol
    import compare_jcope_sources as c

    from xfvcom.io.jcope_grid import JcopeGrid

    grid = JcopeGrid.__new__(JcopeGrid)
    grid.lat = 17.0 + np.arange(10) / 36.0
    grid.lon = 117.0 + np.arange(12) / 36.0
    grid.lat_origin, grid.lon_origin, grid.dy, grid.dx = 17.0, 117.0, 1 / 36, 1 / 36
    grid.jm, grid.im = 10, 12
    reg = xr.Dataset(coords={"lat": grid.lat[2:6], "lon": grid.lon[3:9]})
    assert c.check_region_block(grid, reg) == (2, 3)
    for dlat, dlon in ((0.0, 0.001), (0.0001, 0.0)):
        shifted = xr.Dataset(
            coords={"lat": grid.lat[2:6] + dlat, "lon": grid.lon[3:9] + dlon}
        )
        with pytest.raises(SystemExit, match="not a block"):
            c.check_region_block(grid, shifted)


# ---------------------------------------------------------------------------
# review round 4 (gpt-6.1-sol)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "t",
    [
        np.array([complex(58849, np.nan), complex(58849 + 1 / 24, np.inf)]),
        np.array([58849, 58849 + 1 / 24], dtype=object),
    ],
)
def test_round4_time_axis_must_be_real(t):
    # #9: casting complex times to float64 discarded a non-finite imaginary part
    with pytest.raises(SystemExit, match="real numbers"):
        r.check_time_axis([2020], {2020: t})


# ---------------------------------------------------------------------------
# review round 5 (gpt-6.1-sol)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda ds: ds.drop_vars("level"), "coordinate level is missing"),
        (lambda ds: ds.assign_coords(level=np.arange(1, 47)), "level coordinate"),
        (
            lambda ds: ds.assign_coords(time=1577836800 + 3600 * np.arange(3)),
            "decoded datetimes",
        ),
    ],
)
def test_round5_compare_requires_the_region_schema(mutate, match):
    # #11: a region file without its level coordinate was compared all the same
    import compare_jcope_sources as c

    lat = 17.0 + np.arange(4) / 36.0
    lon = 117.0 + np.arange(5) / 36.0
    times = pd.date_range("2020-01-01", periods=3, freq="h")
    grid = types.SimpleNamespace(lat=lat, lon=lon)
    src = types.SimpleNamespace(lat=lat, lon=lon, time_index=times)
    reg = xr.Dataset(
        coords={"time": times, "level": np.arange(46), "lat": lat[:2], "lon": lon[:2]}
    )
    with pytest.raises(SystemExit, match=match):
        c.check_alignment(src, grid, mutate(reg))
