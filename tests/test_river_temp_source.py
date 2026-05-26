# -*- coding: utf-8 -*-
"""Tests for the schema-v3 ``temp_source`` field on the river_dl adapter.

Covers:
- YAML parser validation (``_load_river_map`` / ``_validate_temp_source``).
- ``RiverNetCDFGenerator._evaluate_temp_source`` for both ``monthly_climatology``
  and ``air_regression`` kinds.
- Rejection of the legacy constant ``temp:`` field.
- End-to-end generator integration: river_temp column comes from
  temp_source, not the river_dl per-source temp_const.

Schema v3 reference:
``~/Github/TB-FVCOM/hydro/docs/directions/20260517_xfvcom_river_temp_seasonal.md``
"""

from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.typing import NDArray

from xfvcom.cli.make_river_nc_from_river_dl import (
    _load_river_map,
    _validate_temp_source,
)
from xfvcom.io.river_nc_generator import RiverNetCDFGenerator

TINY_NML = """\
&NML_RIVER
 RIVER_NAME = 'RA',
 RIVER_FILE = 'dummy.nc',
 RIVER_GRID_LOCATION = 1,
 RIVER_VERTICAL_DISTRIBUTION = 'uniform'
/
"""


@pytest.fixture
def tiny_nml(tmp_path: Path) -> Path:
    p = tmp_path / "rivers.nml"
    p.write_text(TINY_NML, encoding="utf-8")
    return p


def _make_river_dl_nc(path: Path, *, times, discharge):
    ds = xr.Dataset(
        {
            "discharge": (
                ("time",),
                discharge.astype(np.float32),
                {"long_name": "river discharge", "units": "m3/s"},
            ),
        },
        coords={"time": times},
    )
    ds.to_netcdf(path)


def _make_metforce_nc(path: Path, *, hours, lats, lons, t2_field):
    """Write a tiny metforce-style fvcom_forcing NetCDF."""
    ds = xr.Dataset(
        {
            "T2": (
                ("time", "lat", "lon"),
                t2_field.astype(np.float32),
                {"units": "degree_Celsius"},
            ),
        },
        coords={
            "time": (
                ("time",),
                hours.astype("datetime64[ns]"),
                {},
            ),
            "lat": ("lat", lats.astype(np.float32)),
            "lon": ("lon", lons.astype(np.float32)),
        },
    )
    ds.to_netcdf(path)


# ------------------------------------------------------------------
# 1. _validate_temp_source
# ------------------------------------------------------------------
def test_validate_temp_source_monthly_basic() -> None:
    spec = {
        "kind": "monthly_climatology",
        "monthly_means": [10, 11, 12, 15, 18, 22, 25, 27, 24, 19, 15, 11],
    }
    out = _validate_temp_source("R", spec)
    assert out["kind"] == "monthly_climatology"
    assert len(out["monthly_means"]) == 12
    assert out["monthly_means"][0] == 10.0


def test_validate_temp_source_monthly_wrong_length() -> None:
    spec = {"kind": "monthly_climatology", "monthly_means": [10, 11, 12]}
    with pytest.raises(ValueError, match="12-element"):
        _validate_temp_source("R", spec)


def test_validate_temp_source_air_regression_basic() -> None:
    spec = {
        "kind": "air_regression",
        "air_nc_template": "/tmp/forcing_{year}.nc",
        "air_var": "T2",
        "air_lat": 35.5,
        "air_lon": 139.8,
        "slope": 0.75,
        "intercept": 5.0,
    }
    out = _validate_temp_source("R", spec)
    assert out["kind"] == "air_regression"
    assert out["slope"] == 0.75
    assert out["air_lat"] == 35.5


def test_validate_temp_source_air_regression_missing_field() -> None:
    spec = {
        "kind": "air_regression",
        "air_nc_template": "/tmp/f.nc",
        "air_var": "T2",
        "air_lat": 35.5,
        # missing air_lon, slope, intercept
    }
    with pytest.raises(ValueError, match="missing required key"):
        _validate_temp_source("R", spec)


def test_validate_temp_source_unknown_kind() -> None:
    spec = {"kind": "linear_air_with_lag"}
    with pytest.raises(ValueError, match="not supported"):
        _validate_temp_source("R", spec)


# ------------------------------------------------------------------
# 2. _load_river_map rejects schema-v2 'temp:' field
# ------------------------------------------------------------------
def test_load_river_map_rejects_row_temp_scalar(tmp_path: Path) -> None:
    yaml_p = tmp_path / "map.yaml"
    yaml_p.write_text(
        "rivers:\n" "  - name: A\n" "    source: /tmp/a.nc\n" "    temp: 15.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema v3"):
        _load_river_map(yaml_p)


def test_load_river_map_rejects_defaults_temp_scalar(tmp_path: Path) -> None:
    yaml_p = tmp_path / "map.yaml"
    yaml_p.write_text(
        "defaults:\n"
        "  temp: 15.0\n"
        "rivers:\n"
        "  - name: A\n"
        "    source: /tmp/a.nc\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema v3"):
        _load_river_map(yaml_p)


def test_load_river_map_requires_temp_source(tmp_path: Path) -> None:
    yaml_p = tmp_path / "map.yaml"
    yaml_p.write_text(
        "rivers:\n  - name: A\n    source: /tmp/a.nc\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="required"):
        _load_river_map(yaml_p)


def test_load_river_map_defaults_temp_source_inherited(tmp_path: Path) -> None:
    yaml_p = tmp_path / "map.yaml"
    yaml_p.write_text(
        "defaults:\n"
        "  temp_source:\n"
        "    kind: monthly_climatology\n"
        "    monthly_means: [10, 11, 12, 15, 18, 22, 25, 27, 24, 19, 15, 11]\n"
        "rivers:\n"
        "  - name: A\n"
        "    source: /tmp/a.nc\n"
        "  - name: B\n"
        "    source: /tmp/b.nc\n"
        "    temp_source:\n"
        "      kind: monthly_climatology\n"
        "      monthly_means: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]\n",
        encoding="utf-8",
    )
    m, _ = _load_river_map(yaml_p)
    # A inherits default
    assert m["A"]["temp_source"]["monthly_means"][0] == 10.0
    # B overrides
    assert m["B"]["temp_source"]["monthly_means"][0] == 5.0


# ------------------------------------------------------------------
# 3. Generator _evaluate_temp_source(monthly_climatology)
# ------------------------------------------------------------------
def test_generator_monthly_climatology(tmp_path: Path, tiny_nml: Path) -> None:
    # Source NC with constant discharge.
    times = pd.date_range("2020-01-01", periods=24, freq="1h")
    f_src = tmp_path / "src.nc"
    _make_river_dl_nc(f_src, times=times, discharge=np.full(24, 10.0, dtype=np.float32))

    gen = RiverNetCDFGenerator(
        tiny_nml,
        "2020-01-01T00:00:00Z",
        "2020-01-01T23:00:00Z",
        3600,
        river_dl_map={
            "RA": {
                "kind": "river_dl",
                "source": f_src,
                "temp_source": {
                    "kind": "monthly_climatology",
                    "monthly_means": [10, 11, 12, 15, 18, 22, 25, 27, 24, 19, 15, 11],
                },
            }
        },
    )
    # All January, expect 10 °C throughout.
    arr = gen._evaluate_temp_source("RA", gen._temp_sources["RA"])
    assert arr.shape == (24,)
    assert np.allclose(arr, 10.0)


def test_generator_monthly_climatology_across_months(
    tmp_path: Path, tiny_nml: Path
) -> None:
    times = pd.date_range("2020-01-01", periods=24 * 35, freq="1h")
    f_src = tmp_path / "src.nc"
    _make_river_dl_nc(
        f_src,
        times=times,
        discharge=np.full(times.size, 10.0, dtype=np.float32),
    )
    gen = RiverNetCDFGenerator(
        tiny_nml,
        "2020-01-15T00:00:00Z",
        "2020-02-15T00:00:00Z",
        3600,
        river_dl_map={
            "RA": {
                "kind": "river_dl",
                "source": f_src,
                "temp_source": {
                    "kind": "monthly_climatology",
                    "monthly_means": [
                        10,
                        20,
                        30,
                        40,
                        50,
                        60,
                        70,
                        80,
                        90,
                        100,
                        110,
                        120,
                    ],
                },
            }
        },
    )
    arr = gen._evaluate_temp_source("RA", gen._temp_sources["RA"])
    # First half is January = 10; later half is February = 20.
    months = np.asarray(gen.timeline.month)
    assert np.allclose(arr[months == 1], 10.0)
    assert np.allclose(arr[months == 2], 20.0)


# ------------------------------------------------------------------
# 4. Generator _evaluate_temp_source(air_regression)
# ------------------------------------------------------------------
def test_generator_air_regression_basic(tmp_path: Path, tiny_nml: Path) -> None:
    # Build a synthetic metforce NC for 2020 with T2(t, lat, lon).
    # 24 hours, 2x2 grid; T2 chosen so the nearest cell to (35.0, 139.0)
    # is index (0, 0) with a known time series.
    hours = pd.date_range("2020-01-01", periods=24, freq="1h").to_numpy()
    lats = np.array([35.0, 36.0])
    lons = np.array([139.0, 140.0])
    t2: NDArray[np.float32] = np.zeros((24, 2, 2), dtype=np.float32)
    # Make (ilat=0, ilon=0) have a ramp 0..23 °C
    t2[:, 0, 0] = np.arange(24, dtype=np.float32)
    # Other cells stay at 100 (so we can detect if the wrong cell is read).
    t2[:, 0, 1] = 100.0
    t2[:, 1, 0] = 100.0
    t2[:, 1, 1] = 100.0
    metforce_nc = tmp_path / "fvcom_forcing_2020.nc"
    _make_metforce_nc(metforce_nc, hours=hours, lats=lats, lons=lons, t2_field=t2)

    # Source NC just so RiverDLNetCDFSource can be constructed.
    f_src = tmp_path / "src.nc"
    _make_river_dl_nc(
        f_src,
        times=pd.date_range("2020-01-01", periods=24, freq="1h"),
        discharge=np.full(24, 10.0, dtype=np.float32),
    )

    spec = {
        "kind": "air_regression",
        "air_nc_template": str(tmp_path / "fvcom_forcing_{year}.nc"),
        "air_var": "T2",
        "air_lat": 35.0,
        "air_lon": 139.0,
        "slope": 0.5,
        "intercept": 2.0,
    }
    gen = RiverNetCDFGenerator(
        tiny_nml,
        "2020-01-01T00:00:00Z",
        "2020-01-01T23:00:00Z",
        3600,
        river_dl_map={
            "RA": {
                "kind": "river_dl",
                "source": f_src,
                "temp_source": spec,
            }
        },
    )
    arr = gen._evaluate_temp_source("RA", gen._temp_sources["RA"])
    # Expect: 0.5 * (0..23) + 2 = 2..13.5
    expected = 0.5 * np.arange(24, dtype=np.float64) + 2.0
    np.testing.assert_allclose(arr, expected, rtol=1e-4)


def test_generator_air_regression_clip(tmp_path: Path, tiny_nml: Path) -> None:
    hours = pd.date_range("2020-01-01", periods=24, freq="1h").to_numpy()
    lats = np.array([35.0])
    lons = np.array([139.0])
    t2: NDArray[np.float32] = np.zeros((24, 1, 1), dtype=np.float32)
    t2[:, 0, 0] = np.arange(24, dtype=np.float32) - 10  # -10..13
    metforce_nc = tmp_path / "fvcom_forcing_2020.nc"
    _make_metforce_nc(metforce_nc, hours=hours, lats=lats, lons=lons, t2_field=t2)

    f_src = tmp_path / "src.nc"
    _make_river_dl_nc(
        f_src,
        times=pd.date_range("2020-01-01", periods=24, freq="1h"),
        discharge=np.full(24, 10.0, dtype=np.float32),
    )
    spec = {
        "kind": "air_regression",
        "air_nc_template": str(tmp_path / "fvcom_forcing_{year}.nc"),
        "air_var": "T2",
        "air_lat": 35.0,
        "air_lon": 139.0,
        "slope": 1.0,
        "intercept": 0.0,
        "min_temp": 0.0,
        "max_temp": 10.0,
    }
    gen = RiverNetCDFGenerator(
        tiny_nml,
        "2020-01-01T00:00:00Z",
        "2020-01-01T23:00:00Z",
        3600,
        river_dl_map={
            "RA": {
                "kind": "river_dl",
                "source": f_src,
                "temp_source": spec,
            }
        },
    )
    arr = gen._evaluate_temp_source("RA", gen._temp_sources["RA"])
    expected = np.clip(np.arange(24, dtype=np.float64) - 10, 0.0, 10.0)
    np.testing.assert_allclose(arr, expected, rtol=1e-4)


# ------------------------------------------------------------------
# 5. End-to-end: river_temp column in the rendered NC uses temp_source
# ------------------------------------------------------------------
def test_render_river_temp_from_temp_source(tmp_path: Path, tiny_nml: Path) -> None:
    """The river_temp column must come from temp_source, NOT from the
    river_dl source's temp_const constant."""
    f_src = tmp_path / "src.nc"
    _make_river_dl_nc(
        f_src,
        times=pd.date_range("2020-01-01", periods=24, freq="1h"),
        discharge=np.full(24, 10.0, dtype=np.float32),
    )
    out_nc = tmp_path / "out.nc"
    gen = RiverNetCDFGenerator(
        tiny_nml,
        "2020-01-01T00:00:00Z",
        "2020-01-01T03:00:00Z",
        3600,
        river_dl_map={
            "RA": {
                "kind": "river_dl",
                "source": f_src,
                "temp_source": {
                    "kind": "monthly_climatology",
                    "monthly_means": [42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
            }
        },
    )
    gen.write(out_nc)
    with nc.Dataset(out_nc) as ds:
        temp = ds.variables["river_temp"][:, 0]
        # All January at 42 °C (overrides the river_dl temp_const default).
        np.testing.assert_allclose(temp, 42.0, rtol=1e-4)


# ------------------------------------------------------------------
# 6. air_regression antecedent-air-temperature smoothing
#    (docs/directions/20260526_river_temp_antecedent_air_smoothing.md)
# ------------------------------------------------------------------
def _make_metforce_single_cell(path: Path, *, t2_1d, start: str = "2020-01-01"):
    """Write a 1-cell metforce NC whose T2 series is ``t2_1d`` (hourly)."""
    nh = len(t2_1d)
    hours = pd.date_range(start, periods=nh, freq="1h")
    lats = np.array([35.0])
    lons = np.array([139.0])
    t2: NDArray[np.float32] = np.zeros((nh, 1, 1), dtype=np.float32)
    t2[:, 0, 0] = np.asarray(t2_1d, dtype=np.float32)
    _make_metforce_nc(path, hours=hours.to_numpy(), lats=lats, lons=lons, t2_field=t2)
    return hours


def _air_spec(tmp_path: Path, *, slope, intercept, smoothing_days=None, method=None):
    spec: dict = {
        "kind": "air_regression",
        "air_nc_template": str(tmp_path / "fvcom_forcing_{year}.nc"),
        "air_var": "T2",
        "air_lat": 35.0,
        "air_lon": 139.0,
        "slope": slope,
        "intercept": intercept,
    }
    if smoothing_days is not None:
        spec["smoothing_days"] = smoothing_days
    if method is not None:
        spec["smoothing_method"] = method
    return spec


def _build_gen(tiny_nml, tmp_path, start, end, spec, src_hours):
    # Unique src filename per spec object: tests that build two generators
    # must not overwrite a discharge NC still held open by the first one.
    f_src = tmp_path / f"src_{id(spec):x}.nc"
    _make_river_dl_nc(
        f_src,
        times=src_hours,
        discharge=np.full(len(src_hours), 10.0, dtype=np.float32),
    )
    return RiverNetCDFGenerator(
        tiny_nml,
        start,
        end,
        3600,
        river_dl_map={"RA": {"kind": "river_dl", "source": f_src, "temp_source": spec}},
    )


# --- 6a. schema validation of the new optional keys -----------------
def test_validate_smoothing_days_basic() -> None:
    spec = {
        "kind": "air_regression",
        "air_nc_template": "/tmp/f_{year}.nc",
        "air_var": "T2",
        "air_lat": 35.0,
        "air_lon": 139.0,
        "slope": 0.8,
        "intercept": 3.0,
        "smoothing_days": 7,
    }
    out = _validate_temp_source("R", spec)
    assert out["smoothing_days"] == 7.0
    assert "smoothing_method" not in out  # method defaulted downstream, not here


def test_validate_smoothing_method_exponential() -> None:
    spec = {
        "kind": "air_regression",
        "air_nc_template": "/tmp/f_{year}.nc",
        "air_var": "T2",
        "air_lat": 35.0,
        "air_lon": 139.0,
        "slope": 0.8,
        "intercept": 3.0,
        "smoothing_days": 5.0,
        "smoothing_method": "exponential",
    }
    out = _validate_temp_source("R", spec)
    assert out["smoothing_method"] == "exponential"


@pytest.mark.parametrize("bad", [0, -1, "abc"])
def test_validate_smoothing_days_invalid(bad) -> None:
    spec = {
        "kind": "air_regression",
        "air_nc_template": "/tmp/f_{year}.nc",
        "air_var": "T2",
        "air_lat": 35.0,
        "air_lon": 139.0,
        "slope": 0.8,
        "intercept": 3.0,
        "smoothing_days": bad,
    }
    with pytest.raises(ValueError, match="smoothing_days"):
        _validate_temp_source("R", spec)


def test_validate_smoothing_method_unknown() -> None:
    spec = {
        "kind": "air_regression",
        "air_nc_template": "/tmp/f_{year}.nc",
        "air_var": "T2",
        "air_lat": 35.0,
        "air_lon": 139.0,
        "slope": 0.8,
        "intercept": 3.0,
        "smoothing_days": 7.0,
        "smoothing_method": "bogus",
    }
    with pytest.raises(ValueError, match="smoothing_method"):
        _validate_temp_source("R", spec)


def test_validate_smoothing_method_without_days() -> None:
    spec = {
        "kind": "air_regression",
        "air_nc_template": "/tmp/f_{year}.nc",
        "air_var": "T2",
        "air_lat": 35.0,
        "air_lon": 139.0,
        "slope": 0.8,
        "intercept": 3.0,
        "smoothing_method": "simple",
    }
    with pytest.raises(ValueError, match="without smoothing_days"):
        _validate_temp_source("R", spec)


# --- 6b. generator behaviour ----------------------------------------
def test_smoothing_absent_is_unchanged(tmp_path: Path, tiny_nml: Path) -> None:
    """No smoothing_days -> identical to the existing same-timestep path."""
    hours = _make_metforce_single_cell(
        tmp_path / "fvcom_forcing_2020.nc",
        t2_1d=np.arange(24, dtype=np.float64),
    )
    gen = _build_gen(
        tiny_nml,
        tmp_path,
        "2020-01-01T00:00:00Z",
        "2020-01-01T23:00:00Z",
        _air_spec(tmp_path, slope=0.5, intercept=2.0),
        hours,
    )
    arr = gen._evaluate_temp_source("RA", gen._temp_sources["RA"])
    np.testing.assert_allclose(arr, 0.5 * np.arange(24) + 2.0, rtol=1e-4)


def test_smoothing_reduces_variance_preserves_mean(
    tmp_path: Path, tiny_nml: Path
) -> None:
    """A 7-day trailing mean of a pure diurnal sinusoid flattens to the
    daily mean: variance drops, time-mean is preserved."""
    nh = 24 * 40
    hour_of_day: NDArray[np.float64] = np.arange(nh, dtype=np.float64) % 24
    t2 = 10.0 + 5.0 * np.sin(2 * np.pi * hour_of_day / 24.0)
    hours = _make_metforce_single_cell(tmp_path / "fvcom_forcing_2020.nc", t2_1d=t2)

    start, end = "2020-01-08T00:00:00Z", "2020-02-07T00:00:00Z"
    gen_raw = _build_gen(
        tiny_nml,
        tmp_path,
        start,
        end,
        _air_spec(tmp_path, slope=1.0, intercept=0.0),
        hours,
    )
    raw = gen_raw._evaluate_temp_source("RA", gen_raw._temp_sources["RA"])
    gen_s = _build_gen(
        tiny_nml,
        tmp_path,
        start,
        end,
        _air_spec(tmp_path, slope=1.0, intercept=0.0, smoothing_days=7.0),
        hours,
    )
    sm = gen_s._evaluate_temp_source("RA", gen_s._temp_sources["RA"])

    assert not np.isnan(sm).any()
    assert sm.var() < 0.01 * raw.var()  # diurnal swing collapses
    assert abs(float(sm.mean()) - float(raw.mean())) < 0.02  # mean preserved
    np.testing.assert_allclose(sm, 10.0, atol=0.05)  # flat at the daily mean


def test_smoothing_commutes_with_regression(tmp_path: Path, tiny_nml: Path) -> None:
    """Smoothing the air input then regressing == regressing then smoothing
    the output (the model is linear), in the interior where the window is
    fully populated."""
    nh = 24 * 40
    idx: NDArray[np.float64] = np.arange(nh, dtype=np.float64)
    t2 = 5.0 + 0.05 * idx + 3.0 * np.sin(2 * np.pi * (idx % 24) / 24.0)
    hours = _make_metforce_single_cell(tmp_path / "fvcom_forcing_2020.nc", t2_1d=t2)

    start, end = "2020-01-08T00:00:00Z", "2020-02-07T00:00:00Z"
    slope, intercept = 0.8, 3.0
    raw_spec = _air_spec(tmp_path, slope=slope, intercept=intercept)
    raw = _build_gen(
        tiny_nml, tmp_path, start, end, raw_spec, hours
    )._evaluate_temp_source("RA", raw_spec)
    sm_spec = _air_spec(tmp_path, slope=slope, intercept=intercept, smoothing_days=5.0)
    sm = _build_gen(
        tiny_nml, tmp_path, start, end, sm_spec, hours
    )._evaluate_temp_source("RA", sm_spec)

    n = 5 * 24
    rolled = pd.Series(raw).rolling(n, min_periods=n).mean().to_numpy()
    # Compare only where raw's own trailing window is full.
    np.testing.assert_allclose(sm[n - 1 :], rolled[n - 1 :], rtol=1e-4, atol=1e-3)


def test_smoothing_leadin_full_window(tmp_path: Path, tiny_nml: Path) -> None:
    """With a lead-in available, the first output point uses a *full*
    trailing window (not a ramp-up)."""
    nh = 24 * 40
    idx: NDArray[np.float64] = np.arange(nh, dtype=np.float64)
    t2 = 5.0 + 0.1 * idx  # monotone trend so window-mean != instantaneous
    hours = _make_metforce_single_cell(tmp_path / "fvcom_forcing_2020.nc", t2_1d=t2)

    start, end = "2020-01-08T00:00:00Z", "2020-02-07T00:00:00Z"
    spec = _air_spec(tmp_path, slope=1.0, intercept=0.0, smoothing_days=7.0)
    gen = _build_gen(tiny_nml, tmp_path, start, end, spec, hours)
    sm = gen._evaluate_temp_source("RA", spec)

    n = 7 * 24
    idx_t0 = 7 * 24  # 2020-01-08 is 7 days (168 h) after the 2020-01-01 start
    expected0 = float(np.mean(t2[idx_t0 - (n - 1) : idx_t0 + 1]))
    assert not np.isnan(sm).any()
    np.testing.assert_allclose(sm[0], expected0, rtol=1e-4)
    # A ramp-up would have given the instantaneous value t2[idx_t0]; ensure
    # the full-window mean is clearly different from it.
    assert abs(expected0 - t2[idx_t0]) > 5.0


def test_smoothing_no_leadin_warns_and_ramps(tmp_path: Path, tiny_nml: Path) -> None:
    """Starting at the archive's first year (no prior NC) warns and ramps
    up over the first window without producing NaN."""
    nh = 24 * 40
    idx: NDArray[np.float64] = np.arange(nh, dtype=np.float64)
    t2 = 5.0 + 0.1 * idx
    hours = _make_metforce_single_cell(tmp_path / "fvcom_forcing_2020.nc", t2_1d=t2)

    start, end = "2020-01-01T00:00:00Z", "2020-01-31T00:00:00Z"
    spec = _air_spec(tmp_path, slope=1.0, intercept=0.0, smoothing_days=7.0)
    gen = _build_gen(tiny_nml, tmp_path, start, end, spec, hours)
    with pytest.warns(UserWarning, match="lead-in"):
        sm = gen._evaluate_temp_source("RA", spec)
    assert not np.isnan(sm).any()
    # min_periods=1 -> first point is just the instantaneous value.
    np.testing.assert_allclose(sm[0], t2[0], rtol=1e-5)


def test_smoothing_exponential_runs(tmp_path: Path, tiny_nml: Path) -> None:
    """The exponential method produces a finite, variance-reduced series."""
    nh = 24 * 40
    hour_of_day: NDArray[np.float64] = np.arange(nh, dtype=np.float64) % 24
    t2 = 10.0 + 5.0 * np.sin(2 * np.pi * hour_of_day / 24.0)
    hours = _make_metforce_single_cell(tmp_path / "fvcom_forcing_2020.nc", t2_1d=t2)

    start, end = "2020-01-08T00:00:00Z", "2020-02-07T00:00:00Z"
    raw = _build_gen(
        tiny_nml,
        tmp_path,
        start,
        end,
        _air_spec(tmp_path, slope=1.0, intercept=0.0),
        hours,
    )._evaluate_temp_source("RA", _air_spec(tmp_path, slope=1.0, intercept=0.0))
    spec = _air_spec(
        tmp_path, slope=1.0, intercept=0.0, smoothing_days=5.0, method="exponential"
    )
    sm = _build_gen(tiny_nml, tmp_path, start, end, spec, hours)._evaluate_temp_source(
        "RA", spec
    )
    assert not np.isnan(sm).any()
    assert sm.var() < raw.var()
