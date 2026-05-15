# -*- coding: utf-8 -*-
"""Unit tests for MetforceGriddedSource and metforce-mode MetNetCDFGenerator."""

from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xfvcom.io.met_nc_generator import MetNetCDFGenerator
from xfvcom.io.sources.metforce import MetforceGriddedSource


# ------------------------------------------------------------------
# Fixtures
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
    """Tiny 4-node, 2-element grid.dat (unit square, two triangles)."""
    p = tmp_path_factory.mktemp("data") / "tiny_grid.dat"
    p.write_text(GRD_TXT, encoding="utf-8")
    return p


def _make_synthetic_metforce(
    path: Path,
    *,
    times: pd.DatetimeIndex,
    lats: np.ndarray,
    lons: np.ndarray,
    constants: dict[str, float] | None = None,
) -> None:
    """Write a synthetic metforce-style file: ``(time, lat, lon)`` per var.

    By default every variable is filled with ``lat + 10*lon`` so bilinear
    interpolation between grid corners yields predictable closed-form values.
    Pass *constants* to override per-variable fill (useful for testing unit
    conversions in isolation).
    """
    constants = constants or {}
    nt, nlat, nlon = times.size, lats.size, lons.size

    base = np.empty((nt, nlat, nlon), dtype=np.float32)
    for j, la in enumerate(lats):
        for i, lo in enumerate(lons):
            base[:, j, i] = la + 10.0 * lo

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for var in (
        "U10",
        "V10",
        "T2",
        "SH",
        "SLP",
        "DSWRF",
        "DLWRF",
        "PRECIP",
    ):
        if var in constants:
            arr = np.full_like(base, fill_value=constants[var])
        else:
            arr = base.copy()
        data_vars[var] = (("time", "lat", "lon"), arr)

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": times,
            "lat": ("lat", lats.astype(np.float32)),
            "lon": ("lon", lons.astype(np.float32)),
        },
    )
    ds.to_netcdf(path)


# ------------------------------------------------------------------
# 1. Source: bilinear at known points
# ------------------------------------------------------------------
def test_metforce_source_bilinear_at_grid_points(tmp_path: Path) -> None:
    times = pd.date_range("2020-01-01", periods=3, freq="1h")
    lats = np.array([0.0, 0.5, 1.0])
    lons = np.array([0.0, 0.5, 1.0])
    fpath = tmp_path / "mf.nc"
    _make_synthetic_metforce(fpath, times=times, lats=lats, lons=lons)

    # Target points = the source grid corners; result must equal the
    # analytic fill (la + 10*lo) exactly.
    tgt_lon = np.array([0.0, 1.0, 0.0, 1.0])
    tgt_lat = np.array([0.0, 0.0, 1.0, 1.0])
    src = MetforceGriddedSource(
        fpath,
        target_node_lon=tgt_lon,
        target_node_lat=tgt_lat,
        target_elem_lon=tgt_lon,
        target_elem_lat=tgt_lat,
    )

    out = src.get_spatial_series("uwind", times, on="node")
    assert out.shape == (3, 4)
    expected = np.array([0.0, 10.0, 1.0, 11.0])
    np.testing.assert_allclose(out[0], expected, rtol=1e-5)
    np.testing.assert_allclose(out[1], expected, rtol=1e-5)


def test_metforce_source_bilinear_midpoint(tmp_path: Path) -> None:
    times = pd.date_range("2020-01-01", periods=1, freq="1h")
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0])
    fpath = tmp_path / "mf.nc"
    _make_synthetic_metforce(fpath, times=times, lats=lats, lons=lons)

    # Centre of the unit square: bilinear of corners {0, 10, 1, 11} = 5.5
    tgt = np.array([0.5])
    src = MetforceGriddedSource(
        fpath,
        target_node_lon=tgt,
        target_node_lat=tgt,
        target_elem_lon=tgt,
        target_elem_lat=tgt,
    )
    out = src.get_spatial_series("uwind", times, on="node")
    np.testing.assert_allclose(out, np.array([[5.5]]), rtol=1e-5)


def test_metforce_source_outside_bbox_nearest_fallback(tmp_path: Path) -> None:
    """Points outside the source bbox must fall back to nearest, not NaN."""
    times = pd.date_range("2020-01-01", periods=1, freq="1h")
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0])
    fpath = tmp_path / "mf.nc"
    _make_synthetic_metforce(fpath, times=times, lats=lats, lons=lons)

    # Outside the bbox on three of four sides.
    tgt_lon = np.array([-0.1, 1.1, 0.5])
    tgt_lat = np.array([0.5, 0.5, 1.1])
    src = MetforceGriddedSource(
        fpath,
        target_node_lon=tgt_lon,
        target_node_lat=tgt_lat,
        target_elem_lon=tgt_lon,
        target_elem_lat=tgt_lat,
        fallback_method="nearest",
    )
    out = src.get_spatial_series("uwind", times, on="node")
    assert np.isfinite(out).all(), "nearest fallback should suppress NaNs"


def test_metforce_source_no_fallback_propagates_nan(tmp_path: Path) -> None:
    times = pd.date_range("2020-01-01", periods=1, freq="1h")
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0])
    fpath = tmp_path / "mf.nc"
    _make_synthetic_metforce(fpath, times=times, lats=lats, lons=lons)

    tgt_lon = np.array([-1.0])
    tgt_lat = np.array([-1.0])
    src = MetforceGriddedSource(
        fpath,
        target_node_lon=tgt_lon,
        target_node_lat=tgt_lat,
        target_elem_lon=tgt_lon,
        target_elem_lat=tgt_lat,
        fallback_method=None,
    )
    out = src.get_spatial_series("uwind", times, on="node")
    assert np.isnan(out).all()


def test_metforce_source_precip_unit_conversion(tmp_path: Path) -> None:
    """mm/h in metforce → m/s in FVCOM (factor 1/3.6e6)."""
    times = pd.date_range("2020-01-01", periods=1, freq="1h")
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0])
    fpath = tmp_path / "mf.nc"
    _make_synthetic_metforce(
        fpath,
        times=times,
        lats=lats,
        lons=lons,
        constants={"PRECIP": 3.6},  # 3.6 mm/h → 1e-6 m/s exactly
    )
    tgt = np.array([0.5])
    src = MetforceGriddedSource(
        fpath,
        target_node_lon=tgt,
        target_node_lat=tgt,
        target_elem_lon=tgt,
        target_elem_lat=tgt,
    )
    out = src.get_spatial_series("precip", times, on="node")
    np.testing.assert_allclose(out, np.array([[1.0e-6]]), rtol=1e-5)


def test_metforce_source_missing_variable_raises(tmp_path: Path) -> None:
    times = pd.date_range("2020-01-01", periods=1, freq="1h")
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0])
    # Write only one variable instead of the full set → should error on init.
    arr = np.zeros((1, 2, 2), dtype=np.float32)
    ds = xr.Dataset(
        {"U10": (("time", "lat", "lon"), arr)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    fpath = tmp_path / "mf_partial.nc"
    ds.to_netcdf(fpath)
    with pytest.raises(KeyError, match="V10"):
        MetforceGriddedSource(
            fpath,
            target_node_lon=np.array([0.5]),
            target_node_lat=np.array([0.5]),
            target_elem_lon=np.array([0.5]),
            target_elem_lat=np.array([0.5]),
        )


# ------------------------------------------------------------------
# 2. Generator integration: tiny grid + synthetic metforce → .nc
# ------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore:Ambiguous reference date")
def test_metforce_generator_smoke(tmp_path: Path, tiny_grid: Path) -> None:
    times = pd.date_range("2020-01-01", periods=3, freq="1h")
    lats = np.linspace(-1.0, 2.0, 6)  # bbox encloses the unit-square grid
    lons = np.linspace(-1.0, 2.0, 6)
    mfpath = tmp_path / "synth_metforce.nc"
    _make_synthetic_metforce(mfpath, times=times, lats=lats, lons=lons)

    out_nc = tmp_path / "tb_met.nc"
    gen = MetNetCDFGenerator(
        grid_nc=tiny_grid,
        start="2020-01-01T00:00:00Z",
        end="2020-01-01T02:00:00Z",
        dt_seconds=3600,
        utm_zone=54,
        metforce_file=mfpath,
    )
    gen.write(out_nc)
    assert out_nc.exists()

    with nc.Dataset(out_nc, "r") as ds:
        # Required FVCOM forcing variables
        for v in (
            "uwind_speed",
            "vwind_speed",
            "air_temperature",
            "relative_humidity",
            "air_pressure",
            "short_wave",
            "long_wave",
            "Precipitation",
            "cloud_cover",
        ):
            assert v in ds.variables, f"missing variable {v!r}"
        # Time axis is hourly MJD over the requested window
        t = ds.variables["time"][:]
        assert t.shape == (3,)
        # Global "infos" attribute reflects metforce provenance
        assert "metforce" in ds.getncattr("infos").lower()


def test_metforce_generator_mutually_exclusive_with_mpos(
    tmp_path: Path, tiny_grid: Path
) -> None:
    times = pd.date_range("2020-01-01", periods=1, freq="1h")
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0])
    mfpath = tmp_path / "mf.nc"
    _make_synthetic_metforce(mfpath, times=times, lats=lats, lons=lons)

    with pytest.raises(ValueError, match="mutually exclusive"):
        MetNetCDFGenerator(
            grid_nc=tiny_grid,
            start="2020-01-01T00:00:00Z",
            end="2020-01-01T01:00:00Z",
            utm_zone=54,
            metforce_file=mfpath,
            mpos_dir=tmp_path,  # any path; the validation runs first
        )
