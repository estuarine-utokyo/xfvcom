# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Tests for :mod:`xfvcom.io.jcope_grid`.

The real archive (``basic.nc``) is ~268 MB and lives on the group-shared
filesystem; tests here build a small synthetic NetCDF in the same shape so
they run quickly and offline.
"""

from __future__ import annotations

import netCDF4 as nc
import numpy as np
import pytest

from xfvcom.io.jcope_grid import JcopeGrid

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_basic_nc(tmp_path):
    """Build a small synthetic basic.nc and return its path.

    Grid is 10x8 (lon x lat) with 5 vertical levels; ocean is the eastern
    half of the domain to give find_nearest_ocean something to find.
    """
    path = tmp_path / "basic.nc"

    im, jm, km = 10, 8, 5
    lon_origin, lat_origin = 117.0, 17.0
    dx = dy = 1.0 / 36.0

    lon = lon_origin + np.arange(im) * dx
    lat = lat_origin + np.arange(jm) * dy

    # Bathymetry: shallow ramp, deepest at the east edge
    h = np.maximum(
        np.tile(np.arange(im) * 50.0, (jm, 1)).astype(np.float32),
        0.0,
    )
    # West half (i=0..4) is "land" (mask=0), east half is ocean
    mask = np.zeros((jm, im), dtype=np.int8)
    mask[:, 5:] = 1
    h[mask == 0] = 0.0

    # Synthetic Z, ZZ, DZ: uniform layers down to -h
    Z = np.zeros((km, jm, im), dtype=np.float32)
    ZZ = np.zeros((km, jm, im), dtype=np.float32)
    DZ = np.zeros((km, jm, im), dtype=np.float32)
    for j in range(jm):
        for i in range(im):
            depth = -h[j, i]
            for k in range(km):
                Z[k, j, i] = depth * k / (km - 1)
                ZZ[k, j, i] = depth * (k + 0.5) / (km - 1)
                DZ[k, j, i] = -depth / (km - 1)

    with nc.Dataset(path, "w") as ds:
        ds.createDimension("lat", jm)
        ds.createDimension("lon", im)
        ds.createDimension("level", km)

        v_lat = ds.createVariable("lat", "f8", ("lat",))
        v_lat[:] = lat
        v_lon = ds.createVariable("lon", "f8", ("lon",))
        v_lon[:] = lon
        v_lvl = ds.createVariable("level", "i4", ("level",))
        v_lvl[:] = np.arange(km, dtype=np.int32)

        v_Z = ds.createVariable("Z", "f4", ("level", "lat", "lon"))
        v_Z[:] = Z
        v_ZZ = ds.createVariable("ZZ", "f4", ("level", "lat", "lon"))
        v_ZZ[:] = ZZ
        v_DZ = ds.createVariable("DZ", "f4", ("level", "lat", "lon"))
        v_DZ[:] = DZ
        v_h = ds.createVariable("h", "f4", ("lat", "lon"))
        v_h[:] = h
        v_mask = ds.createVariable("mask", "i1", ("lat", "lon"))
        v_mask[:] = mask

        ds.grid_origin_lon = lon_origin
        ds.grid_origin_lat = lat_origin
        ds.grid_dx_deg = dx
        ds.grid_dy_deg = dy

    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJcopeGridConstruction:
    def test_dimensions_loaded(self, synthetic_basic_nc):
        g = JcopeGrid(synthetic_basic_nc)
        try:
            assert g.im == 10
            assert g.jm == 8
            assert g.km == 5
        finally:
            g.close()

    def test_origin_and_spacing(self, synthetic_basic_nc):
        with JcopeGrid(synthetic_basic_nc) as g:
            assert g.lon_origin == pytest.approx(117.0)
            assert g.lat_origin == pytest.approx(17.0)
            assert g.dx == pytest.approx(1.0 / 36.0)
            assert g.dy == pytest.approx(1.0 / 36.0)

    def test_mask_loaded_as_bool(self, synthetic_basic_nc):
        with JcopeGrid(synthetic_basic_nc) as g:
            assert g.mask.dtype == bool
            assert g.mask.shape == (8, 10)
            assert g.mask[:, :5].sum() == 0
            assert g.mask[:, 5:].all()


class TestIndexOf:
    def test_scalar_round_trip(self, synthetic_basic_nc):
        with JcopeGrid(synthetic_basic_nc) as g:
            lat0 = 17.0 + 3 * g.dy
            lon0 = 117.0 + 7 * g.dx
            j, i = g.index_of(lat0, lon0)
            assert int(j) == 3
            assert int(i) == 7

    def test_array_input(self, synthetic_basic_nc):
        with JcopeGrid(synthetic_basic_nc) as g:
            lats = np.array([17.0, 17.0 + 4 * g.dy])
            lons = np.array([117.0 + 1 * g.dx, 117.0 + 8 * g.dx])
            j, i = g.index_of(lats, lons)
            np.testing.assert_array_equal(j, [0, 4])
            np.testing.assert_array_equal(i, [1, 8])

    def test_clip_to_extent(self, synthetic_basic_nc):
        with JcopeGrid(synthetic_basic_nc) as g:
            j, i = g.index_of(20.0, 200.0)
            assert int(j) == g.jm - 1
            assert int(i) == g.im - 1


class TestFindNearestOcean:
    def test_already_ocean(self, synthetic_basic_nc):
        with JcopeGrid(synthetic_basic_nc) as g:
            lat0 = 17.0 + 2 * g.dy
            lon0 = 117.0 + 7 * g.dx
            j, i = g.find_nearest_ocean(lat0, lon0)
            assert int(j) == 2
            assert int(i) == 7

    def test_snap_from_land(self, synthetic_basic_nc):
        with JcopeGrid(synthetic_basic_nc) as g:
            lat0 = 17.0 + 3 * g.dy
            lon0 = 117.0 + 1 * g.dx  # land
            j, i = g.find_nearest_ocean(lat0, lon0)
            assert int(j) == 3
            assert int(i) == 5  # snaps to first ocean column to the east

    def test_unreachable_raises(self, synthetic_basic_nc):
        with JcopeGrid(synthetic_basic_nc) as g:
            with pytest.raises(ValueError, match="No ocean cell"):
                g.find_nearest_ocean(17.0, 117.0, max_radius_cells=1)


class TestDepthAndProfile:
    def test_h_at(self, synthetic_basic_nc):
        with JcopeGrid(synthetic_basic_nc) as g:
            # Eastmost ocean column at i=9 has h = 9 * 50 = 450
            lat0 = 17.0 + 2 * g.dy
            lon0 = 117.0 + 9 * g.dx
            assert g.h_at(lat0, lon0) == pytest.approx(450.0)

    def test_profile_shape_and_sign(self, synthetic_basic_nc):
        with JcopeGrid(synthetic_basic_nc) as g:
            lat0 = 17.0 + 2 * g.dy
            lon0 = 117.0 + 9 * g.dx
            Z = g.profile("Z", lat0, lon0)
            ZZ = g.profile("ZZ", lat0, lon0)
            DZ = g.profile("DZ", lat0, lon0)
            assert Z.shape == (g.km,)
            assert Z[0] == pytest.approx(0.0)
            assert Z[-1] == pytest.approx(-450.0)
            assert (Z <= 0).all()
            assert (ZZ <= 0).all()
            assert (DZ > 0).all()

    def test_profile_multipoint(self, synthetic_basic_nc):
        with JcopeGrid(synthetic_basic_nc) as g:
            lats = np.array([17.0 + 2 * g.dy, 17.0 + 5 * g.dy])
            lons = np.array([117.0 + 9 * g.dx, 117.0 + 6 * g.dx])
            Z = g.profile("Z", lats, lons)
            assert Z.shape == (2, g.km)

    def test_profile_invalid_kind(self, synthetic_basic_nc):
        with JcopeGrid(synthetic_basic_nc) as g:
            with pytest.raises(ValueError, match="kind"):
                g.profile("X", 17.0, 117.0)
