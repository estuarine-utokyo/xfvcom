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


# ---------------------------------------------------------------------------
# review round 2 (gpt-6.1-sol) #1: nearest ocean measured from the requested position
# ---------------------------------------------------------------------------


def _basic_with_mask(path, mask):
    jm, im = mask.shape
    dx = 1.0 / 36.0
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("lat", jm)
        ds.createDimension("lon", im)
        ds.createDimension("level", 2)
        ds.createVariable("lat", "f8", ("lat",))[:] = 17.0 + np.arange(jm) * dx
        ds.createVariable("lon", "f8", ("lon",))[:] = 117.0 + np.arange(im) * dx
        ds.createVariable("h", "f4", ("lat", "lon"))[:] = np.where(mask, 100.0, 0.0)
        ds.createVariable("mask", "i1", ("lat", "lon"))[:] = mask.astype(np.int8)
        ds.createVariable("ZZ", "f4", ("level", "lat", "lon"))[:] = -1.0
        ds.grid_origin_lon, ds.grid_origin_lat = 117.0, 17.0
        ds.grid_dx_deg = ds.grid_dy_deg = dx
    return path


def test_nearest_ocean_searches_the_whole_radius(tmp_path):
    # The first non-empty square (Chebyshev radius 9) holds (j0+8, i0+9), squared distance 145;
    # (j0, i0+10) at radius 10 is nearer (squared distance 100) and must win.
    mask = np.zeros((40, 40), dtype=bool)
    j0, i0 = 15, 15
    mask[j0 + 8, i0 + 9] = True
    mask[j0, i0 + 10] = True
    with JcopeGrid(_basic_with_mask(tmp_path / "b.nc", mask)) as g:
        j, i = g.find_nearest_ocean(17.0 + j0 * g.dy, 117.0 + i0 * g.dx)
        assert (int(j), int(i)) == (j0, i0 + 10)


def test_nearest_ocean_uses_the_fractional_position(tmp_path):
    # Land at the rounded cell (5, 5); ocean at (5, 4) and (5, 6), equally far from the rounded
    # cell. The requested point lies at i = 5.4, so (5, 6) is nearer.
    mask = np.zeros((11, 11), dtype=bool)
    mask[5, 4] = mask[5, 6] = True
    with JcopeGrid(_basic_with_mask(tmp_path / "b.nc", mask)) as g:
        j, i = g.find_nearest_ocean(17.0 + 5 * g.dy, 117.0 + 5.4 * g.dx)
        assert (int(j), int(i)) == (5, 6)
        j, i = g.find_nearest_ocean(17.0 + 5 * g.dy, 117.0 + 4.6 * g.dx)
        assert (int(j), int(i)) == (5, 4)


# ---------------------------------------------------------------------------
# review round 3 (gpt-6.1-sol)
# ---------------------------------------------------------------------------


def _unit_basic(path, mask, *, origin=(0.0, 0.0), spacing=(1.0, 1.0), shift_lon=0.0):
    """basic.nc on a unit grid (exact fractional positions); attributes may disagree with the axes."""
    jm, im = mask.shape
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("lat", jm)
        ds.createDimension("lon", im)
        ds.createDimension("level", 2)
        ds.createVariable("lat", "f8", ("lat",))[:] = np.arange(jm, dtype=float)
        ds.createVariable("lon", "f8", ("lon",))[:] = np.arange(im, dtype=float)
        ds.createVariable("h", "f4", ("lat", "lon"))[:] = np.where(mask, 100.0, 0.0)
        ds.createVariable("mask", "i1", ("lat", "lon"))[:] = mask.astype(np.int8)
        ds.createVariable("ZZ", "f4", ("level", "lat", "lon"))[:] = -1.0
        ds.grid_origin_lat, ds.grid_origin_lon = origin[0], origin[1] + shift_lon
        ds.grid_dy_deg, ds.grid_dx_deg = spacing
    return path


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(shift_lon=1.0), "lon axis is not grid_origin"),
        (dict(spacing=(1.0, 2.0)), "lon axis is not grid_origin"),
        (dict(spacing=(0.0, 1.0)), "grid_dy_deg must be finite and positive"),
        (dict(spacing=(1.0, np.nan)), "grid_dx_deg must be finite and positive"),
        (dict(origin=(np.inf, 0.0)), "grid_origin_lat must be finite"),
    ],
)
def test_round3_attributes_must_describe_the_axes(tmp_path, kw, match):
    # #12: shifting only grid_origin_lon used to pair the payload with another cell's geometry
    p = _unit_basic(tmp_path / "b.nc", np.ones((4, 5), dtype=bool), **kw)
    with pytest.raises(ValueError, match=match):
        JcopeGrid(p)


def test_round3_real_basic_nc_still_loads():
    p = "/home/pj24001722/share/Data/jcopetda/grid/basic.nc"
    import os

    if not os.path.isfile(p):
        pytest.skip("real basic.nc not available")
    with JcopeGrid(p) as g:
        assert (g.jm, g.im) == (1190, 1190)


@pytest.mark.parametrize(
    "lat, lon, match",
    [
        ([np.nan], [1.0], "finite"),
        ([2.0], [1.0, 3.0], "same length"),
        ([[1.0, 2.0]], [[1.0, 2.0]], "1-D"),
        (np.ma.masked_array([1.0], mask=[True]), [1.0], "masked"),
    ],
)
def test_round3_lookup_coordinates_are_validated(tmp_path, lat, lon, match):
    # #17
    with JcopeGrid(_unit_basic(tmp_path / "b.nc", np.ones((4, 5), dtype=bool))) as g:
        with pytest.raises(ValueError, match=match):
            g.find_nearest_ocean(lat, lon)
        with pytest.raises(ValueError, match=match):
            g.index_of(lat, lon)


@pytest.mark.parametrize("radius", [0, -1, 1.5, True, "3"])
def test_round3_search_radius_is_validated(tmp_path, radius):
    # #17
    with JcopeGrid(_unit_basic(tmp_path / "b.nc", np.ones((4, 5), dtype=bool))) as g:
        with pytest.raises(ValueError, match="max_radius_cells"):
            g.find_nearest_ocean(1.0, 1.0, max_radius_cells=radius)


def test_round3_ties_go_to_the_first_cell_in_j_i_order(tmp_path):
    # #20: (1.5, 1.5) is equally far from (1,1), (1,2), (2,1), (2,2); the rounded-cell shortcut
    # returned (2, 2)
    with JcopeGrid(_unit_basic(tmp_path / "b.nc", np.ones((4, 5), dtype=bool))) as g:
        j, i = g.find_nearest_ocean(1.5, 1.5)
        assert (int(j), int(i)) == (1, 1)
        j, i = g.find_nearest_ocean([1.5, 2.2], [1.5, 3.0])
        assert j.tolist() == [1, 2] and i.tolist() == [1, 3]
