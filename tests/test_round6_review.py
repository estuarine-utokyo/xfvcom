# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Regression tests for review round 6 (gpt-6.1-sol, scratch/reviews/fulldomain/r6_final.md).

#1 connectivity base in the I/O consumers, #2 geographic grids and metric areas,
#3 three-element meshes in gtsz_builder.load_mesh, #4 permissions of written files,
#6 geographic control-volume polygons, #7 stale topology caches, #8 area selectors,
#9 JCOPE static mask, #10 elems2nodes on integer fields.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest
import xarray as xr

import xfvcom.io.jcope_obc_generator as gen_mod
from xfvcom.grid.connectivity import zero_based_connectivity
from xfvcom.grid.geo_utils import utm_to_lonlat
from xfvcom.grid.grid_obj import FvcomGrid, get_grid
from xfvcom.grid.interpolation import elems2nodes
from xfvcom.io.core import FvcomDataLoader
from xfvcom.io.input_loader import FvcomInputLoader
from xfvcom.io.jcope_grid import JcopeGrid
from xfvcom.io.met_nc_generator import MetNetCDFGenerator

# ---------------------------------------------------------------------------
# #1 one connectivity normalisation for every consumer
# ---------------------------------------------------------------------------


def test_round6_data_loader_honours_zero_based_start_index(tmp_path):
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    g = FvcomGrid(x=x, y=y, nv=np.array([[0, 1, 2], [1, 3, 2]]).T)
    p = tmp_path / "g.nc"
    g.to_xarray().to_netcdf(p)
    ds = FvcomDataLoader(str(p), utm2geo=False).ds
    np.testing.assert_array_equal(ds["nv_zero"].values, [[0, 1, 2], [1, 3, 2]])
    # node -> element averaging now samples the right nodes
    vals = np.array([10.0, 20.0, 30.0, 1000.0])
    np.testing.assert_allclose(vals[ds["nv_zero"].values].mean(axis=1), [20.0, 350.0])


def test_round6_data_loader_keeps_one_based_fvcom_output(tmp_path):
    # FVCOM output: one-based, no start_index attribute (today's behaviour)
    p = tmp_path / "out.nc"
    xr.Dataset(
        {
            "x": ("node", np.arange(4.0)),
            "y": ("node", np.arange(4.0)),
            "nv": (("three", "nele"), np.array([[1, 2], [2, 4], [3, 3]])),
        }
    ).to_netcdf(p)
    ds = FvcomDataLoader(str(p), utm2geo=False).ds
    np.testing.assert_array_equal(ds["nv_zero"].values, [[0, 1, 2], [1, 3, 2]])


def test_round6_data_loader_refuses_out_of_mesh_indices(tmp_path):
    p = tmp_path / "bad.nc"
    xr.Dataset(
        {
            "x": ("node", np.arange(4.0)),
            "y": ("node", np.arange(4.0)),
            "nv": (
                ("three", "nele"),
                np.array([[0], [1], [2]]),
            ),  # zero-based, unstated
        }
    ).to_netcdf(p)
    with pytest.raises(ValueError, match="outside the mesh"):
        FvcomDataLoader(str(p), utm2geo=False)


_DAT_SKIPS_NODE_1 = """Node Number = 5
Cell Number = 2
1 2 3 4 1
2 3 5 4 1
1 400000.0 3900000.0 1.0
2 400000.0 3900000.0 1.0
3 401000.0 3900000.0 1.0
4 400000.0 3901000.0 1.0
5 401000.0 3901000.0 1.0
"""


def test_round6_input_loader_shifts_zero_based_nv_that_omits_node_zero(tmp_path):
    p = tmp_path / "skip_grd.dat"
    p.write_text(_DAT_SKIPS_NODE_1)
    ld = FvcomInputLoader(grid_path=p, utm_zone=54)
    np.testing.assert_array_equal(ld.ds["nv"].values.T, [[2, 3, 4], [3, 5, 4]])
    np.testing.assert_array_equal(ld.ds["nv_zero"].values, [[1, 2, 3], [2, 4, 3]])


@pytest.mark.filterwarnings("ignore:Ambiguous reference date")
def test_round6_met_writer_shifts_zero_based_nv_that_omits_node_zero(tmp_path):
    p = tmp_path / "skip_grd.dat"
    p.write_text(_DAT_SKIPS_NODE_1)
    out = tmp_path / "met.nc"
    MetNetCDFGenerator(
        grid_nc=p,
        start="2025-01-01T00:00:00Z",
        end="2025-01-01T01:00:00Z",
        dt_seconds=3600,
        utm_zone=54,
        northern=True,
        uwind=1.0,
        vwind=0.0,
    ).write(out)
    with nc.Dataset(out) as ds:
        np.testing.assert_array_equal(ds["nv"][:].T, [[2, 3, 4], [3, 5, 4]])


def test_round6_helper_precedence_and_inference():
    da = xr.DataArray(np.array([[1], [2], [3]]), dims=("three", "nele"))
    assert zero_based_connectivity(da)[:, 0].tolist() == [0, 1, 2]  # min_is_one
    assert zero_based_connectivity(da, fallback=0)[:, 0].tolist() == [1, 2, 3]
    assert zero_based_connectivity(da, fallback="min_is_zero")[:, 0].tolist() == [
        0,
        1,
        2,
    ]
    da.attrs["start_index"] = 0
    assert zero_based_connectivity(da, fallback=1)[:, 0].tolist() == [1, 2, 3]
    assert zero_based_connectivity(da, start_index=1)[:, 0].tolist() == [0, 1, 2]
    # orientation follows the names, not the shape
    t = xr.DataArray(np.array([[0, 1, 2], [1, 3, 2]]), dims=("nele", "three"))
    assert zero_based_connectivity(t, fallback=0).shape == (3, 2)
    with pytest.raises(ValueError, match="outside the mesh"):
        zero_based_connectivity(t, fallback=0, n_nodes=3)


# ---------------------------------------------------------------------------
# #2 geographic-only datasets do not yield "m^2" in degrees^2
# ---------------------------------------------------------------------------


def _geo_only_ds():
    return xr.Dataset(
        {
            "lon": ("node", np.array([139.0, 139.01, 139.0])),
            "lat": ("node", np.array([35.0, 35.0, 35.01])),
            "nv": (("three", "nele"), np.array([[1], [2], [3]])),
        }
    )


def test_round6_geographic_grid_refuses_metric_areas():
    g = get_grid(_geo_only_ds())
    assert g.geographic
    for call in (
        lambda: g.calculate_node_area(),
        lambda: g.calculate_element_area(),
        lambda: g.calculate_node_area_median_dual(),
    ):
        with pytest.raises(ValueError, match="degrees"):
            call()
    # plotting-type helpers still work, and the flag survives a round trip
    assert len(g.get_node_element_boundaries([1])) == 3
    assert FvcomGrid.from_dataset(g.to_xarray()).geographic


def test_round6_projected_grid_is_not_geographic():
    x = np.array([0.0, 1.0, 0.0])
    g = FvcomGrid(x=x, y=np.array([0.0, 0.0, 1.0]), nv=np.array([[0, 1, 2]]).T)
    assert not g.geographic
    assert g.calculate_element_area().tolist() == [0.5]


# ---------------------------------------------------------------------------
# #3 gtsz_builder.load_mesh with exactly three elements
# ---------------------------------------------------------------------------


def test_round6_load_mesh_three_elements(tmp_path):
    from xfvcom.grid.gtsz_builder import load_mesh

    x = [0.0, 1.0, 0.0, 1.0, 0.0]
    y = [0.0, 0.0, 1.0, 1.0, 2.0]
    h = [100.0, 100.0, 100.0, 100.0, 5.0]
    tris = [[1, 2, 3], [2, 4, 3], [3, 4, 5]]
    grd = tmp_path / "m_grd.dat"
    grd.write_text(
        "Node Number = 5\nCell Number = 3\n"
        + "".join(f"{i + 1} {a} {b} {c} 1\n" for i, (a, b, c) in enumerate(tris))
        + "".join(f"{i + 1} {x[i]} {y[i]} {h[i]}\n" for i in range(5))
    )
    dep = tmp_path / "m_dep.dat"
    dep.write_text(
        "Node Number = 5\n" + "".join(f"{x[i]} {y[i]} {h[i]}\n" for i in range(5))
    )
    m = load_mesh(grd, dep)
    np.testing.assert_array_equal(m.nv, [[0, 1, 2], [1, 3, 2], [2, 3, 4]])


# ---------------------------------------------------------------------------
# #4 permissions of files written through _atomic_write
# ---------------------------------------------------------------------------


def _write_bytes(path: Path) -> None:
    path.write_bytes(b"x")


def test_round6_unknown_umask_keeps_restrictive_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(gen_mod, "_current_umask", lambda: None)
    out = gen_mod._atomic_write(tmp_path / "new.nc", _write_bytes)
    assert stat.S_IMODE(out.stat().st_mode) == 0o600


def test_round6_replaced_destination_keeps_its_mode(tmp_path, monkeypatch):
    dest = tmp_path / "old.nc"
    dest.write_bytes(b"old")
    os.chmod(dest, 0o600)
    monkeypatch.setattr(gen_mod, "_current_umask", lambda: 0o022)
    gen_mod._atomic_write(dest, _write_bytes)
    assert stat.S_IMODE(dest.stat().st_mode) == 0o600
    assert dest.read_bytes() == b"x"
    os.chmod(dest, 0o640)
    gen_mod._atomic_write(dest, _write_bytes)
    assert stat.S_IMODE(dest.stat().st_mode) == 0o640


def test_round6_new_file_follows_known_umask(tmp_path, monkeypatch):
    monkeypatch.setattr(gen_mod, "_current_umask", lambda: 0o077)
    out = gen_mod._atomic_write(tmp_path / "a.nc", _write_bytes)
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    monkeypatch.setattr(gen_mod, "_current_umask", lambda: 0o022)
    out = gen_mod._atomic_write(tmp_path / "b.nc", _write_bytes)
    assert stat.S_IMODE(out.stat().st_mode) == 0o644


def test_round6_unreadable_procfs_reports_unknown_umask(monkeypatch):
    import builtins

    real_open = builtins.open

    def _no_proc(file, *a, **k):
        if str(file) == "/proc/self/status":
            raise OSError("no procfs")
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", _no_proc)
    assert gen_mod._current_umask() is None


# ---------------------------------------------------------------------------
# #6 control-volume polygons use the grid's projection
# ---------------------------------------------------------------------------


def _utm_square(zone=54, northern=True, *, with_zone=True):
    x = 400000.0 + 1000.0 * np.array([0.0, 1.0, 0.0, 1.0])
    y = 3900000.0 + 1000.0 * np.array([0.0, 0.0, 1.0, 1.0])
    lon, lat = utm_to_lonlat(x, y, zone=zone, northern=northern)
    return FvcomGrid(
        x=x,
        y=y,
        nv=np.array([[0, 1, 2], [1, 3, 2]]).T,
        zone=zone if with_zone else None,
        northern=northern,
        lon=lon,
        lat=lat,
    )


def _to_utm(lon, lat, northern):
    from pyproj import Transformer

    epsg = (32600 if northern else 32700) + 54
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    return t.transform(lon, lat)


@pytest.mark.parametrize("northern", [True, False])
def test_round6_control_volumes_are_projected_exactly(northern):
    g = _utm_square(northern=northern)
    for node in (1, 2, 3, 4):
        poly = np.asarray(g.get_node_control_volume(node))
        ref_lon, ref_lat = utm_to_lonlat(
            poly[:, 0], poly[:, 1], zone=54, northern=northern
        )
        got = np.asarray(g.get_node_control_volumes([node])[0])
        # back to metres: the vertices agree to far below a millimetre
        bx, by = _to_utm(got[:, 0], got[:, 1], northern)
        np.testing.assert_allclose(bx, poly[:, 0], atol=1e-6)
        np.testing.assert_allclose(by, poly[:, 1], atol=1e-6)
        np.testing.assert_allclose(got[:, 0], ref_lon, rtol=0, atol=1e-12)
        np.testing.assert_allclose(got[:, 1], ref_lat, rtol=0, atol=1e-12)


def test_round6_control_volumes_without_zone_are_refused():
    g = _utm_square(with_zone=False)
    with pytest.raises(ValueError, match="UTM zone"):
        g.get_node_control_volumes([1])


# ---------------------------------------------------------------------------
# #7 cached topology follows nv
# ---------------------------------------------------------------------------


def _fan(n=6, r=10.0):
    ang = 2 * np.pi * np.arange(n) / n
    x = np.concatenate([[0.0], r * np.cos(ang)])
    y = np.concatenate([[0.0], r * np.sin(ang)])
    tris = [[0, 1 + k, 1 + (k + 1) % n] for k in range(n)]
    return x, y, np.array(tris).T


@pytest.mark.parametrize("how", ["replace", "inplace"])
def test_round6_editing_nv_invalidates_topology_cache(how):
    x, y, nv = _fan()
    g = FvcomGrid(x=x, y=y, nv=nv.copy())
    g.calculate_node_area_median_dual([1])  # fills the caches
    if how == "replace":
        g.nv = nv[:, :5].copy()
    else:
        # same shape, one triangle collapsed onto an existing one
        g.nv[:, 5] = g.nv[:, 4]
    fresh = FvcomGrid(x=x, y=y, nv=g.nv.copy())
    assert g.calculate_node_area_median_dual([1]) == pytest.approx(
        fresh.calculate_node_area_median_dual([1]), rel=1e-12
    )
    assert g._get_boundary_edges() == fresh._get_boundary_edges()
    assert g._get_boundary_nodes() == fresh._get_boundary_nodes()


def test_round6_removing_a_fan_triangle_gives_the_fresh_area():
    x, y, nv = _fan()
    g = FvcomGrid(x=x, y=y, nv=nv.copy())
    g.calculate_node_area_median_dual([1])
    g.nv = nv[:, :5].copy()
    assert g.calculate_node_area_median_dual([1]) == pytest.approx(72.168784, rel=1e-6)


# ---------------------------------------------------------------------------
# #8 selectors
# ---------------------------------------------------------------------------


def _unit_triangle():
    return FvcomGrid(
        x=np.array([0.0, 1.0, 0.0]),
        y=np.array([0.0, 0.0, 1.0]),
        nv=np.array([[0, 1, 2]]).T,
    )


@pytest.mark.parametrize(
    "method",
    [
        "calculate_node_area",
        "calculate_node_area_median_dual",
        "get_node_element_boundaries",
        "get_node_control_volumes",
        "calculate_element_area",
    ],
)
def test_round6_selectors_refuse_fractions_masks_and_bad_base(method):
    g = _unit_triangle()
    f = getattr(g, method)
    with pytest.raises(ValueError, match="non-integral"):
        f([1.9])
    with pytest.raises(ValueError, match="masked"):
        f(np.ma.masked_array([1, 2], mask=[False, True]))
    with pytest.raises(ValueError, match="index_base"):
        f([1], index_base=2)
    with pytest.raises(ValueError, match="integers"):
        f(np.array([True]))
    # valid selectors still work, including integral floats and unmasked masked arrays
    f([1])
    f([1.0])
    f(np.ma.masked_array([1], mask=[False]))


def test_round6_single_node_control_volume_is_validated():
    g = _unit_triangle()
    with pytest.raises(ValueError, match="Invalid node index"):
        g.get_node_control_volume(1.9)
    with pytest.raises(ValueError, match="Invalid node index"):
        g.get_node_control_volume(4)
    assert g.get_node_control_volume(1.0) == g.get_node_control_volume(1)


def test_round6_duplicate_nodes_count_once_in_area_sums():
    g = _unit_triangle()
    one = g.calculate_node_area_median_dual([1])
    assert g.calculate_node_area_median_dual([1, 1]) == pytest.approx(one)
    assert g.calculate_node_area([1, 1]) == pytest.approx(g.calculate_node_area([1]))


# ---------------------------------------------------------------------------
# #9 JCOPE static mask must be binary
# ---------------------------------------------------------------------------


def _basic(path, mask_values, dtype="f4", fill=None):
    n = 2
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("lat", n)
        ds.createDimension("lon", n)
        ds.createDimension("level", 2)
        ds.createVariable("lat", "f8", ("lat",))[:] = np.arange(n, dtype=float)
        ds.createVariable("lon", "f8", ("lon",))[:] = np.arange(n, dtype=float)
        ds.createVariable("h", "f4", ("lat", "lon"))[:] = 10.0
        v = ds.createVariable("mask", dtype, ("lat", "lon"), fill_value=fill)
        v[:] = mask_values
        ds.createVariable("ZZ", "f4", ("level", "lat", "lon"))[:] = -1.0
        ds.grid_origin_lat = ds.grid_origin_lon = 0.0
        ds.grid_dy_deg = ds.grid_dx_deg = 1.0
    return path


def test_round6_nan_mask_is_refused(tmp_path):
    p = _basic(tmp_path / "a.nc", np.array([[np.nan, 1], [0, 0]], dtype="f4"))
    with pytest.raises(ValueError, match="missing"):
        JcopeGrid(p)


def test_round6_fill_value_mask_is_refused(tmp_path):
    # a decoded _FillValue becomes NaN in xarray
    vals = np.ma.masked_array([[0, 1], [0, 0]], mask=[[True, False], [False, False]])
    p = _basic(tmp_path / "b.nc", vals, dtype="i1", fill=-1)
    with pytest.raises(ValueError, match="missing"):
        JcopeGrid(p)


def test_round6_non_binary_mask_is_refused(tmp_path):
    p = _basic(tmp_path / "c.nc", np.array([[2, 1], [0, 0]]), dtype="i1")
    with pytest.raises(ValueError, match="only 0 and 1"):
        JcopeGrid(p)


def test_round6_binary_mask_still_loads(tmp_path):
    p = _basic(tmp_path / "d.nc", np.array([[1, 1], [0, 0]]), dtype="i1")
    with JcopeGrid(p) as g:
        assert g.mask.dtype == bool
        np.testing.assert_array_equal(g.mask, [[True, True], [False, False]])


# ---------------------------------------------------------------------------
# #10 elems2nodes on integer fields, all supported ranks
# ---------------------------------------------------------------------------

_NV = np.array([[0, 1, 2], [1, 3, 2]]).T


def test_round6_elems2nodes_integer_ranks():
    exp = np.array([1.0, 1.5, 1.5, 2.0])
    out1 = elems2nodes(np.array([1, 2]), _NV, n_nodes=4)
    assert out1.dtype == np.float64
    np.testing.assert_allclose(out1, exp)
    out2 = elems2nodes(np.array([[1, 2], [3, 4]]), _NV, n_nodes=4)
    np.testing.assert_allclose(out2, [exp, exp + 2])
    out3 = elems2nodes(np.array([[[1, 2]], [[3, 4]]], dtype=np.int16), _NV, n_nodes=4)
    np.testing.assert_allclose(out3, [[exp], [exp + 2]])
    # float32 keeps its dtype, as before
    assert elems2nodes(np.array([1, 2], dtype=np.float32), _NV, 4).dtype == np.float32
