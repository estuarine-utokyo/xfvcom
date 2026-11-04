# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""FvcomGrid geometry and reconstruction (review round 5 #1-#6, gpt-6.1-sol)."""

from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr

from xfvcom.grid.dat_reader import read_dat
from xfvcom.grid.grid_obj import FvcomGrid

# ---------------------------------------------------------------------------
# #1 median-dual area does not depend on the coordinate origin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x0, y0", [(0.0, 0.0), (400000.0, 3900000.0), (-7.5e5, 9.99e6), (1e7, 1e7)]
)
def test_round5_median_dual_area_is_translation_invariant(x0, y0):
    x = x0 + 5 * np.array([0.0, 1.0, 0.0, 1.0])
    y = y0 + 5 * np.array([0.0, 0.0, 1.0, 1.0])
    nv = np.array([[0, 1, 2], [1, 3, 2]]).T
    g = FvcomGrid(x=x, y=y, nv=nv)
    total = g.calculate_node_area_median_dual()
    assert total == pytest.approx(25.0, rel=0, abs=1e-6)
    # every node, not only the sum: the same areas as at the origin
    ref = FvcomGrid(x=x - x0, y=y - y0, nv=nv)
    for n in range(1, 5):
        assert g.calculate_node_area_median_dual([n]) == pytest.approx(
            ref.calculate_node_area_median_dual([n]), rel=0, abs=1e-6
        )


def test_round5_interior_node_area_is_translation_invariant():
    # a fan of six triangles around node 0: the interior closure must still be removed
    ang = np.arange(6) * np.pi / 3
    for x0, y0 in ((0.0, 0.0), (400000.0, 3900000.0)):
        x = x0 + np.r_[0.0, 10 * np.cos(ang)]
        y = y0 + np.r_[0.0, 10 * np.sin(ang)]
        nv = np.array([[0, 1 + k, 1 + (k + 1) % 6] for k in range(6)]).T
        g = FvcomGrid(x=x, y=y, nv=nv)
        tri = 0.5 * 10 * 10 * np.sin(np.pi / 3)
        assert g.calculate_node_area_median_dual([0], index_base=0) == pytest.approx(
            6 * tri / 3, rel=1e-9
        )
        assert g.calculate_node_area_median_dual() == pytest.approx(6 * tri, rel=1e-9)


def test_round5_real_tokyo_bay_dual_areas_sum_to_the_mesh_area():
    p = os.path.expanduser("~/Github/TB-FVCOM/input/goto2023/grid/TokyoBay_grd.dat")
    if not os.path.isfile(p):
        pytest.skip("TokyoBay_grd.dat not available")
    g = FvcomGrid.from_dat(p, utm_zone=54)
    x, y = g.x[g.nv], g.y[g.nv]
    tri = (
        0.5
        * np.abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])).sum()
    )
    assert g.calculate_node_area_median_dual() == pytest.approx(tri, rel=1e-9)


# ---------------------------------------------------------------------------
# #2 orientation by dimension names
# ---------------------------------------------------------------------------


def _strip_mesh():
    x = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0, 2.0])
    rows = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]])
    return FvcomGrid(x=x, y=y, nv=rows.T), rows


def test_round5_three_element_connectivity_is_oriented_by_names():
    g, rows = _strip_mesh()
    ds = g.to_xarray().transpose("nele", "three", "node")
    assert ds["nv"].dims == ("nele", "three")
    back = FvcomGrid.from_dataset(ds)
    assert back.nv.T.tolist() == rows.tolist()
    np.testing.assert_allclose(back.calculate_element_area(), [0.5, 0.5, 0.5])
    np.testing.assert_allclose(
        back.calculate_element_area(), g.calculate_element_area()
    )


def test_round5_ambiguous_unnamed_connectivity_is_refused():
    g, rows = _strip_mesh()
    ds = xr.Dataset(
        {
            "x": ("node", g.x),
            "y": ("node", g.y),
            "nv": (("a", "b"), rows.T, {"start_index": 0}),
        }
    )
    with pytest.raises(ValueError, match="ambiguous"):
        FvcomGrid.from_dataset(ds)


@pytest.mark.parametrize("dims", [("nele", "foo"), ("foo", "three")])
def test_round5_named_connectivity_must_have_three_vertices(dims):
    x = np.arange(5.0)
    nv = np.zeros((2, 4), dtype=int) if dims[0] == "nele" else np.zeros((4, 2), int)
    ds = xr.Dataset({"x": ("node", x), "y": ("node", x), "nv": (dims, nv)})
    with pytest.raises(ValueError, match="length 3"):
        FvcomGrid.from_dataset(ds)


# ---------------------------------------------------------------------------
# #3 nv_zero keeps its zero base
# ---------------------------------------------------------------------------


def test_round5_nv_zero_fallback_keeps_zero_base():
    x = np.arange(5.0)
    ds = xr.Dataset(
        {
            "x": ("node", x),
            "y": ("node", x),
            "nv_zero": (("nele", "three"), [[1, 2, 3]]),
        }
    )
    assert FvcomGrid.from_dataset(ds, validate=False).nv[:, 0].tolist() == [1, 2, 3]
    # an explicit override still wins
    got = FvcomGrid.from_dataset(ds, validate=False, start_index=1)
    assert got.nv[:, 0].tolist() == [0, 1, 2]


# ---------------------------------------------------------------------------
# #4 values validated before integer conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "nv, match",
    [
        ([[0.9], [1.9], [2.9]], "non-integral"),
        ([[0.0], [np.nan], [2.0]], "non-finite"),
        ([[0.0], [np.inf], [2.0]], "non-finite"),
        ([[0 + 0j], [1 + 0j], [2 + 0j]], "real numeric"),
        ([[True], [False], [True]], "real numeric"),
        ([["0"], ["1"], ["2"]], "real numeric"),
    ],
)
def test_round5_connectivity_is_validated_before_conversion(nv, match):
    x = np.arange(5.0)
    ds = xr.Dataset(
        {"x": ("node", x), "y": ("node", x), "nv": (("three", "nele"), np.array(nv))}
    )
    with pytest.raises(ValueError, match=match):
        FvcomGrid.from_dataset(ds, start_index=0)


def test_round5_integral_float_connectivity_is_accepted():
    x = np.arange(5.0)
    ds = xr.Dataset(
        {
            "x": ("node", x),
            "y": ("node", x),
            "nv": (("three", "nele"), [[1.0], [2.0], [3.0]]),
        }
    )
    assert FvcomGrid.from_dataset(ds, start_index=0).nv[:, 0].tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# #5 projection metadata survives a round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zone, northern", [(54, True), (54, False), (1, False)])
def test_round5_projection_round_trip(zone, northern):
    g, _ = _strip_mesh()
    g.zone, g.northern = zone, northern
    back = FvcomGrid.from_dataset(g.to_xarray())
    assert (back.zone, back.northern) == (zone, northern)


def test_round5_projection_round_trip_through_netcdf(tmp_path):
    g, _ = _strip_mesh()
    g.zone, g.northern = 54, False
    p = tmp_path / "g.nc"
    g.to_xarray().to_netcdf(p)
    with xr.open_dataset(p) as ds:
        back = FvcomGrid.from_dataset(ds)
    assert (back.zone, back.northern) == (54, False)


def test_round5_datasets_without_projection_keep_the_defaults():
    g, _ = _strip_mesh()
    ds = g.to_xarray()
    ds.attrs.pop("utm_zone", None)
    ds.attrs.pop("utm_hemisphere", None)
    back = FvcomGrid.from_dataset(ds)
    assert (back.zone, back.northern) == (None, True)


@pytest.mark.parametrize(
    "attrs, match",
    [({"utm_zone": 61}, "utm_zone"), ({"utm_hemisphere": "N"}, "utm_hemisphere")],
)
def test_round5_bad_projection_attributes_are_refused(attrs, match):
    g, _ = _strip_mesh()
    ds = g.to_xarray()
    ds.attrs.update(attrs)
    with pytest.raises(ValueError, match=match):
        FvcomGrid.from_dataset(ds)


# ---------------------------------------------------------------------------
# #6 element records placed by id
# ---------------------------------------------------------------------------

_NODES4 = "1 0.0 0.0 0.0\n2 1.0 0.0 0.0\n3 0.0 1.0 0.0\n4 1.0 1.0 0.0\n"


def _dat2(path, elems):
    path.write_text("Node Number = 4\nCell Number = 2\n" + elems + _NODES4)
    return path


def test_round5_element_records_are_placed_by_their_id(tmp_path):
    p = _dat2(tmp_path / "g.dat", "2 2 4 3 1\n1 1 2 3 1\n")
    d = read_dat(p)
    assert d["nv"].T.tolist() == [[0, 1, 2], [1, 3, 2]]


@pytest.mark.parametrize(
    "elems, msg",
    [
        ("1 1 2 3 1\n1 2 4 3 1\n", "appears more than once"),
        ("1 1 2 3 1\n3 2 4 3 1\n", r"outside 1\.\.2"),
        ("0 1 2 3 1\n1 2 4 3 1\n", r"outside 1\.\.2"),
    ],
)
def test_round5_bad_element_record_ids_are_refused(tmp_path, elems, msg):
    with pytest.raises(ValueError, match=msg):
        read_dat(_dat2(tmp_path / "g.dat", elems))
