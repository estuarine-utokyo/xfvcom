# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Connectivity bounds of the FVCOM ``*_grd.dat`` reader (review round 3 #19, gpt-6.1-sol).

A node id 0 used to become index -1 and silently address the last node.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from xfvcom.grid.dat_reader import read_dat
from xfvcom.grid.grid_obj import FvcomGrid

_NODES = """1 400000.0 3900000.0 0.0
2 401000.0 3900000.0 0.0
3 400000.0 3901000.0 0.0
"""


def _dat(path, elem):
    path.write_text(f"Node Number = 3\nCell Number = 1\n1 {elem} 1\n" + _NODES)
    return path


@pytest.mark.parametrize("elem", ["0 2 3", "1 2 4", "-1 2 3"])
def test_out_of_range_node_ids_are_refused(tmp_path, elem):
    p = _dat(tmp_path / "bad_grd.dat", elem)
    with pytest.raises(ValueError, match=r"outside 1\.\.3"):
        read_dat(p)
    with pytest.raises(ValueError, match=r"outside 1\.\.3"):
        FvcomGrid.from_dat(p, utm_zone=54)


def test_valid_one_based_mesh_parses(tmp_path):
    g = FvcomGrid.from_dat(_dat(tmp_path / "ok_grd.dat", "1 2 3"), utm_zone=54)
    assert g.nv[:, 0].tolist() == [0, 1, 2]


def test_real_tokyo_bay_mesh_still_parses():
    p = os.path.expanduser("~/Github/TB-FVCOM/input/goto2023/grid/TokyoBay_grd.dat")
    if not os.path.isfile(p):
        pytest.skip("TokyoBay_grd.dat not available")
    d = read_dat(p)
    assert d["nv"].min() == 0 and d["nv"].max() == d["x"].size - 1
    assert np.isfinite(d["x"]).all()


# ---------------------------------------------------------------------------
# review round 4 (gpt-6.1-sol)
# ---------------------------------------------------------------------------


def _dat_nodes(path, nodes):
    path.write_text("Node Number = 3\nCell Number = 1\n1 1 2 3 1\n" + nodes)
    return path


def test_round4_node_records_are_placed_by_their_id(tmp_path):
    # #3: records ordered 2,1,3 used to be assigned by row position (x[node 1] became 401000)
    p = _dat_nodes(
        tmp_path / "r_grd.dat",
        "2 401000.0 3900000.0 0.0\n1 400000.0 3900000.0 0.0\n3 400000.0 3901000.0 0.0\n",
    )
    # jsasaki 2026-11-05: review round 9 #2: id placement is now the explicit permissive mode
    d = read_dat(p, allow_unordered=True)
    assert d["x"].tolist() == [400000.0, 401000.0, 400000.0]
    assert d["y"].tolist() == [3900000.0, 3900000.0, 3901000.0]


@pytest.mark.parametrize(
    "first, msg",
    [
        ("0", r"node record id 0 outside 1\.\.3"),
        ("4", r"id 4 outside"),
        ("2", "more than once"),
    ],
)
def test_round4_bad_node_record_ids_are_refused(tmp_path, first, msg):
    p = _dat_nodes(
        tmp_path / "b_grd.dat",
        f"{first} 400000.0 3900000.0 0.0\n2 401000.0 3900000.0 0.0\n"
        "3 400000.0 3901000.0 0.0\n",
    )
    # jsasaki 2026-11-05: review round 9 #2: id checks of the permissive mode; strict refuses too
    with pytest.raises(ValueError, match=msg):
        read_dat(p, allow_unordered=True)
    with pytest.raises(ValueError):
        read_dat(p)


def test_round4_from_dataset_honours_start_index():
    # #4: zero-based nv that does not use node 0 was shifted as if one-based
    import xarray as xr

    x = np.arange(5.0)
    nv = xr.DataArray(np.array([[1], [2], [3]]), dims=("three", "nele"))
    ds = xr.Dataset({"x": ("node", x), "y": ("node", x), "nv": nv})
    assert FvcomGrid.from_dataset(ds).nv[:, 0].tolist() == [0, 1, 2]  # inferred
    ds["nv"].attrs["start_index"] = 0
    assert FvcomGrid.from_dataset(ds).nv[:, 0].tolist() == [1, 2, 3]
    ds["nv"].attrs["start_index"] = 1
    assert FvcomGrid.from_dataset(ds).nv[:, 0].tolist() == [0, 1, 2]
    assert FvcomGrid.from_dataset(ds, start_index=0).nv[:, 0].tolist() == [1, 2, 3]
    ds["nv"].attrs["start_index"] = 2
    with pytest.raises(ValueError, match="start_index"):
        FvcomGrid.from_dataset(ds)
    # the export states its base, so a round trip keeps the geometry
    g = FvcomGrid.from_dataset(ds, start_index=0)
    out = g.to_xarray()
    assert out["nv"].attrs["start_index"] == 0
    assert FvcomGrid.from_dataset(out).nv[:, 0].tolist() == [1, 2, 3]
