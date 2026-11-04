# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Regression tests for review round 16 (gpt-6.1-sol, scratch/reviews/fulldomain/r16_final.md).

#1 next_obc_nodes scans neighbours in FVCOM's NBSN order (tge.F on the reversed
   connectivity) and keeps the FIRST maximum, so tied candidates resolve as in FVCOM,
#2 read_dat accepts a count header on physical line 201 (SCAN_FILE tests NLINE > 200
   before reading) and refuses one on line 202,
#3 read_dat and read_dep strip an attached '!' comment before tokenising a record.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xfvcom.grid.dat_reader import read_dat
from xfvcom.grid.gtsz_builder import load_mesh
from xfvcom.io.dep_reader import read_dep
from xfvcom.io.obc_depth_control import (
    _fvcom_boundary_neighbours,
    next_obc_nodes,
    substituted_obc_depths,
)

_ELEM = "1 1 2 3\n"
_NODES = "1 400000.0 3900000.0\n2 400001.0 3900000.0\n3 400000.0 3900001.0\n"
_DEP = (
    "Node Number = 3\n"
    "400000.0 3900000.0 10.0\n"
    "400001.0 3900000.0 20.0\n"
    "400000.0 3900001.0 30.0\n"
)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    f = tmp_path / name
    f.write_text(text)
    return f


# ---------------------------------------------------------------------------
# #1 ties resolve in FVCOM's neighbour order
# ---------------------------------------------------------------------------
# The reviewer's five-node fan: at OBC node 1 the inward candidates 4 and 5 have
# identical dot products; FVCOM's NBSN(1,:) is 1,2,5,4,3,1 so node 5 (100 m) wins.
_X = np.array([0.0, -2.0, 2.0, 1.0, -1.0]) * 1000.0 + 400000.0
_Y = np.array([0.0, 0.0, 0.0, 1.0, 1.0]) * 1000.0 + 3900000.0
_NV = (
    np.array([[1, 3, 4], [1, 4, 5], [1, 5, 2]]).T - 1
)  # 1-based CCW -> (3, nele) 0-based
_OBC = np.array([2, 1, 3])
_H = np.array([20.0, 20.0, 20.0, 10.0, 100.0])


def test_round16_nbsn_order_matches_fvcom():
    assert [j + 1 for j in _fvcom_boundary_neighbours(0, _NV, [0, 1, 2], _X, _Y)] == [
        2,
        5,
        4,
        3,
    ]


@pytest.mark.parametrize("rule", ["observed", "source"])
def test_round16_tied_maximum_takes_fvcom_first(rule):
    h, nxt = substituted_obc_depths(_X, _Y, _NV, _OBC, _H, endpoint_rule=rule)
    assert nxt[1] == 5
    assert h[1] == 100.0


def test_round16_tie_independent_of_element_order():
    # the boundary walk starts at the unique boundary cell, so element order is irrelevant
    perm = _NV[:, [2, 0, 1]]
    assert next_obc_nodes(_X, _Y, perm, _OBC)[1] == 5


# ---------------------------------------------------------------------------
# #2 SCAN_FILE reads physical line 201
# ---------------------------------------------------------------------------
def _padded_mesh(tmp_path: Path, n_pad: int) -> Path:
    text = (
        "Cell Number = 1\n" + "! pad\n" * n_pad + "Node Number = 3\n" + _ELEM + _NODES
    )
    return _write(tmp_path, "m_grd.dat", text)


def test_round16_header_on_line_201_is_accepted(tmp_path):
    g = read_dat(_padded_mesh(tmp_path, 199))  # Node Number on line 201
    assert g["x"].shape == (3,)
    np.testing.assert_array_equal(g["nv"][:, 0], [0, 1, 2])


def test_round16_header_on_line_202_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"line 202: .*beyond line 201"):
        read_dat(_padded_mesh(tmp_path, 200))


# ---------------------------------------------------------------------------
# #3 attached '!' comments
# ---------------------------------------------------------------------------
def test_round16_dep_attached_comment(tmp_path):
    f = _write(tmp_path, "m_dep.dat", _DEP.replace("10.0\n", "10.0!survey\n"))
    d = read_dep(f)
    np.testing.assert_array_equal(d.h, [10.0, 20.0, 30.0])


def test_round16_load_mesh_attached_comments(tmp_path):
    grd = _write(
        tmp_path,
        "m_grd.dat",
        "Node Number = 3\nCell Number = 1\n"
        + "1 1 2 3!triangle\n"
        + _NODES.replace("3900001.0\n", "3900001.0!apex\n"),
    )
    dep = _write(tmp_path, "m_dep.dat", _DEP.replace("10.0\n", "10.0!survey\n"))
    m = load_mesh(grd, dep)
    np.testing.assert_array_equal(m.h, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(m.nv[0], [0, 1, 2])
    np.testing.assert_array_equal(m.y, [3900000.0, 3900000.0, 3900001.0])


def test_round16_attached_comment_keeps_count_checks(tmp_path):
    # a record that is only a comment is not a record: the count check still fires
    dep = _write(tmp_path, "m_dep.dat", _DEP.replace("400000.0 3900001.0 30.0", "!x"))
    with pytest.raises(ValueError, match=r"ended prematurely \(2/3\)"):
        read_dep(dep)
    grd = _write(
        tmp_path,
        "m_grd.dat",
        "Node Number = 3\nCell Number = 1\n"
        + "1 1 2 3!c\n"
        + _NODES
        + "4 1.0 1.0!extra\n",
    )
    with pytest.raises(
        ValueError, match="line 7: .*follows the 3 declared node records"
    ):
        read_dat(grd)


def test_round16_extra_element_with_attached_comment_is_refused(tmp_path):
    grd = _write(
        tmp_path,
        "m_grd.dat",
        "Node Number = 3\nCell Number = 1\n" + "1 1 2 3\n2 1 3 2!extra\n" + _NODES,
    )
    with pytest.raises(ValueError, match="line 4: .*declared connectivity records"):
        read_dat(grd)
