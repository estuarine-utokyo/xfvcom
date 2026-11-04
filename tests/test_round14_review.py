# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Regression tests for review round 14 (gpt-6.1-sol, scratch/reviews/fulldomain/r14_final.md).

#2 mesh coordinates (read_dat) and depth records (read_dep, run_jcope_obc.parse_dep_dat)
   accept Fortran real literals (D/E/Q exponents, implicit exponents) as FVCOM's
   list-directed reads do; ids stay integers and the finiteness / count / order /
   mesh-identity checks stay,
#3 read_dat matches the count headers after removing leading SPACES (FVCOM's
   ADJUSTL); a tab is still not accepted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from xfvcom.fortran_numbers import fortran_real
from xfvcom.grid.dat_reader import read_dat
from xfvcom.io import sigma_dat
from xfvcom.io.dep_reader import read_dep

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_jcope_obc as r  # noqa: E402  # isort: skip

_ELEM = "1 1 2 3 1\n"


def _mesh(tmp_path: Path, nodes: str, head: str = "Node Number = 3\nCell Number = 1\n"):
    f = tmp_path / "m_grd.dat"
    f.write_text(head + _ELEM + nodes)
    return f


# ---------------------------------------------------------------------------
# #2 Fortran real literals in mesh and depth records
# ---------------------------------------------------------------------------
def test_round14_fortran_real_is_shared():
    # one conversion, re-exported from its old home
    assert sigma_dat.fortran_real is fortran_real


def test_round14_mesh_d_exponents(tmp_path):
    f = _mesh(tmp_path, "1 4.0D+5 3.9D+6\n2 4.01D+5 3.9D+6\n3 4.0D+5 3.901D+6\n")
    d = read_dat(f)
    np.testing.assert_array_equal(d["x"], [4.0e5, 4.01e5, 4.0e5])
    np.testing.assert_array_equal(d["y"], [3.9e6, 3.9e6, 3.901e6])
    assert d["nv"][:, 0].tolist() == [0, 1, 2]


def test_round14_mesh_implicit_exponents(tmp_path):
    f = _mesh(tmp_path, "1 4.0+5 3.9q6\n2 4.01e5 3.9+6\n3 .4+6 3.901d6\n")
    d = read_dat(f)
    np.testing.assert_array_equal(d["x"], [4.0e5, 4.01e5, 4.0e5])
    np.testing.assert_array_equal(d["y"], [3.9e6, 3.9e6, 3.901e6])


@pytest.mark.parametrize("bad", ["nan", "inf", "1.0D+999", "4.0.0", "1e"])
def test_round14_mesh_rejects_non_literals(tmp_path, bad):
    # not a finite Fortran real: the record is not a node record, the block is short
    f = _mesh(tmp_path, f"1 4.0D+5 3.9D+6\n2 {bad} 3.9D+6\n3 4.0D+5 3.901D+6\n")
    with pytest.raises(ValueError):
        read_dat(f)


def test_round14_mesh_ids_stay_integer_and_ordered(tmp_path):
    # a real-literal id is not an id record: the next record is taken as row 1
    f = _mesh(tmp_path, "1.0D0 4.0D+5 3.9D+6\n2 4.01D+5 3.9D+6\n3 4.0D+5 3.901D+6\n")
    with pytest.raises(ValueError, match="node record 1 carries id 2"):
        read_dat(f)
    f = _mesh(tmp_path, "2 4.0D+5 3.9D+6\n1 4.01D+5 3.9D+6\n3 4.0D+5 3.901D+6\n")
    with pytest.raises(ValueError, match="row position"):
        read_dat(f)


def test_round14_read_dep_fortran_literals(tmp_path):
    f = tmp_path / "t_dep.dat"
    f.write_text("Node Number = 2\n400000.0 3900000.0 1.0D+1\n4.01D+5 3.9+6 2.5-1\n")
    d = read_dep(f)
    np.testing.assert_array_equal(d.x, [4.0e5, 4.01e5])
    np.testing.assert_array_equal(d.y, [3.9e6, 3.9e6])
    np.testing.assert_array_equal(d.h, [10.0, 0.25])
    f.write_text("Node Number = 2\n400000.0 3900000.0 nan\n4.01D+5 3.9+6 2.5-1\n")
    with pytest.raises(ValueError, match="prematurely"):
        read_dep(f)


def test_round14_parse_dep_dat_fortran_literals(tmp_path):
    xy = np.array([[4.0e5, 3.9e6], [4.01e5, 3.9e6]])
    f = tmp_path / "d.dat"
    f.write_text("Node Number = 2\n400000.0 3900000.0 1.0D+1\n4.01D+5 3.9+6 2.5-1\n")
    assert r.parse_dep_dat(f, xy).tolist() == [10.0, 0.25]
    # mesh identity is still checked on the converted coordinates
    with pytest.raises(ValueError, match="does not belong"):
        r.parse_dep_dat(f, xy[::-1].copy())
    for bad in ("inf", "1.0D+999", "1.0.0", "x"):
        f.write_text(
            f"Node Number = 2\n400000.0 3900000.0 {bad}\n4.01D+5 3.9+6 2.5-1\n"
        )
        with pytest.raises(ValueError, match="malformed depth record"):
            r.parse_dep_dat(f, xy)


# ---------------------------------------------------------------------------
# #3 indented count headers
# ---------------------------------------------------------------------------
_NODES = "1 0.0 0.0\n2 1.0 0.0\n3 0.0 1.0\n"


def test_round14_indented_headers(tmp_path):
    plain = read_dat(_mesh(tmp_path, _NODES))
    ind = read_dat(
        _mesh(tmp_path, _NODES, head="  Node Number = 3\n  Cell Number = 1\n")
    )
    for k in ("x", "y", "nv"):
        np.testing.assert_array_equal(plain[k], ind[k])


def test_round14_tab_indented_header_still_refused(tmp_path):
    f = _mesh(tmp_path, _NODES, head="\tNode Number = 3\n\tCell Number = 1\n")
    with pytest.raises(ValueError, match="counts not found"):
        read_dat(f)


def test_round14_indented_headers_keep_validation(tmp_path):
    # the count is still enforced after an indented header
    f = _mesh(
        tmp_path,
        "1 0.0 0.0\n2 1.0 0.0\n",
        head="  Node Number = 3\n  Cell Number = 1\n",
    )
    with pytest.raises(ValueError, match="prematurely"):
        read_dat(f)
