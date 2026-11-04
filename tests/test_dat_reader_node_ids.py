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
