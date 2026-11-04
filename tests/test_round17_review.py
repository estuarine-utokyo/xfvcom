# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Regression tests for review round 17 (gpt-6.1-sol, scratch/reviews/fulldomain/r17_final.md).

#1 next_obc_nodes gives the same NEXT_OBC whether the grid stores file-order (counter-clockwise)
   or native FVCOM (reversed, clockwise) connectivity,
#2 read_dat recognises only the complete, case-sensitive count-header keys 'Node Number' and
   'Cell Number' (SCAN_FILE: TRIM(VARNAME) == TRIM(VNAME)),
#3 read_dat refuses a count header longer than FVCOM's 400-character INPLINE buffer,
#4 commas separate record fields like blanks (list-directed READ) in read_dat, read_dep and the
   runner's depth and OBC parsers.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from xfvcom.fortran_numbers import list_fields
from xfvcom.grid.dat_reader import read_dat
from xfvcom.grid.grid_obj import FvcomGrid
from xfvcom.io.dep_reader import read_dep
from xfvcom.io.obc_depth_control import next_obc_nodes, substituted_obc_depths

_ROOT = Path(__file__).resolve().parents[1]


def _runner():
    spec = importlib.util.spec_from_file_location(
        "run_jcope_obc_r17", _ROOT / "scripts" / "run_jcope_obc.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write(tmp_path: Path, name: str, text: str) -> Path:
    f = tmp_path / name
    f.write_text(text)
    return f


# ---------------------------------------------------------------------------
# #1 connectivity winding
# ---------------------------------------------------------------------------
# The round-16 five-node fan: file records are counter-clockwise; FVCOM selects node 5 / 100 m.
_X = np.array([0.0, -2.0, 2.0, 1.0, -1.0]) * 1000.0 + 400000.0
_Y = np.array([0.0, 0.0, 0.0, 1.0, 1.0]) * 1000.0 + 3900000.0
_NV = np.array([[1, 3, 4], [1, 4, 5], [1, 5, 2]]).T - 1
_OBC = np.array([2, 1, 3])
_H = np.array([20.0, 20.0, 20.0, 10.0, 100.0])


def _native_grid() -> FvcomGrid:
    # FVCOM output: NV = (N1, N3, N2), one-based, dims ("three", "nele")
    native_nv = _NV[[0, 2, 1], :] + 1
    ds = xr.Dataset(
        {
            "x": ("node", _X),
            "y": ("node", _Y),
            "nv": (("three", "nele"), native_nv),
        }
    )
    return FvcomGrid.from_dataset(ds)


def _dat_grid(tmp_path: Path) -> FvcomGrid:
    elems = "".join(
        f"{k + 1} {a + 1} {b + 1} {c + 1}\n" for k, (a, b, c) in enumerate(_NV.T)
    )
    nodes = "".join(f"{k + 1} {x} {y}\n" for k, (x, y) in enumerate(zip(_X, _Y)))
    f = _write(
        tmp_path, "fan_grd.dat", "Node Number = 5\nCell Number = 3\n" + elems + nodes
    )
    return FvcomGrid.from_dat(f, utm_zone=54)


@pytest.mark.parametrize("rule", ["observed", "source"])
def test_round17_native_and_file_winding_agree(tmp_path, rule):
    native = _native_grid()
    dat = _dat_grid(tmp_path)
    # the two grids really hold opposite windings
    np.testing.assert_array_equal(native.nv, dat.nv[[0, 2, 1], :])
    for g in (native, dat):
        h, nxt = substituted_obc_depths(g.x, g.y, g.nv, _OBC, _H, endpoint_rule=rule)
        assert nxt[1] == 5
        assert h[1] == 100.0
    np.testing.assert_array_equal(
        next_obc_nodes(native.x, native.y, native.nv, _OBC, endpoint_rule=rule),
        next_obc_nodes(dat.x, dat.y, dat.nv, _OBC, endpoint_rule=rule),
    )


def test_round17_mixed_winding_per_cell():
    # the winding is decided per cell, so a mixture still gives FVCOM's choice
    mixed = _NV.copy()
    mixed[:, 1] = mixed[[0, 2, 1], 1]
    assert next_obc_nodes(_X, _Y, mixed, _OBC)[1] == 5


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_round17_degenerate_cell_is_refused():
    y = _Y.copy()
    y[3] = y[
        0
    ]  # node 4 on the line through nodes 1 and 3: cell (1, 3, 4) has zero area
    with pytest.raises(ValueError, match="zero or non-finite area"):
        next_obc_nodes(_X, y, _NV, _OBC)


# ---------------------------------------------------------------------------
# #2 complete, case-sensitive header keys
# ---------------------------------------------------------------------------
_ELEM = "1 1 2 3\n"
_NODES = "1 400000.0 3900000.0\n2 400001.0 3900000.0\n3 400000.0 3900001.0\n"
_MESH = "Node Number = 3\nCell Number = 1\n" + _ELEM + _NODES


@pytest.mark.parametrize(
    "old,new",
    [
        ("Node Number", "node number"),
        ("Node Number", "Node Number unused"),
        ("Node Number", "Node Numbers"),
        ("Cell Number", "Elem Number"),
        ("Cell Number", "CELL NUMBER"),
        ("Node Number =", "Node Number ! ="),
    ],
)
def test_round17_inexact_header_keys_are_refused(tmp_path, old, new):
    f = _write(tmp_path, "m_grd.dat", _MESH.replace(old, new, 1))
    with pytest.raises(ValueError, match="counts not found"):
        read_dat(f)


@pytest.mark.parametrize(
    "line",
    [
        "Node Number = 3",
        "   Node Number=3",
        "Node Number      =   3   ! comment",
        "Node Number = 3,",
        "Node Number" + " " * 29 + "= 3",  # key padded to VARNAME's 40 characters
    ],
)
def test_round17_exact_header_keys_are_accepted(tmp_path, line):
    f = _write(tmp_path, "m_grd.dat", _MESH.replace("Node Number = 3", line, 1))
    assert read_dat(f)["x"].shape == (3,)


def test_round17_unrecognised_key_is_not_a_header(tmp_path):
    # FVCOM skips a line whose key is not 'Node Number'; the real header that follows is used
    f = _write(tmp_path, "m_grd.dat", "node number = 99\n" + _MESH)
    assert read_dat(f)["x"].shape == (3,)


# ---------------------------------------------------------------------------
# #3 FVCOM's 400-character header buffer
# ---------------------------------------------------------------------------
def _header_of_length(n: int) -> str:
    base = "Node Number"
    tail = "= 3"
    return base + " " * (n - len(base) - len(tail)) + tail


def test_round17_header_of_400_characters_is_accepted(tmp_path):
    line = _header_of_length(400)
    assert len(line) == 400
    f = _write(tmp_path, "m_grd.dat", _MESH.replace("Node Number = 3", line, 1))
    assert read_dat(f)["x"].shape == (3,)


def test_round17_header_of_401_characters_is_refused(tmp_path):
    line = _header_of_length(401)
    assert len(line) == 401
    f = _write(tmp_path, "m_grd.dat", _MESH.replace("Node Number = 3", line, 1))
    with pytest.raises(ValueError, match="line 1: .*first 400 characters"):
        read_dat(f)


def test_round17_reviewer_400_spaces_is_refused(tmp_path):
    line = "Node Number" + " " * 400 + "= 3"
    f = _write(tmp_path, "m_grd.dat", _MESH.replace("Node Number = 3", line, 1))
    with pytest.raises(ValueError, match="first 400 characters"):
        read_dat(f)


def test_round17_long_value_split_at_400_is_refused(tmp_path):
    # FVCOM would read '3' of '32...' only; the line is refused rather than read as either value
    line = "Node Number =" + " " * (399 - len("Node Number =")) + "32"
    assert len(line) == 401
    f = _write(tmp_path, "m_grd.dat", _MESH.replace("Node Number = 3", line, 1))
    with pytest.raises(ValueError, match="first 400 characters"):
        read_dat(f)


def test_round17_long_non_header_line_is_ignored(tmp_path):
    f = _write(tmp_path, "m_grd.dat", "! " + "x" * 500 + "\n" + _MESH)
    assert read_dat(f)["x"].shape == (3,)


# ---------------------------------------------------------------------------
# #4 comma separators
# ---------------------------------------------------------------------------
def test_round17_list_fields():
    assert list_fields("1, 2 ,3") == ["1", "2", "3"]
    assert list_fields("1,2,3,") == ["1", "2", "3"]
    assert list_fields("1,,2") == ["1", "", "2"]  # null value
    assert list_fields(",1") == ["", "1"]
    assert list_fields("  ") == []


def _comma(text: str) -> str:
    return "\n".join(", ".join(ln.split()) for ln in text.splitlines()) + "\n"


def test_round17_dat_commas(tmp_path):
    ref = read_dat(_write(tmp_path, "a_grd.dat", _MESH))
    text = "Node Number = 3\nCell Number = 1\n" + _comma(_ELEM + _NODES)
    got = read_dat(_write(tmp_path, "b_grd.dat", text))
    for k in ("x", "y", "nv"):
        np.testing.assert_array_equal(got[k], ref[k])


def test_round17_dat_commas_keep_checks(tmp_path):
    text = (
        "Node Number = 3\nCell Number = 1\n" + _comma(_ELEM + _NODES) + "4, 1.0, 1.0\n"
    )
    with pytest.raises(ValueError, match="follows the 3 declared node records"):
        read_dat(_write(tmp_path, "m_grd.dat", text))
    # a null value is not a coordinate: the block ends prematurely
    text = (
        "Node Number = 3\nCell Number = 1\n"
        + _ELEM
        + _NODES.replace("2 400001.0 3900000.0", "2,,3900000.0")
    )
    with pytest.raises(ValueError):
        read_dat(_write(tmp_path, "n_grd.dat", text))


_DEP = (
    "Node Number = 3\n"
    "400000.0 3900000.0 10.0\n"
    "400001.0 3900000.0 20.0\n"
    "400000.0 3900001.0 30.0\n"
)


def test_round17_dep_commas(tmp_path):
    text = "Node Number = 3\n" + _comma("".join(_DEP.splitlines(True)[1:]))
    d = read_dep(_write(tmp_path, "m_dep.dat", text))
    np.testing.assert_array_equal(d.h, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(d.x, [400000.0, 400001.0, 400000.0])


def test_round17_runner_depth_and_obc_commas(tmp_path):
    run = _runner()
    dep = _write(
        tmp_path,
        "m_dep.dat",
        "Node Number = 3\n"
        "400000.0, 3900000.0, 10.0\n"
        "400001.0 ,3900000.0,20.0 ! c\n"
        "400000.0 3900001.0 30.0\n",
    )
    xy = np.array([[400000.0, 3900000.0], [400001.0, 3900000.0], [400000.0, 3900001.0]])
    np.testing.assert_array_equal(run.parse_dep_dat(dep, xy), [10.0, 20.0, 30.0])
    # correspondence check still applies
    with pytest.raises(ValueError, match="does not belong to this mesh"):
        run.parse_dep_dat(dep, xy[::-1].copy())
    obc = _write(tmp_path, "m_obc.dat", "OBC Node Number = 2\n1, 3, 1\n2,  1 , 1 ! c\n")
    np.testing.assert_array_equal(run.parse_obc_dat(obc), [3, 1])
    bad = _write(tmp_path, "b_obc.dat", "OBC Node Number = 1\n1,,1\n")
    with pytest.raises(ValueError, match="malformed OBC record"):
        run.parse_obc_dat(bad)
    bad_dep = _write(tmp_path, "b_dep.dat", "Node Number = 1\n400000.0,,10.0\n")
    with pytest.raises(ValueError, match="malformed depth record"):
        run.parse_dep_dat(bad_dep)
