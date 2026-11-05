# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Regression tests for review round 20 (gpt-6.1-sol, scratch/reviews/fulldomain/r20_final.md).

#1 '!' is not a comment in a list-directed READ: an attached '!' inside a value that is still
   needed fails the READ (FVCOM keeps the previous value and counts the row); text after all
   required values is ignored (this reverses the round-16 acceptance of '10.0!survey'),
#2 a comma that begins a continuation line completes the separator left by the previous line's
   end ('1 3150' + ',1' reads 1, 3150, 1; '1,3150,' + ',1' leaves item 3 null),
#3 FVCOM's connectivity-to-coordinate transition (BACKSPACE, READ J, BACKSPACE J) is modelled for
   gfortran and Intel Fortran; a node-1 record it cannot replay is refused,
#4 failed / unset READs inside the mesh blocks count as rows, as in FVCOM (a comment-only line
   inside or after the blocks is refused; blank lines are still harmless),
#5 repetition counts are converted lazily and compared textually against MAX_REPEAT,
#6 unused fortran_real imports removed.

Every Fortran behaviour asserted here was checked with gfortran 8.5 and ifx 2025.1 on 2026-11-05
(replicas of the READ loops of mod_input.F READ_COLDSTART_GRID / _COORDS / _DEPTH).
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from xfvcom.fortran_numbers import MAX_REPEAT, list_fields, repeat_count
from xfvcom.fvcom_input import INT, REAL, list_read
from xfvcom.grid.dat_reader import read_dat
from xfvcom.io.dep_reader import read_dep

_ROOT = Path(__file__).resolve().parents[1]


def _runner():
    spec = importlib.util.spec_from_file_location(
        "run_jcope_obc_r20", _ROOT / "scripts" / "run_jcope_obc.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _write(tmp_path: Path, name: str, text: str) -> Path:
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    return f


def _lines(text: str) -> list[tuple[int, str]]:
    return list(enumerate(text.split("\n"), 1))


# ---------------------------------------------------------------------------
# #1 attached '!'
# ---------------------------------------------------------------------------
_DEP2 = "Node Number = 2\n400000.0 3900000.0 10.0\n400001.0 3900000.0 20.0{}\n"


def test_round20_attached_comment_fails_the_read():
    rd = list_read(_lines("400001.0 3900000.0 20.0!note"), 0, (REAL, REAL, REAL))
    assert rd.status == "error"
    assert rd.values == (400001.0, 3900000.0)
    assert "'!' is not a comment" in rd.message
    rd = list_read(_lines("400001.0 3900000.0 20.0 !note"), 0, (REAL, REAL, REAL))
    assert rd.status == "ok" and rd.values == (400001.0, 3900000.0, 20.0)
    # an integer record too (gfortran 5010 / ifx 59 on '3!triangle')
    assert list_read(_lines("1 1 2 3!c"), 0, (INT,) * 4).status == "error"
    assert list_read(_lines("1 1 2 3 !c"), 0, (INT,) * 4).status == "ok"


def test_round20_dep_attached_comment_is_refused(tmp_path):
    # the reviewer's reproduction: FVCOM reads depths [10, 10]
    f = _write(tmp_path, "d_dep.dat", _DEP2.format("!note"))
    with pytest.raises(
        ValueError, match=r"malformed depth record.*'!' is not a comment"
    ):
        R.parse_dep_dat(f)
    with pytest.raises(ValueError, match="prematurely"):
        read_dep(f)
    f = _write(tmp_path, "e_dep.dat", _DEP2.format(" !note"))
    assert R.parse_dep_dat(f).tolist() == [10.0, 20.0]
    np.testing.assert_array_equal(read_dep(f).h, [10.0, 20.0])


def test_round20_obc_attached_comment_is_refused(tmp_path):
    f = _write(tmp_path, "o.dat", "OBC Node Number = 2\n1 3150 1\n2 3142 1!x\n")
    with pytest.raises(ValueError, match="malformed OBC record"):
        R.parse_obc_dat(f)
    f = _write(tmp_path, "p.dat", "OBC Node Number = 2\n1 3150 1\n2 3142 1 !x\n")
    assert R.parse_obc_dat(f).tolist() == [3150, 3142]


# ---------------------------------------------------------------------------
# #2 separator state across lines
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, status, values",
    [
        ("1 3150\n,1", "ok", (1, 3150, 1)),
        ("1 3150\n\n,1", "ok", (1, 3150, 1)),
        ("1 3150\n  ,  1", "ok", (1, 3150, 1)),
        ("1\n,3150 1", "ok", (1, 3150, 1)),
        ("1,3150,\n,1", "unset", (1, 3150, None)),
        ("1 3150 , \n , 1", "unset", (1, 3150, None)),
        ("1 ,\n,3150 1", "unset", (1, None, 3150)),
        (",1 3150 1", "unset", (None, 1, 3150)),
    ],
)
def test_round20_continuation_comma(text, status, values):
    rd = list_read(_lines(text), 0, (INT, INT, INT))
    assert (rd.status, rd.values) == (status, values)


def test_round20_obc_continuation_comma(tmp_path):
    f = _write(tmp_path, "o.dat", "OBC Node Number = 1\n1 3150\n,1\n")
    assert R.parse_obc_dat(f).tolist() == [3150]
    f = _write(tmp_path, "p.dat", "OBC Node Number = 1\n1,3150,\n,1\n")
    with pytest.raises(ValueError, match="null value leaves an item unset"):
        R.parse_obc_dat(f)


# ---------------------------------------------------------------------------
# #3 / #4 mesh transition and READ counting
# ---------------------------------------------------------------------------
_HEAD = "Node Number = 4\nCell Number = 2\n1 1 2 3\n2 2 4 3\n"
_N = [
    "1 400000.0 3900000.0",
    "2 400001.0 3900000.0",
    "3 400000.0 3900001.0",
    "4 400001.0 3900001.0",
]
_X = [400000.0, 400001.0, 400000.0, 400001.0]


def _mesh(tmp_path: Path, nodes: list[str]) -> Path:
    return _write(tmp_path, "m_grd.dat", _HEAD + "\n".join(nodes) + "\n")


@pytest.mark.parametrize(
    "nodes",
    [
        _N,
        # integer-valued coordinates (the transition READ succeeds over two lines)
        [
            "1 400000 3900000",
            "2 400001 3900000",
            "3 400000 3900001",
            "4 400001 3900001",
        ],
        # blank lines before node 1, inside the block after node 2, and at the end
        ["", "", _N[0], _N[1], _N[2], "", _N[3], ""],
        # a later record split over two lines
        [_N[0], "2 400001.0", "3900000.0", _N[2], _N[3]],
        # node 1 with an extra column, then a blank line
        [_N[0] + " 0.0", "", _N[1], _N[2], _N[3]],
    ],
)
def test_round20_mesh_layouts_fvcom_reads(tmp_path, nodes):
    g = read_dat(_mesh(tmp_path, nodes))
    np.testing.assert_array_equal(g["x"], _X)


@pytest.mark.parametrize(
    "nodes, msg",
    [
        # the reviewer's split first-node record: gfortran's READ J fails on '400000.0'
        (
            ["1", "400000.0 3900000.0"] + _N[1:],
            r"under gfortran: its READ\(GRIDUNIT,\*\) J",
        ),
        # node 1 split after X: Intel Fortran BACKSPACEs onto the Y line
        (["1 400000.0", "3900000.0"] + _N[1:], "under Intel Fortran: it reads J = 2"),
        # a blank line between nodes 1 and 2: Intel Fortran lands on the blank, skipping node 1
        ([_N[0], "", *_N[1:]], "under Intel Fortran: it reads J = 2"),
    ],
)
def test_round20_split_first_node_is_refused(tmp_path, nodes, msg):
    with pytest.raises(ValueError, match=msg):
        read_dat(_mesh(tmp_path, nodes))


def test_round20_comment_inside_or_after_mesh_blocks_is_refused(tmp_path):
    # interior: FVCOM counts five coordinate rows
    with pytest.raises(ValueError, match="cannot be replayed|cannot be read"):
        read_dat(_mesh(tmp_path, [_N[0], _N[1], "! note", _N[2], _N[3]]))
    with pytest.raises(ValueError, match="node record 3 of 4 '! note' cannot be read"):
        read_dat(_mesh(tmp_path, [_N[0], _N[1], "! note", _N[2], _N[3]]))
    # trailing standalone comment: the fifth READ
    with pytest.raises(
        ValueError, match="'! note' follows the 4 declared node records"
    ):
        read_dat(_mesh(tmp_path, [*_N, "! note"]))
    # inside the connectivity block (a failed READ counts as a cell row)
    f = _write(
        tmp_path,
        "c_grd.dat",
        "Node Number = 4\nCell Number = 2\n1 1 2 3\n! note\n2 2 4 3\n"
        + "\n".join(_N)
        + "\n",
    )
    with pytest.raises(
        ValueError, match="connectivity record 2 of 2 '! note' cannot be read"
    ):
        read_dat(f)
    # an unset READ (null value) inside the coordinate block keeps the previous values in FVCOM
    with pytest.raises(ValueError, match="null value"):
        read_dat(_mesh(tmp_path, [_N[0], "2,,3900000.0", _N[2], _N[3]]))


# ---------------------------------------------------------------------------
# #5 lazy repetition counts
# ---------------------------------------------------------------------------
_HUGE = "9" * 5000


def test_round20_unused_huge_count_is_ignored(tmp_path):
    rd = list_read(_lines(f"1 3150 1 {_HUGE}*1"), 0, (INT, INT, INT))
    assert (rd.status, rd.values) == ("ok", (1, 3150, 1))
    f = _write(tmp_path, "o.dat", f"OBC Node Number = 1\n1 3150 1 {_HUGE}*1\n")
    assert R.parse_obc_dat(f).tolist() == [3150]


def test_round20_consumed_huge_count_fails_cleanly():
    rd = list_read(_lines(f"1 3150 {_HUGE}*1"), 0, (INT, INT, INT))
    assert rd.status == "error"
    assert f"exceeds {MAX_REPEAT}" in rd.message and "5000 digits" in rd.message
    with pytest.raises(ValueError, match=f"exceeds {MAX_REPEAT}"):
        list_fields(f"{_HUGE}*1")


def test_round20_repeat_count_textual_limit():
    assert repeat_count(str(MAX_REPEAT)) == MAX_REPEAT
    assert repeat_count("000" + str(MAX_REPEAT)) == MAX_REPEAT
    for bad in (str(MAX_REPEAT + 1), "1" + "0" * 9, _HUGE):
        with pytest.raises(ValueError, match="exceeds"):
            repeat_count(bad)


# ---------------------------------------------------------------------------
# #6 unused imports
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rel", ["scripts/run_jcope_obc.py", "xfvcom/io/dep_reader.py"])
def test_round20_no_unused_fortran_real_import(rel):
    tree = ast.parse((_ROOT / rel).read_text())
    imported = {
        a.asname or a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom)
        for a in n.names
    }
    assert "fortran_real" not in imported
