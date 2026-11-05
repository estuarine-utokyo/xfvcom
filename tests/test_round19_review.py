# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Regression tests for review round 19 (gpt-6.1-sol, scratch/reviews/fulldomain/r19_final.md).

The depth / OBC / mesh readers now model FVCOM's list-directed READs instead of physical lines
(xfvcom/fvcom_input.py list_read / read_all / locate_records):

#1 a free-text title before the count header is a failed READ, i.e. a header line (it was
   mistaken for the first data record in round 18),
#2 a repetition count above gfortran's limit (200000000) fails the READ; list_fields no longer
   substitutes a capped count silently,
#3 '/' ends a list-directed READ, attached or separated; an early '/' leaves items unset (refused),
#4 a record may continue onto further lines (READs, not lines, are counted),
#5 a blank line before a count header misaligns FVCOM's NHEADER rewind and is refused.

Fortran behaviour was checked with gfortran 8.5 and ifx 2025.1 on 2026-11-05.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from xfvcom.fortran_numbers import MAX_REPEAT, list_fields
from xfvcom.fvcom_input import INT, list_read
from xfvcom.grid.dat_reader import read_dat
from xfvcom.io.dep_reader import read_dep

_ROOT = Path(__file__).resolve().parents[1]


def _runner():
    spec = importlib.util.spec_from_file_location(
        "run_jcope_obc_r19", _ROOT / "scripts" / "run_jcope_obc.py"
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


_ELEM = "1 1 2 3\n"
_NODES = "1 400000.0 3900000.0\n2 400001.0 3900000.0\n3 400000.0 3900001.0\n"
_MESH = "Node Number = 3\nCell Number = 1\n" + _ELEM + _NODES
_XY = np.array([[400000.0, 3900000.0], [400001.0, 3900000.0], [400000.0, 3900001.0]])


# ---------------------------------------------------------------------------
# #1 free-text titles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "title",
    ["Tokyo Bay bathymetry", "Tokyo Bay grid = v3 (UTM 54)", "2020 survey", "! note"],
)
def test_round19_free_text_title_before_count_header(tmp_path, title):
    dep = _write(
        tmp_path, "d.dat", f"{title}\nNode Number = 1\n400000.0 3900000.0 10.0\n"
    )
    assert read_dep(dep).h.tolist() == [10.0]
    assert R.parse_dep_dat(dep).tolist() == [10.0]
    obc = _write(tmp_path, "o.dat", f"{title}\nOBC Node Number = 1\n1 3150 1\n")
    assert R.parse_obc_dat(obc).tolist() == [3150]
    m = read_dat(_write(tmp_path, "m_grd.dat", f"{title}\n" + _MESH))
    np.testing.assert_array_equal(m["x"], _XY[:, 0])


def test_round19_numeric_title_is_still_a_record(tmp_path):
    # a title that READs as X,Y,H IS the first record for FVCOM; the header then follows the block
    dep = _write(tmp_path, "d.dat", "1 2 3\nNode Number = 1\n400000.0 3900000.0 10.0\n")
    with pytest.raises(ValueError, match="follows the start of the depth records"):
        read_dep(dep)


def test_round19_broken_first_record_is_named(tmp_path):
    # FVCOM's header search passes over a broken first record; the message still names it
    dep = _write(
        tmp_path,
        "d.dat",
        "Node Number = 2\n400000.0 3900000.0 x\n400001.0 3900000.0 20.0\n",
    )
    with pytest.raises(
        ValueError, match=r"line 2: malformed depth record.*header search"
    ):
        R.parse_dep_dat(dep)


# ---------------------------------------------------------------------------
# #2 repetition counts
# ---------------------------------------------------------------------------
def test_round19_oversized_repeat_count_is_refused(tmp_path):
    obc = _write(
        tmp_path, "o.dat", "OBC Node Number = 1\n1 3150 999999999999999999999*1\n"
    )
    with pytest.raises(ValueError, match="malformed OBC record.*exceeds 200000000"):
        R.parse_obc_dat(obc)
    with pytest.raises(ValueError, match="exceeds 200000000"):
        list_fields("1 3150 999999999999999999999*1")


def test_round19_repeat_count_limit_is_gfortran_s():
    lines = [(1, f"{MAX_REPEAT}*1")]
    assert list_read(lines, 0, (INT, INT, INT)).values == (1, 1, 1)
    lines = [(1, f"{MAX_REPEAT + 1}*1")]
    assert list_read(lines, 0, (INT, INT, INT)).status == "error"
    # items after the ones the READ needs are never looked at (Fortran ignores them)
    lines = [(1, "1 3150 1 999999999999999999999*1")]
    assert list_read(lines, 0, (INT, INT, INT)).values == (1, 3150, 1)


def test_round19_large_valid_repeat_is_lazy(tmp_path):
    obc = _write(tmp_path, "o.dat", "OBC Node Number = 1\n1 100000000*7\n")
    assert R.parse_obc_dat(obc).tolist() == [7]


# ---------------------------------------------------------------------------
# #3 '/' terminators
# ---------------------------------------------------------------------------
def test_round19_slash_terminator(tmp_path):
    for rec in ("1 3150 1/", "1 3150 1 /", "1,3150,1/ trailing", "1 3150 1 / 9 9"):
        obc = _write(tmp_path, "o.dat", f"OBC Node Number = 1\n{rec}\n")
        assert R.parse_obc_dat(obc).tolist() == [3150], rec
    dep = _write(tmp_path, "d.dat", "Node Number = 1\n400000.0 3900000.0 10.0/\n")
    assert R.parse_dep_dat(dep).tolist() == [10.0]
    assert read_dep(dep).h.tolist() == [10.0]


def test_round19_early_slash_is_refused(tmp_path):
    obc = _write(tmp_path, "o.dat", "OBC Node Number = 2\n1 3150/\n2 3142 1\n")
    with pytest.raises(
        ValueError, match=r"line 2: malformed OBC record.*'/' ends the READ"
    ):
        R.parse_obc_dat(obc)
    dep = _write(tmp_path, "d.dat", "Node Number = 1\n400000.0 / 3900000.0 10.0\n")
    with pytest.raises(ValueError, match="malformed depth record"):
        R.parse_dep_dat(dep)
    with pytest.raises(ValueError, match="prematurely"):
        read_dep(dep)


# ---------------------------------------------------------------------------
# #4 records spanning lines
# ---------------------------------------------------------------------------
def test_round19_obc_records_spanning_lines(tmp_path):
    obc = _write(tmp_path, "o.dat", "OBC Node Number = 2\n1 3150\n1\n2 3142\n1\n")
    assert R.parse_obc_dat(obc).tolist() == [3150, 3142]
    # blank lines inside a continued record are passed over, as by the Fortran runtime
    obc = _write(tmp_path, "p.dat", "OBC Node Number = 2\n1\n\n3150 1\n2 3142 1\n")
    assert R.parse_obc_dat(obc).tolist() == [3150, 3142]
    # unused trailing fields of the line that completes a record are ignored
    obc = _write(tmp_path, "q.dat", "OBC Node Number = 1\n1 3150\n1 99 junk\n")
    assert R.parse_obc_dat(obc).tolist() == [3150]


def test_round19_depth_records_spanning_lines(tmp_path):
    dep = _write(
        tmp_path,
        "d.dat",
        "Node Number = 3\n400000.0\n3900000.0 10.0\n400001.0 3900000.0\n20.0\n"
        "400000.0 3900001.0 30.0\n",
    )
    assert R.parse_dep_dat(dep, _XY).tolist() == [10.0, 20.0, 30.0]
    assert read_dep(dep).h.tolist() == [10.0, 20.0, 30.0]


def test_round19_mesh_records_spanning_lines(tmp_path):
    # jsasaki 2026-11-05: review round 20 #3: the split moved from node 1 to node 2 -- FVCOM's
    # connectivity-to-coordinate transition cannot replay a split node-1 record (refused now)
    text = (
        "Node Number = 3\nCell Number = 1\n1 1\n2 3\n"
        "1 400000.0 3900000.0\n2 400001.0\n3900000.0\n3 400000.0 3900001.0\n"
    )
    m = read_dat(_write(tmp_path, "m_grd.dat", text))
    np.testing.assert_array_equal(m["nv"][:, 0], [0, 1, 2])
    np.testing.assert_array_equal(m["x"], _XY[:, 0])
    np.testing.assert_array_equal(m["y"], _XY[:, 1])


def test_round19_continued_record_with_comment_and_eof(tmp_path):
    # a '!' reached while values are still needed is read as a value and fails
    obc = _write(tmp_path, "o.dat", "OBC Node Number = 2\n1 3150 ! x\n1\n2 3142 1\n")
    # jsasaki 2026-11-05: review round 20 #1: the message says '!' is not a comment there
    with pytest.raises(ValueError, match="malformed OBC record.*'!' is not a comment"):
        R.parse_obc_dat(obc)
    # an incomplete last record
    obc = _write(tmp_path, "p.dat", "OBC Node Number = 2\n1 3150 1\n2 3142\n")
    with pytest.raises(ValueError, match="incomplete OBC record"):
        R.parse_obc_dat(obc)
    # READs, not lines, are counted against the header
    obc = _write(tmp_path, "q.dat", "OBC Node Number = 3\n1 3150\n1\n2 3142\n1\n")
    with pytest.raises(ValueError, match="declares 3 record"):
        R.parse_obc_dat(obc)


# ---------------------------------------------------------------------------
# #5 blank lines and FVCOM's NHEADER rewind
# ---------------------------------------------------------------------------
def test_round19_leading_blank_line_is_refused(tmp_path):
    obc = _write(tmp_path, "o.dat", "\nOBC Node Number = 2\n1 3150 1\n2 3142 1\n")
    with pytest.raises(ValueError, match=r"blank line 1 .*NHEADER = 1.*header line 2"):
        R.parse_obc_dat(obc)
    dep = _write(tmp_path, "d.dat", "\nNode Number = 1\n400000.0 3900000.0 10.0\n")
    with pytest.raises(ValueError, match="blank line 1"):
        R.parse_dep_dat(dep)
    with pytest.raises(ValueError, match="blank line 1"):
        read_dep(dep)
    mesh = _write(
        tmp_path, "m_grd.dat", "Node Number = 3\n\nCell Number = 1\n" + _ELEM + _NODES
    )
    with pytest.raises(ValueError, match="blank line 2"):
        read_dat(mesh)


def test_round19_harmless_blank_lines_are_accepted(tmp_path):
    # after the last header line and between records the rewind is unaffected
    obc = _write(tmp_path, "o.dat", "OBC Node Number = 2\n\n1 3150 1\n\n2 3142 1\n\n")
    assert R.parse_obc_dat(obc).tolist() == [3150, 3142]
    dep = _write(tmp_path, "d.dat", "Node Number = 1\n\n\n400000.0 3900000.0 10.0\n\n")
    assert R.parse_dep_dat(dep).tolist() == [10.0]
    assert read_dep(dep).h.tolist() == [10.0]
    mesh = _write(
        tmp_path,
        "m_grd.dat",
        "Node Number = 3\nCell Number = 1\n\n" + _ELEM + "\n" + _NODES,
    )
    np.testing.assert_array_equal(read_dat(mesh)["x"], _XY[:, 0])


def test_round19_production_obc_with_leading_blank(tmp_path):
    tb = Path.home() / "Github" / "TB-FVCOM"
    cands = [
        tb / "input/goto2023/grid/TokyoBay_obc.dat",
        tb / "ersem/input/TokyoBay_obc.dat",
    ]
    src = next((c for c in cands if c.is_file()), None)
    if src is None:
        pytest.skip("TB-FVCOM production OBC file not available")
    nodes = R.parse_obc_dat(src)
    assert nodes.size == 13
    f = _write(tmp_path, "o.dat", "\n" + src.read_text())
    with pytest.raises(ValueError, match="blank line 1"):
        R.parse_obc_dat(f)
