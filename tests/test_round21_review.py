# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Regression tests for review round 21 (gpt-6.1-sol, scratch/reviews/fulldomain/r21_final.md).

#1 list-directed values are separated by spaces, tabs and commas only (not by Python's Unicode
   whitespace): U+00A0, a form feed, a vertical tab, ';' or a carriage return inside a line stay in
   their token, which then fails to convert; physical lines are split at '\\n' only (one CRLF
   '\\r' removed), never by str.splitlines / universal newlines,
#2 a non-null repetition that crosses to an item of the other type fails (gfortran IOSTAT 5010);
   null repetitions stay type independent,
#3 FVCOM leaves the connectivity block only when J == 1 .AND. LM1 /= 1, so it cannot read a
   one-element mesh: the strict reader refuses it, allow_unordered=True reads it,
#4 strict reading enforces FVCOM's NODECOUNT check (largest referenced node id == Node Number).

Fortran behaviour checked on 2026-11-05 with gfortran 8.5 and ifx 2025.1 (a list-directed READ of
three INTEGERs over lines holding each separator candidate; READs of INTEGER/REAL lists from
repetitions; a replica of mod_input.F READ_COLDSTART_GRID / _COORDS on the meshes below):

* separators: space and tab (both compilers); '\\r' inside a line (gfortran: blank, ifx: IOSTAT
  59) and ';' (gfortran: separator, ifx: 59) are not portable; U+00A0, '\\f', '\\v', '\\x1c',
  NUL fail on both (5010 / 59); CRLF line endings read on both;
* '3*1' into INTEGER, REAL, REAL and '2*1.5' into REAL, INTEGER: gfortran 5010, ifx converts;
  '3*' and '2* 5.0' read on both;
* a three-node one-element mesh: both stop with 'rows of CONNECTIVITY exceed the stated number of
  Cells'; four nodes / two elements declared as five nodes: both stop with 'nodes in the grid file
  CONNECTIVITY does not equal the stated number' (NODECOUNT 4, MGL 5).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from xfvcom.fortran_numbers import list_items
from xfvcom.fvcom_input import INT, REAL, is_blank, list_read, physical_lines
from xfvcom.grid.dat_reader import read_dat
from xfvcom.io.dep_reader import read_dep

_ROOT = Path(__file__).resolve().parents[1]


def _runner():
    spec = importlib.util.spec_from_file_location(
        "run_jcope_obc_r21", _ROOT / "scripts" / "run_jcope_obc.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _write(tmp_path: Path, name: str, text: str) -> Path:
    f = tmp_path / name
    f.write_bytes(text.encode("utf-8"))
    return f


def _lines(text: str) -> list[tuple[int, str]]:
    return list(enumerate(text.split("\n"), 1))


_INT3 = (INT, INT, INT)


# ---------------------------------------------------------------------------
# #1 portable separators and physical lines
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sep", [" ", "\t", "  \t ", ",", " , ", "\t,\t"])
def test_round21_portable_separators(sep):
    rd = list_read(_lines(f"1{sep}2{sep}3"), 0, _INT3)
    assert rd.status == "ok" and rd.values == (1, 2, 3)
    # leading / trailing tabs are blanks too
    rd = list_read(_lines("\t1 2 3\t"), 0, _INT3)
    assert rd.status == "ok" and rd.values == (1, 2, 3)


@pytest.mark.parametrize(
    "line",
    [
        "1\u00a02 3",  # U+00A0 inside: one token '1\xa02'
        "\u00a01 2 3",  # leading U+00A0 is not trimmed
        "1 2 3\u00a0",  # trailing U+00A0 joins the third value (both compilers fail item 3)
        "1\x0c2 3",  # form feed
        "1\x0b2 3",  # vertical tab
        "1\x1c2 3",  # FS (str.split / splitlines treat it as whitespace / a line break)
        "1;2;3",  # gfortran separator, ifx error: not portable
        "1\r2 3",  # CR inside a line: gfortran blank, ifx error
        "1\u20282 3",  # U+2028 LINE SEPARATOR
        "1\x852 3",  # U+0085 NEXT LINE
    ],
)
def test_round21_other_whitespace_stays_in_the_token(line):
    rd = list_read(_lines(line), 0, _INT3)
    assert rd.status == "error"


def test_round21_nbsp_before_attached_comment_fails_like_fortran(tmp_path):
    # the reviewer's reproduction: FVCOM keeps the previous depth (10) for the second row; the
    # strict parser refuses the row instead of reading 20
    text = (
        "Node Number = 2\n"
        "400000.0 3900000.0 10.0\n"
        "400001.0 3900000.0 20.0\u00a0!note\n"
    )
    f = _write(tmp_path, "d.dat", text)
    with pytest.raises(ValueError, match="malformed depth record"):
        R.parse_dep_dat(f)
    # the trailing value text is one token for list_read too
    rd = list_read(_lines("400001.0 3900000.0 20.0\u00a0!note"), 0, (REAL, REAL, REAL))
    assert rd.status == "error" and rd.values == (400001.0, 3900000.0)
    # with a plain blank before the comment the value reads
    ok = _write(tmp_path, "o.dat", text.replace("\u00a0", " "))
    assert R.parse_dep_dat(ok).tolist() == [10.0, 20.0]


def test_round21_list_items_do_not_use_unicode_whitespace():
    assert list(list_items("1\u00a02 3")) == [("1", "1\u00a02"), ("1", "3")]
    assert list(list_items("\u3000")) == [("1", "\u3000")]
    assert list(list_items(" \t ")) == []
    assert is_blank(" \t") and is_blank("")
    assert not is_blank("\u00a0") and not is_blank("\x0c") and not is_blank("\r")


def test_round21_physical_lines_split_at_newline_only(tmp_path):
    f = tmp_path / "p.dat"
    f.write_bytes(b"a\r\nb\rc\x0cd\nlast")
    assert physical_lines(f) == [(1, "a"), (2, "b\rc\x0cd"), (3, "last")]
    f.write_bytes("x\u2028y\x85z\n\n".encode("utf-8"))
    assert physical_lines(f) == [(1, "x\u2028y\x85z"), (2, "")]
    f.write_bytes(b"")
    assert physical_lines(f) == []


_MESH4 = (
    "Node Number = 4\nCell Number = 2\n1 1 2 3\n2 2 4 3\n"
    "1 400000.0 3900000.0\n2 400001.0 3900000.0\n"
    "3 400000.0 3900001.0\n4 400001.0 3900001.0\n"
)
_X4 = [400000.0, 400001.0, 400000.0, 400001.0]


def test_round21_crlf_files_still_read(tmp_path):
    f = tmp_path / "m_grd.dat"
    f.write_bytes(_MESH4.replace("\n", "\r\n").encode())
    np.testing.assert_array_equal(read_dat(f)["x"], _X4)
    d = tmp_path / "d.dat"
    d.write_bytes(b"Node Number = 2\r\n400000.0 3900000.0 10.0\r\n1.0 2.0 20.0\r\n")
    assert read_dep(d).h.tolist() == [10.0, 20.0]
    assert R.parse_dep_dat(d).tolist() == [10.0, 20.0]


@pytest.mark.parametrize("ch", ["\x0c", "\u2028", "\x85", "\x1c", "\r"])
def test_round21_mesh_line_break_lookalikes_are_refused(tmp_path, ch):
    # str.splitlines / universal newlines used to break the node-2 record here, so the mesh read
    # as if the character were a newline; Fortran keeps it inside the record
    text = _MESH4.replace("2 400001.0 3900000.0\n", f"2 400001.0{ch}3900000.0\n")
    with pytest.raises(ValueError):
        read_dat(_write(tmp_path, "m_grd.dat", text))


def test_round21_nbsp_only_line_is_not_blank(tmp_path):
    # a line holding U+00A0 only is a failed READ (a counted row) in the node block, not a blank
    text = _MESH4.replace("3 400000.0 3900001.0\n", "\u00a0\n3 400000.0 3900001.0\n")
    with pytest.raises(ValueError, match="node record 3 of 4"):
        read_dat(_write(tmp_path, "m_grd.dat", text))
    # a tab-only line is blank and harmless
    text = _MESH4.replace("3 400000.0 3900001.0\n", "\t\n3 400000.0 3900001.0\n")
    np.testing.assert_array_equal(
        read_dat(_write(tmp_path, "t_grd.dat", text))["x"], _X4
    )


# ---------------------------------------------------------------------------
# #2 repetitions across item types
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, types",
    [
        ("3*1", (INT, REAL, REAL)),  # the reviewer's reproduction
        ("2*1 3900000.0", (INT, REAL, REAL)),
        ("1 2*7", (INT, INT, REAL)),
        ("2*1.5", (REAL, INT)),
        ("2*1", (REAL, INT)),  # REAL -> INTEGER fails too, even for an integer literal
    ],
)
def test_round21_repetition_across_types_fails(text, types):
    rd = list_read(_lines(text), 0, types)
    assert rd.status == "error"
    assert "repeated value across items of different types" in rd.message


@pytest.mark.parametrize(
    "text, types, status, values",
    [
        ("1 2*5.0", (INT, REAL, REAL), "ok", (1, 5.0, 5.0)),
        ("1*1 2*2", (INT, REAL, REAL), "ok", (1, 2.0, 2.0)),
        ("3*", (INT, REAL, REAL), "unset", (None, None, None)),
        ("2* 5.0", (INT, REAL, REAL), "unset", (None, None, 5.0)),
        ("1 2*", (INT, REAL, REAL), "unset", (1, None, None)),
        ("2*3 4.0", (INT, INT, REAL), "ok", (3, 3, 4.0)),
        # the repetition is cut where the READ's list ends; the rest of the line is ignored
        ("1 5*2.0", (INT, REAL, REAL), "ok", (1, 2.0, 2.0)),
    ],
)
def test_round21_repetition_within_one_type_or_null(text, types, status, values):
    rd = list_read(_lines(text), 0, types)
    assert rd.status == status and rd.values == values


def test_round21_intel_model_converts_across_types():
    rd = list_read(_lines("3*1"), 0, (INT, REAL, REAL), int_reals=True)
    assert rd.status == "ok" and rd.values == (1, 1.0, 1.0)


def test_round21_mesh_repetition_across_types_is_refused(tmp_path):
    # the reviewer's reproduction: node 1 as '2*1 3900000.0' used to read as (1, 3900000)
    text = _MESH4.replace("1 400000.0 3900000.0\n", "2*1 3900000.0\n")
    with pytest.raises(ValueError):
        read_dat(_write(tmp_path, "m_grd.dat", text))
    # inside the node block (node 2) the message names the repetition
    text = _MESH4.replace("2 400001.0 3900000.0\n", "2*2 3900000.0\n")
    with pytest.raises(ValueError, match="different types"):
        read_dat(_write(tmp_path, "n_grd.dat", text))
    # a depth record repeating a REAL into REALs is fine
    d = _write(tmp_path, "d.dat", "Node Number = 2\n2*400000.0 10.0\n1.0 2.0 20.0\n")
    assert R.parse_dep_dat(d).tolist() == [10.0, 20.0]


# ---------------------------------------------------------------------------
# #3 one-element meshes (J == 1 .AND. LM1 /= 1)
# ---------------------------------------------------------------------------
_MESH1 = (
    "Node Number = 3\nCell Number = 1\n1 1 2 3 1\n"
    "1 400000.0 3900000.0 0\n2 400001.0 3900000.0 0\n3 400000.0 3900001.0 0\n"
)


def test_round21_one_element_mesh_is_refused_strict_and_read_permissive(tmp_path):
    f = _write(tmp_path, "m_grd.dat", _MESH1)
    with pytest.raises(ValueError, match=r"one element.*LM1 /= 1.*line 4"):
        read_dat(f)
    g = read_dat(f, allow_unordered=True)
    assert g["nv"][:, 0].tolist() == [0, 1, 2]
    assert g["x"].tolist() == [400000.0, 400001.0, 400000.0]


def test_round21_one_element_other_errors_come_first(tmp_path):
    # the one-element refusal is raised last: a broken node record is still named
    text = _MESH1.replace("2 400001.0 3900000.0 0\n", "2 400001.0 x\n")
    with pytest.raises(ValueError, match="node record 2 of 3"):
        read_dat(_write(tmp_path, "m_grd.dat", text))
    # an extra connectivity record after element 1 is named as such
    text = _MESH1.replace("1 1 2 3 1\n", "1 1 2 3 1\n2 1 3 2 1\n")
    with pytest.raises(ValueError, match="line 4: .*declared connectivity records"):
        read_dat(_write(tmp_path, "e_grd.dat", text))


def test_round21_two_element_mesh_still_reads(tmp_path):
    g = read_dat(_write(tmp_path, "m_grd.dat", _MESH4))
    assert g["nv"].T.tolist() == [[0, 1, 2], [1, 3, 2]]
    np.testing.assert_array_equal(g["x"], _X4)


# ---------------------------------------------------------------------------
# #4 NODECOUNT: the largest referenced node id equals Node Number
# ---------------------------------------------------------------------------
def test_round21_unreferenced_declared_node_is_refused(tmp_path):
    # the reviewer's reproduction: five nodes declared, connectivity references 1..4
    text = (
        _MESH4.replace("Node Number = 4", "Node Number = 5") + "5 400002.0 3900002.0\n"
    )
    f = _write(tmp_path, "m_grd.dat", text)
    with pytest.raises(
        ValueError, match=r"largest node id .* is 4, but 'Node Number = 5'"
    ):
        read_dat(f)
    g = read_dat(f, allow_unordered=True)
    assert g["x"].shape == (5,)


def test_round21_nodecount_uses_the_largest_id_not_every_id(tmp_path):
    # FVCOM checks only the maximum: an unreferenced node below it passes its check
    text = (
        "Node Number = 4\nCell Number = 2\n1 1 2 4\n2 2 4 1\n"
        "1 400000.0 3900000.0\n2 400001.0 3900000.0\n"
        "3 400000.0 3900001.0\n4 400001.0 3900001.0\n"
    )
    g = read_dat(_write(tmp_path, "m_grd.dat", text))
    assert g["x"].shape == (4,)
