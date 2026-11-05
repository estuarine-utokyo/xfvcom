# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Regression tests for review round 18 (gpt-6.1-sol, scratch/reviews/fulldomain/r18_final.md).

One shared FVCOM header/record scanner (xfvcom/fvcom_input.py) now serves read_dat, read_dep and
the runner's parse_obc_dat / parse_dep_dat:

#1 an overlong '!' comment after a complete count header is harmless (the 400-byte cut keeps it),
#2 the 400-byte INPLINE buffer counts UTF-8 bytes, not characters,
#3 blank lines are not records; a comment-only line before the first record is a header line;
   a comment-only line INSIDE the record block is still refused (gfortran/ifx fail on it and
   FVCOM's READ_COLDSTART_* loops count it as a row) with a message saying so,
#4 the runner accepts 'OBC Node Number = 13,' and 'Node Number = +3210',
#5 a tab in a count value is refused (GET_VALUE types it as a string),
#6 every header line SCAN_FILE traverses is converted like GET_VALUE ('mesh_scale = 1e3' stops it),
#7 list-directed repetition 'r*value' is expanded (bounded); repeated nulls are refused,
#8 load_siglev_npz refuses complex (non-real) siglev / h before casting.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# jsasaki 2026-11-05: review round 21 #3: one-element meshes are refused by the strict reader
from tests.one_element import read_one_element
from xfvcom.fortran_numbers import MAX_LIST_FIELDS, list_fields
from xfvcom.grid.dat_reader import read_dat
from xfvcom.io.dep_reader import read_dep

_ROOT = Path(__file__).resolve().parents[1]


def _runner():
    spec = importlib.util.spec_from_file_location(
        "run_jcope_obc_r18", _ROOT / "scripts" / "run_jcope_obc.py"
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
_DEP = (
    "Node Number = 3\n"
    "400000.0 3900000.0 10.0\n"
    "400001.0 3900000.0 20.0\n"
    "400000.0 3900001.0 30.0\n"
)
_OBC = "OBC Node Number = 2\n1 3 1\n2 1 1\n"


def _mesh(tmp_path: Path, header: str) -> Path:
    return _write(tmp_path, "m_grd.dat", _MESH.replace("Node Number = 3", header, 1))


# ---------------------------------------------------------------------------
# #1 harmless long comment
# ---------------------------------------------------------------------------
_LONG = "Node Number = 3 ! " + "x" * 400


def test_round18_long_comment_after_count_is_accepted(tmp_path):
    # jsasaki 2026-11-05: review round 21 #3: FVCOM cannot read a one-element mesh (strict refuses it
    # after its other checks); tests/one_element.py
    assert read_one_element(_mesh(tmp_path, _LONG))["x"].shape == (3,)
    dep = _write(tmp_path, "d.dat", _DEP.replace("Node Number = 3", _LONG, 1))
    np.testing.assert_array_equal(read_dep(dep).h, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(R.parse_dep_dat(dep, _XY), [10.0, 20.0, 30.0])
    obc = _write(tmp_path, "o.dat", _OBC.replace("= 2", "= 2 ! " + "y" * 500, 1))
    np.testing.assert_array_equal(R.parse_obc_dat(obc), [3, 1])


def test_round18_value_beyond_the_buffer_is_still_refused(tmp_path):
    line = "Node Number =" + " " * 400 + "3 ! c"
    with pytest.raises(ValueError, match="first 400 characters"):
        read_dat(_mesh(tmp_path, line))


# ---------------------------------------------------------------------------
# #2 bytes, not characters
# ---------------------------------------------------------------------------
_MB = "Node Number" + " " * 29 + "あ" * 120 + " = 3"


def test_round18_multibyte_header_counts_bytes(tmp_path):
    assert len(_MB) < 400 < len(_MB.encode("utf-8"))
    with pytest.raises(ValueError, match="first 400 characters"):
        read_dat(_mesh(tmp_path, _MB))
    dep = _write(tmp_path, "d.dat", _DEP.replace("Node Number = 3", _MB, 1))
    with pytest.raises(ValueError, match="first 400 characters"):
        R.parse_dep_dat(dep)
    with pytest.raises(ValueError, match="first 400 characters"):
        read_dep(dep)


def test_round18_short_multibyte_comment_is_fine(tmp_path):
    # jsasaki 2026-11-05: review round 21 #3: FVCOM cannot read a one-element mesh (strict refuses it
    # after its other checks); tests/one_element.py
    assert read_one_element(_mesh(tmp_path, "Node Number = 3 ! 節点数"))["x"].shape == (
        3,
    )


# ---------------------------------------------------------------------------
# #3 blank and comment-only lines in depth / OBC files
# ---------------------------------------------------------------------------
def test_round18_blank_lines_and_leading_comments_are_not_records(tmp_path):
    obc = _write(tmp_path, "o.dat", "! title\n" + _OBC.replace("1 3 1\n", "1 3 1\n\n"))
    np.testing.assert_array_equal(R.parse_obc_dat(obc), [3, 1])
    obc = _write(tmp_path, "p.dat", _OBC.replace("\n1 3 1", "\n! first segment\n1 3 1"))
    np.testing.assert_array_equal(R.parse_obc_dat(obc), [3, 1])


def test_round18_comment_inside_the_record_block_is_refused(tmp_path):
    # gfortran (IOSTAT 5010) and ifx (59) fail on '! segment note'; READ_COLDSTART_OBC_GRID counts
    # that READ as a row (only IOS < 0 ends the loop), so FVCOM stops on 3 rows for 2 declared
    obc = _write(tmp_path, "o.dat", _OBC.replace("1 3 1\n", "1 3 1\n! segment note\n"))
    with pytest.raises(ValueError, match=r"line 3: comment-only line .*declares 2"):
        R.parse_obc_dat(obc)
    dep = _write(tmp_path, "d.dat", _DEP.replace("20.0\n", "20.0\n   ! note\n", 1))
    with pytest.raises(ValueError, match="comment-only line inside the record block"):
        R.parse_dep_dat(dep)


# ---------------------------------------------------------------------------
# #4 runner count-header syntax
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "header",
    ["OBC Node Number = 2,", "OBC Node Number = +2", "  OBC Node Number=2 ! c"],
)
def test_round18_runner_obc_header_forms(tmp_path, header):
    obc = _write(tmp_path, "o.dat", _OBC.replace("OBC Node Number = 2", header, 1))
    np.testing.assert_array_equal(R.parse_obc_dat(obc), [3, 1])


@pytest.mark.parametrize("header", ["Node Number = +3", "Node Number = 3,"])
def test_round18_runner_depth_header_forms(tmp_path, header):
    dep = _write(tmp_path, "d.dat", _DEP.replace("Node Number = 3", header, 1))
    np.testing.assert_array_equal(R.parse_dep_dat(dep, _XY), [10.0, 20.0, 30.0])


@pytest.mark.parametrize(
    "header",
    [
        "OBC Node Number = -2",
        "OBC Node Number = 0",
        "OBC Node Number = 2.0",
        "obc node number = 2",
    ],
)
def test_round18_runner_bad_obc_headers_are_refused(tmp_path, header):
    obc = _write(tmp_path, "o.dat", _OBC.replace("OBC Node Number = 2", header, 1))
    with pytest.raises(ValueError, match="positive integer|not found"):
        R.parse_obc_dat(obc)


# ---------------------------------------------------------------------------
# #5 tabs in a count value
# ---------------------------------------------------------------------------
def test_round18_tab_in_count_value_is_refused(tmp_path):
    with pytest.raises(ValueError, match="tab"):
        read_dat(_mesh(tmp_path, "Node Number =\t3"))
    dep = _write(tmp_path, "d.dat", _DEP.replace("= 3", "= 3\t", 1))
    with pytest.raises(ValueError, match="tab"):
        R.parse_dep_dat(dep)
    obc = _write(tmp_path, "o.dat", _OBC.replace("= 2", "=\t2", 1))
    with pytest.raises(ValueError, match="tab"):
        R.parse_obc_dat(obc)


def test_round18_tabs_in_records_are_still_separators(tmp_path):
    obc = _write(tmp_path, "o.dat", _OBC.replace("1 3 1", "1\t3\t1", 1))
    np.testing.assert_array_equal(R.parse_obc_dat(obc), [3, 1])


# ---------------------------------------------------------------------------
# #6 GET_VALUE conversion of every traversed header line
# ---------------------------------------------------------------------------
def test_round18_unconvertible_numeric_header_is_refused(tmp_path):
    f = _write(tmp_path, "m_grd.dat", "mesh_scale = 1e3\n" + _MESH)
    with pytest.raises(ValueError, match="'1e3' is not a valid integer"):
        read_dat(f)
    dep = _write(tmp_path, "d.dat", "mesh_scale = 1e3\n" + _DEP)
    with pytest.raises(ValueError, match="not a valid integer"):
        R.parse_dep_dat(dep)
    with pytest.raises(ValueError, match="not a valid integer"):
        read_dep(dep)
    obc = _write(tmp_path, "o.dat", "scale = 1.0.0\n" + _OBC)
    with pytest.raises(ValueError, match="not a valid floating value"):
        R.parse_obc_dat(obc)


def test_round18_convertible_and_untraversed_lines_pass(tmp_path):
    # a valid float, a string value and a line after both headers (never traversed)
    # jsasaki 2026-11-05: review round 21 #3: FVCOM cannot read a one-element mesh (strict refuses it
    # after its other checks); tests/one_element.py
    text = "mesh_scale = 1.0e3\ntitle = Tokyo Bay\nNode Number = 3\nCell Number = 1\n"
    assert read_one_element(_write(tmp_path, "a_grd.dat", text + _ELEM + _NODES))[
        "x"
    ].shape == (3,)
    text = "Node Number = 3\nCell Number = 1\nunused = 1e3\n"
    assert read_one_element(_write(tmp_path, "b_grd.dat", text + _ELEM + _NODES))[
        "x"
    ].shape == (3,)


def test_round18_continued_header_line_is_refused(tmp_path):
    f = _write(tmp_path, "m_grd.dat", "! note \\\\\n" + _MESH)
    with pytest.raises(ValueError, match="continuation"):
        read_dat(f)


# ---------------------------------------------------------------------------
# #7 list-directed repetition
# ---------------------------------------------------------------------------
def test_round18_list_fields_repetition():
    assert list_fields("1 1*3150 1") == ["1", "3150", "1"]
    assert list_fields("3 2*7") == ["3", "7", "7"]
    assert list_fields("1 2*") == ["1", "", ""]  # repeated null values
    assert list_fields("0*5 *5") == [
        "0*5",
        "*5",
    ]  # not repetitions: no conversion accepts them
    # jsasaki 2026-11-05: review round 19 #2: no silent cap -- an oversized count is an explicit error
    with pytest.raises(ValueError, match="exceeds 200000000"):
        list_fields("1 " + "9" * 40 + "*0")
    with pytest.raises(ValueError, match="is not supported"):
        list_fields(f"1 {MAX_LIST_FIELDS}*0")


def test_round18_repetition_in_records(tmp_path):
    obc = _write(tmp_path, "o.dat", _OBC.replace("1 3 1", "1 1*3 1", 1))
    np.testing.assert_array_equal(R.parse_obc_dat(obc), [3, 1])
    dep = _write(
        tmp_path,
        "d.dat",
        _DEP.replace("400001.0 3900000.0 20.0", "400001.0 1*3900000.0 20.0"),
    )
    np.testing.assert_array_equal(R.parse_dep_dat(dep, _XY), [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(read_dep(dep).h, [10.0, 20.0, 30.0])
    # jsasaki 2026-11-05: review round 21 #3: FVCOM cannot read a one-element mesh (strict refuses it
    # after its other checks); tests/one_element.py
    m = read_one_element(
        _write(tmp_path, "m_grd.dat", _MESH.replace("1 1 2 3", "1 1 2 1*3", 1))
    )
    np.testing.assert_array_equal(m["nv"][:, 0], [0, 1, 2])


def test_round18_repeated_null_is_refused(tmp_path):
    obc = _write(tmp_path, "o.dat", _OBC.replace("1 3 1", "1 2*", 1))
    with pytest.raises(ValueError, match="malformed OBC record"):
        R.parse_obc_dat(obc)
    obc = _write(tmp_path, "p.dat", _OBC.replace("1 3 1", "1 0*3 1", 1))
    with pytest.raises(ValueError, match="malformed OBC record"):
        R.parse_obc_dat(obc)


# ---------------------------------------------------------------------------
# #8 complex arrays in the siglev npz
# ---------------------------------------------------------------------------
def _npz(tmp_path: Path, siglev, h) -> Path:
    f = tmp_path / "s.npz"
    np.savez(f, siglev=siglev, obc_node=np.array([5]), h=h)
    return f


def test_round18_complex_siglev_is_refused(tmp_path):
    z = np.array([[0, complex(-0.5, np.nan), -1]], dtype=complex)
    with pytest.raises(ValueError, match="'siglev' must be real numbers"):
        R.load_siglev_npz(
            _npz(tmp_path, z, np.array([10.0])), np.array([5]), np.array([10.0])
        )


def test_round18_complex_h_is_refused(tmp_path):
    z = np.array([[0.0, -0.5, -1.0]])
    h = np.array([complex(10.0, np.inf)])
    with pytest.raises(ValueError, match="'h' must be real numbers"):
        R.load_siglev_npz(_npz(tmp_path, z, h), np.array([5]), np.array([10.0]))


def test_round18_real_npz_still_loads(tmp_path):
    z = np.array([[0.0, -0.5, -1.0]], dtype=np.float32)
    got = R.load_siglev_npz(
        _npz(tmp_path, z, np.array([10.0])), np.array([5]), np.array([10.0])
    )
    np.testing.assert_array_equal(got, [[0.0, -0.5, -1.0]])
