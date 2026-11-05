# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Regression tests for review round 15 (gpt-6.1-sol, scratch/reviews/fulldomain/r15_final.md).

#1 compare_jcope_sources refuses a region variable on non-canonical dimensions (e.g. TT on
   ``wrong_level``) and transposes explicitly to the canonical order,
#2 read_dat parses each count header as ONE positive integer (spaces and a trailing '!'
   comment allowed); fractional, negative, zero and malformed values are refused,
#3 read_dat refuses data records after the declared connectivity and node blocks,
#4 run_jcope_obc.parse_dep_dat / parse_obc_dat ignore trailing '!' comments and unused
   trailing columns, keeping the count, finiteness and mesh-identity checks,
#5 read_dat finds 'Cell Number' and 'Node Number' in either order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xfvcom.grid.dat_reader import read_dat
from xfvcom.grid.grid_obj import FvcomGrid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import compare_jcope_sources as cmp  # noqa: E402  # isort: skip
import run_jcope_obc as r  # noqa: E402  # isort: skip

_HEAD = "Node Number = 3\nCell Number = 1\n"
_ELEM = "1 1 2 3 1\n"
_NODES = "1 0.0 0.0\n2 1.0 0.0\n3 0.0 1.0\n"


def _mesh(tmp_path: Path, text: str) -> Path:
    f = tmp_path / "m_grd.dat"
    f.write_text(text)
    return f


def _same(a, b):
    for k in ("x", "y", "nv"):
        np.testing.assert_array_equal(a[k], b[k])


# ---------------------------------------------------------------------------
# #1 canonical dimensions in the comparison
# ---------------------------------------------------------------------------
def _region(tt_dims=("time", "level", "lat", "lon"), tt_data=None):
    nt, nk, ny, nx = 2, 46, 3, 4
    data = np.arange(nt * nk * ny * nx, dtype=np.float64).reshape(nt, nk, ny, nx)
    coords = dict(
        time=pd.date_range("2020-01-01", periods=nt, freq="h"),
        level=np.arange(nk),
        lat=np.linspace(35.0, 35.1, ny),
        lon=np.linspace(139.0, 139.1, nx),
    )
    ds = xr.Dataset(coords=coords)
    for name in ("ST", "UT", "VT"):
        ds[name] = (("time", "level", "lat", "lon"), data)
    ds["EGT"] = (("time", "lat", "lon"), data[:, 0])
    if tt_data is None:
        tt_data = data
    ds["TT"] = (tt_dims, tt_data)
    return ds, data


def test_round15_wrong_level_dimension_is_refused():
    ds, data = _region(
        tt_dims=("time", "wrong_level", "lat", "lon"),
        tt_data=np.arange(2 * 46 * 3 * 4, dtype=np.float64).reshape(2, 46, 3, 4),
    )
    ds = ds.assign_coords(wrong_level=np.arange(45, -1, -1))
    with pytest.raises(SystemExit, match="wrong_level"):
        cmp.region_columns(
            ds, "TT", np.array([0, 1]), np.array([0, 2]), np.array([1, 3])
        )


def test_round15_canonical_columns_and_explicit_transpose():
    ds, data = _region()
    pos, rj, ri = np.array([1, 0]), np.array([0, 2]), np.array([1, 3])
    ref = cmp.region_columns(ds, "TT", pos, rj, ri)
    assert ref.shape == (2, 46, 2)
    np.testing.assert_array_equal(ref, data[pos][:, :, rj, ri])
    # the same field stored in another dimension order gives the same columns
    perm = ds.copy()
    perm["TT"] = ds["TT"].transpose("lon", "level", "time", "lat")
    np.testing.assert_array_equal(cmp.region_columns(perm, "TT", pos, rj, ri), ref)
    egt = cmp.region_columns(ds, "EGT", pos, rj, ri)
    np.testing.assert_array_equal(egt, data[pos][:, 0, rj, ri])


def test_round15_missing_dimension_is_refused():
    ds, data = _region()
    ds["EGT"] = (("time", "lat", "lon", "level"), data.transpose(0, 2, 3, 1))
    with pytest.raises(SystemExit, match="EGT"):
        cmp.region_columns(ds, "EGT", np.array([0]), np.array([0]), np.array([0]))


def test_round15_no_shape_assert_left():
    src = Path(cmp.__file__).read_text()
    assert "assert got.shape" not in src


# ---------------------------------------------------------------------------
# #2 count headers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "head",
    [
        "Node Number = 3.5\nCell Number = 1\n",
        "Node Number = 3\nCell Number = 1.5\n",
        "Node Number = -3\nCell Number = 1\n",
        "Node Number = 3\nCell Number = -1\n",
        "Node Number = 0\nCell Number = 1\n",
        "Node Number = 3 4\nCell Number = 1\n",
        "Node Number = 3x\nCell Number = 1\n",
        "Node Number = 1e3\nCell Number = 1\n",
        "Node Number 3\nCell Number = 1\n",
        "Node Number =\nCell Number = 1\n",
    ],
)
def test_round15_malformed_count_headers_are_refused(tmp_path, head):
    # jsasaki 2026-11-05: review round 17 #2: a line without '=' is no header for SCAN_FILE (not found)
    with pytest.raises(ValueError, match="positive integer|counts not found"):
        read_dat(_mesh(tmp_path, head + _ELEM + _NODES))


def test_round15_fractional_counts_refused_by_from_dat(tmp_path):
    f = _mesh(tmp_path, "Node Number = 3.5\nCell Number = 1.5\n" + _ELEM + _NODES)
    with pytest.raises(ValueError):
        FvcomGrid.from_dat(f, utm_zone=54)


@pytest.mark.parametrize(
    "head",
    [
        "Node Number=3\nCell Number=1\n",
        "Node Number =   +3   \nCell Number = 1 ! one triangle\n",
        "Node Number = 3 ! nodes\nCell Number = 1\n",
    ],
)
def test_round15_valid_count_headers(tmp_path, head):
    plain = read_dat(_mesh(tmp_path, _HEAD + _ELEM + _NODES))
    _same(read_dat(_mesh(tmp_path, head + _ELEM + _NODES)), plain)


def test_round15_duplicate_count_header_is_refused(tmp_path):
    f = _mesh(
        tmp_path, "Node Number = 3\nNode Number = 4\nCell Number = 1\n" + _ELEM + _NODES
    )
    with pytest.raises(ValueError, match="duplicate count header"):
        read_dat(f)


# ---------------------------------------------------------------------------
# #3 records after the declared blocks
# ---------------------------------------------------------------------------
def test_round15_extra_node_record_is_refused(tmp_path):
    f = _mesh(tmp_path, _HEAD + _ELEM + _NODES + "3 0.0 0.0\n")
    with pytest.raises(ValueError, match=r"line 7: .*follows the 3 declared node"):
        read_dat(f)
    with pytest.raises(ValueError, match="follows the 3 declared node"):
        read_dat(f, allow_unordered=True)
    with pytest.raises(ValueError):
        FvcomGrid.from_dat(f, utm_zone=54)


def test_round15_extra_text_after_nodes_is_refused(tmp_path):
    f = _mesh(tmp_path, _HEAD + _ELEM + _NODES + "\nfoo\n")
    with pytest.raises(ValueError, match="line 8"):
        read_dat(f)


def test_round15_extra_connectivity_record_is_refused(tmp_path):
    f = _mesh(tmp_path, _HEAD + _ELEM + "2 1 2 3\n" + _NODES)
    with pytest.raises(ValueError, match=r"line 4: .*declared connectivity"):
        read_dat(f)
    # without the order check the extra record still overflows the node block
    with pytest.raises(ValueError):
        read_dat(f, allow_unordered=True)


def test_round15_blank_and_comment_lines_may_follow(tmp_path):
    plain = read_dat(_mesh(tmp_path, _HEAD + _ELEM + _NODES))
    # jsasaki 2026-11-05: review round 20 #4: only blank lines may follow; FVCOM counts a comment-only
    # line between or after the blocks as one more row (tests/test_round20_review.py)
    f = _mesh(tmp_path, _HEAD + _ELEM + "\n" + _NODES + "\n   \n")
    _same(read_dat(f), plain)


# ---------------------------------------------------------------------------
# #4 depth / OBC records with comments and extra columns
# ---------------------------------------------------------------------------
def test_round15_depth_trailing_comment_and_columns(tmp_path):
    f = tmp_path / "d.dat"
    f.write_text("Node Number = 1\n400000.0 3900000.0 10.0 ! surveyed depth\n")
    assert r.parse_dep_dat(f).tolist() == [10.0]
    f.write_text("Node Number = 1 ! one\n400000.0 3900000.0 10.0 7 extra\n")
    assert r.parse_dep_dat(f, np.array([[4.0e5, 3.9e6]])).tolist() == [10.0]


def test_round15_depth_checks_are_kept(tmp_path):
    f = tmp_path / "d.dat"
    # the comment is not a field: only two fields remain
    f.write_text("Node Number = 1\n400000.0 3900000.0 ! 10.0\n")
    with pytest.raises(ValueError, match="malformed depth record"):
        r.parse_dep_dat(f)
    # a comment-only line inside the record block is counted by FVCOM as a bad row
    # jsasaki 2026-11-05: review round 18 #3: before the first record it is a header line (FVCOM's
    # header-skip loop passes over it); inside the block it still breaks the row count
    f.write_text("Node Number = 2\n400000.0 3900000.0 10.0\n! note\n4.0e5 3.9e6 1.0\n")
    with pytest.raises(ValueError, match="declares 2 record"):
        r.parse_dep_dat(f)
    f.write_text("Node Number = 1\n400000.0 3900000.0 inf ! x\n")
    with pytest.raises(ValueError, match="malformed depth record"):
        r.parse_dep_dat(f)
    f.write_text("Node Number = 2\n400000.0 3900000.0 10.0 ! x\n")
    with pytest.raises(ValueError, match="declares 2 record"):
        r.parse_dep_dat(f)
    f.write_text("Node Number = 1\n400001.0 3900000.0 10.0 ! x\n")
    with pytest.raises(ValueError, match="does not belong"):
        r.parse_dep_dat(f, np.array([[4.0e5, 3.9e6]]))


def test_round15_obc_trailing_comment_and_columns(tmp_path):
    f = tmp_path / "o.dat"
    f.write_text("OBC Node Number = 2 ! two\n1 5 1 ! first\n2 7 1 0.5 extra\n")
    assert r.parse_obc_dat(f).tolist() == [5, 7]
    f.write_text("OBC Node Number = 2\n1 5 ! 1\n2 7 1\n")
    with pytest.raises(ValueError, match="malformed OBC record"):
        r.parse_obc_dat(f)


# ---------------------------------------------------------------------------
# #5 header order
# ---------------------------------------------------------------------------
def test_round15_cell_header_first(tmp_path):
    plain = read_dat(_mesh(tmp_path, _HEAD + _ELEM + _NODES))
    swapped = read_dat(
        _mesh(tmp_path, "Cell Number = 1\nNode Number = 3\n" + _ELEM + _NODES)
    )
    _same(swapped, plain)
    # other header lines around them are still skipped
    other = read_dat(
        _mesh(
            tmp_path,
            "! mesh\nCell Number = 1\nTitle = test\nNode Number = 3\n" + _ELEM + _NODES,
        )
    )
    _same(other, plain)


def test_round15_header_after_block_start_is_refused(tmp_path):
    f = _mesh(tmp_path, "Cell Number = 1\n" + _ELEM + "Node Number = 3\n" + _NODES)
    with pytest.raises(ValueError, match="follows the start of the connectivity block"):
        read_dat(f)


def test_round15_header_beyond_scan_file_limit_is_refused(tmp_path):
    pad = "! pad\n" * 200
    f = _mesh(
        tmp_path, "Cell Number = 1\n" + pad + "Node Number = 3\n" + _ELEM + _NODES
    )
    # jsasaki 2026-11-05: review round 16 #2: the limit is line 201 (header here on line 202)
    with pytest.raises(ValueError, match="beyond line 201"):
        read_dat(f)
