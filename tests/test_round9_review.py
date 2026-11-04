# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Regression tests for review round 9 (gpt-6.1-sol, scratch/reviews/fulldomain/r9_final.md).

#1 FVCOM's domain-wide LOWSIG layer-thickness check (and finite / positive
depths), #2 grid records in FVCOM's row order on the runner path, #3 ADJUSTL
before comma normalisation in sigma.dat, #4 continued lines (comments included)
are rejected, #5 exact decimal-bearing serialisation of floating values, #6 no
one-value mandatory vectors are written.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xfvcom.grid import gtsz_builder
from xfvcom.grid.dat_reader import read_dat
from xfvcom.grid.grid_obj import FvcomGrid
from xfvcom.grid.gtsz import GtszSpec
from xfvcom.grid.gtsz_builder import MeshInputs, build_gtsz, load_mesh, suggest_zlev
from xfvcom.io.sigma_dat import gtsz_to_lines, read_sigma_dat, write_sigma_dat

_TB = Path.home() / "Github" / "TB-FVCOM"
_GRID = _TB / "input/goto2023/grid"
_SIGDIR = _TB / "hydro/tuning/sigmaz/input"
_HEAD = "NUMBER OF SIGMA LEVELS = 31\nSIGMA COORDINATE TYPE = SIGMAZ\nGTSZ BASE = 2\n"


def _read(tmp_path, body, head=_HEAD + "GTSZ NZ = 0\n"):
    p = tmp_path / "s_sigma.dat"
    p.write_text(head + body)
    return read_sigma_dat(p)


def _tri(h):
    return MeshInputs(
        x=np.array([0.0, 1000.0, 0.0]),
        y=np.array([0.0, 0.0, 1000.0]),
        nv=np.array([[0, 1, 2]], dtype=np.int64),
        h=np.asarray(h, dtype=np.float64),
    )


_LOWSIG = GtszSpec(kb=5, base=1, k1=3, k2=5, sadapt=True, lowsig=True, smax=0.01)


# ---------------------------------------------------------------------------
# #1 LOWSIG: H > 0 everywhere and FVCOM's domain-wide min-thickness check
# ---------------------------------------------------------------------------


def test_round9_lowsig_valid_mesh_builds():
    c = build_gtsz(_tri([10.0, 20.0, 30.0]), _LOWSIG)
    dz = (c.Z[:, :-1] - c.Z[:, 1:]) * c.H[:, None]
    assert (dz > 0).all()


@pytest.mark.parametrize("bad", [0.0, -3.0])
def test_round9_lowsig_nonpositive_depth_is_refused(bad):
    with pytest.raises(ValueError, match=r"1 node\(s\) have H <= 0 \(first node 2"):
        build_gtsz(_tri([10.0, bad, 30.0]), _LOWSIG)


def test_round9_nonfinite_depth_is_refused():
    with pytest.raises(ValueError, match=r"1 non-finite value\(s\), first node 3"):
        build_gtsz(_tri([10.0, 20.0, np.nan]), GtszSpec(kb=5))


def test_round9_nonpositive_depth_still_allowed_without_lowsig():
    # FVCOM checks H only for LOWSIG; other coordinates keep their behaviour
    c = build_gtsz(_tri([10.0, -1.0, 30.0]), GtszSpec(kb=5))
    assert c.Z.shape == (3, 5)


def test_round9_lowsig_thickness_check_is_domain_wide(monkeypatch):
    real = gtsz_builder.build_coordinate

    def broken(h, spec, href=None):
        Z = real(h, spec, href=href)
        Z[1, 3] = Z[1, 2]  # an interior node, a zero-thickness layer
        return Z

    monkeypatch.setattr(gtsz_builder, "build_coordinate", broken)
    with pytest.raises(ValueError, match=r"1 non-positive layer.*first node 2"):
        build_gtsz(_tri([10.0, 20.0, 30.0]), _LOWSIG)


_RAW = _GRID / "TokyoBay_dep_m7001tp2_raw.dat"
_CAP = _GRID / "TokyoBay_dep_m7001tp_rfac0p2_cap300.dat"
_SS = _SIGDIR / "tb_ss30_m7tp_kb31_sigma.dat"
_PROD = _SIGDIR / "tb_sz_m7prod_k30_s02_sigma.dat"


@pytest.mark.skipif(
    not all(p.is_file() for p in (_RAW, _CAP, _SS, _PROD, _GRID / "TokyoBay_grd.dat")),
    reason="production Tokyo Bay files not present",
)
def test_round9_production_lowsig_pairs():
    spec = read_sigma_dat(_SS).gtsz
    assert spec is not None and spec.lowsig
    # the reviewer's reproduction: ten nonpositive depths -> FVCOM stops
    with pytest.raises(ValueError, match=r"10 node\(s\) have H <= 0"):
        build_gtsz(load_mesh(_GRID / "TokyoBay_grd.dat", _RAW), spec)
    # the production depth file still builds, with LOWSIG and with the SADAPT spec
    mesh = load_mesh(_GRID / "TokyoBay_grd.dat", _CAP)
    build_gtsz(mesh, spec)
    prod = read_sigma_dat(_PROD).gtsz
    assert prod is not None
    build_gtsz(mesh, prod)


# ---------------------------------------------------------------------------
# #2 grid records must be in FVCOM's row order (strict by default)
# ---------------------------------------------------------------------------


def _grd(path, elems, nodes):
    path.write_text(
        f"Node Number = 4\nCell Number = 2\n{elems}"
        + "".join(f"{i} {x} {y} 0.0\n" for i, x, y in nodes)
    )
    return path


_ELEMS = "1 1 2 3 1\n2 2 4 3 1\n"
_NODES = [
    (1, 400000.0, 3900000.0),
    (2, 401000.0, 3900000.0),
    (3, 400000.0, 3901000.0),
    (4, 401000.0, 3901000.0),
]


def test_round9_swapped_node_records_are_refused_by_default(tmp_path):
    nodes = [_NODES[0], _NODES[2], _NODES[1], _NODES[3]]  # ids 1 3 2 4
    p = _grd(tmp_path / "s_grd.dat", _ELEMS, nodes)
    with pytest.raises(ValueError, match="node record 2 carries id 3"):
        read_dat(p)
    with pytest.raises(ValueError, match="row position"):
        FvcomGrid.from_dat(p, utm_zone=54)
    d = read_dat(p, allow_unordered=True)  # the explicit permissive mode
    assert d["x"].tolist() == [400000.0, 401000.0, 400000.0, 401000.0]
    g = FvcomGrid.from_dat(p, utm_zone=54, allow_unordered=True)
    assert g.node == 4


def test_round9_swapped_element_records_are_refused_by_default(tmp_path):
    p = _grd(tmp_path / "e_grd.dat", "2 2 4 3 1\n1 1 2 3 1\n", _NODES)
    with pytest.raises(ValueError, match="element record 1 carries id 2"):
        read_dat(p)
    assert read_dat(p, allow_unordered=True)["nv"][:, 0].tolist() == [0, 1, 2]


def test_round9_ordered_grid_reads_identically(tmp_path):
    p = _grd(tmp_path / "o_grd.dat", _ELEMS, _NODES)
    a, b = read_dat(p), read_dat(p, allow_unordered=True)
    assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["nv"], b["nv"])


@pytest.mark.skipif(
    not (_GRID / "TokyoBay_grd.dat").is_file(), reason="TokyoBay_grd.dat not present"
)
def test_round9_production_swap_is_refused(tmp_path):
    # the reviewer's reproduction: swap the complete records of nodes 3142 and 3016
    lines = (_GRID / "TokyoBay_grd.dat").read_text().splitlines(keepends=True)
    rows = {}
    for i, ln in enumerate(lines):
        p = ln.split()
        if len(p) == 4 and p[0] in ("3142", "3016") and "." in p[1]:
            rows[p[0]] = i
    assert set(rows) == {"3142", "3016"}
    a, b = rows["3142"], rows["3016"]
    lines[a], lines[b] = lines[b], lines[a]
    p = tmp_path / "TokyoBay_grd.dat"
    p.write_text("".join(lines))
    with pytest.raises(ValueError, match="carries id 3142"):
        FvcomGrid.from_dat(p, utm_zone=54)
    FvcomGrid.from_dat(_GRID / "TokyoBay_grd.dat", utm_zone=54)  # unchanged file: fine


# ---------------------------------------------------------------------------
# #3 ADJUSTL first, then commas -> blanks: a leading comma spoils the key
# ---------------------------------------------------------------------------


def test_round9_leading_comma_key_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not spelled exactly 'GTSZ P1'"):
        _read(tmp_path, ",GTSZ P1 = 4.0\n")
    with pytest.raises(ValueError, match="not spelled exactly 'GTSZ P1'"):
        _read(tmp_path, "  ,GTSZ P1 = 4.0\n")


def test_round9_leading_blanks_and_trailing_commas_still_read(tmp_path):
    g = _read(tmp_path, "   GTSZ P1 = 4.0,\n").gtsz
    assert g is not None and g.p1 == 4.0


# ---------------------------------------------------------------------------
# #4 continued lines (comments too) are rejected by the reader and the writer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", ["\\\\", "\\\\\\\\", "\\\\  "])
def test_round9_continued_comment_is_refused(tmp_path, suffix):
    with pytest.raises(ValueError, match="line 5 ends in a backslash pair"):
        _read(tmp_path, f"! a comment {suffix}\nGTSZ P1 = 4.0\n")


def test_round9_continued_data_line_is_refused(tmp_path):
    with pytest.raises(ValueError, match="backslash pair"):
        _read(tmp_path, "OTHER KEY = x \\\\\nGTSZ P1 = 4.0\n")


def test_round9_single_backslash_comment_still_reads(tmp_path):
    g = _read(tmp_path, "! path C:\\\nGTSZ P1 = 4.0\n").gtsz
    assert g is not None and g.p1 == 4.0


def test_round9_writer_refuses_continued_header_comment(tmp_path):
    with pytest.raises(ValueError, match="backslash pair"):
        gtsz_to_lines(GtszSpec(kb=5), header_comment="made by x \\\\\\\\")
    with pytest.raises(ValueError, match="backslash pair"):
        gtsz_to_lines(GtszSpec(kb=5), header_comment="a\nb \\\\\nc")
    lines = gtsz_to_lines(GtszSpec(kb=5), header_comment="plain\none \\ slash")
    assert lines[:2] == ["! plain", "! one \\ slash"]


# ---------------------------------------------------------------------------
# #5 floating values are written exactly, with a decimal point
# ---------------------------------------------------------------------------


def _roundtrip(tmp_path, spec):
    p = tmp_path / "rt_sigma.dat"
    write_sigma_dat(p, spec)
    g = read_sigma_dat(p).gtsz
    assert g is not None
    return g


@pytest.mark.parametrize("smax", [0.0000125, 4e-7, 0.02, 1.0 / 3.0])
def test_round9_smax_round_trips_exactly(tmp_path, smax):
    spec = GtszSpec(kb=31, base=1, k1=3, k2=31, sadapt=True, lowsig=True, smax=smax)
    g = _roundtrip(tmp_path, spec)
    assert g.smax == smax and g.smax > 0
    assert g == spec


def test_round9_dref_and_lowfrac_round_trip_exactly(tmp_path):
    lowfrac = (1e-7,) + ((1 - 1e-7) / 27,) * 27
    spec = GtszSpec(
        kb=31,
        base=1,
        k1=3,
        k2=31,
        sadapt=True,
        lowsig=True,
        smax=0.005,
        dref=27.123456789,
        lowfrac=lowfrac,
    )
    with pytest.raises(ValueError, match="400"):
        # 28 exact 18-digit fractions do not fit FVCOM's 400-character line
        gtsz_to_lines(spec)
    lowfrac = (1e-7,) + (0.25 - 1e-7, 0.25, 0.25, 0.25)
    spec = GtszSpec(
        kb=8,
        base=1,
        k1=3,
        k2=8,
        sadapt=True,
        lowsig=True,
        smax=0.005,
        dref=27.123456789,
        lowfrac=lowfrac,
    )
    text = "\n".join(gtsz_to_lines(spec))
    assert "GTSZ LOWFRAC = 1e-07" not in text and "1.0e-07" in text
    g = _roundtrip(tmp_path, spec)
    assert g.lowfrac == lowfrac and g.lowfrac[0] > 0
    assert g.dref == 27.123456789
    assert g == spec


def test_round9_integral_values_keep_a_decimal_point():
    lines = gtsz_to_lines(GtszSpec(kb=5, base=1, smax=1e16))
    assert "GTSZ SMAX = 1.0e+16" in lines
    assert "GTSZ SMAX = 0.0" in gtsz_to_lines(GtszSpec(kb=5))


def test_round9_suggested_levels_fit_and_round_trip(tmp_path):
    zlev = suggest_zlev(300.0)
    assert len(zlev) > 20
    spec = GtszSpec(
        kb=len(zlev) + 11,
        k1=10,
        nz=len(zlev),
        zlev=tuple(float(z) for z in zlev),
        mask=True,
        sadapt=True,
        smax=0.0008,
    )
    assert _roundtrip(tmp_path, spec) == spec


@pytest.mark.skipif(
    not (_SS.is_file() and _CAP.is_file()), reason="production LOWSIG files not present"
)
def test_round9_production_lowsig_spec_round_trips(tmp_path):
    spec = read_sigma_dat(_SS).gtsz
    assert spec is not None
    g = _roundtrip(tmp_path, spec)
    assert g == spec
    mesh = load_mesh(_GRID / "TokyoBay_grd.dat", _CAP)
    np.testing.assert_array_equal(build_gtsz(mesh, g).Z, build_gtsz(mesh, spec).Z)


# ---------------------------------------------------------------------------
# #6 one-value mandatory vectors are valid in memory but not written
# ---------------------------------------------------------------------------


def test_round9_single_zlev_is_not_written():
    spec = GtszSpec(kb=4, k1=1, k2=3, nz=1, zlev=(-2.0,))
    spec.validate()
    with pytest.raises(ValueError, match="GTSZ ZLEV has 1 value"):
        gtsz_to_lines(spec)


def test_round9_single_bpg_ref_zlev_is_not_written(tmp_path):
    spec = GtszSpec(kb=3, bpg_ref_zlev=(0.0,))
    spec.validate()
    with pytest.raises(ValueError, match="BPG REF ZLEV has one value"):
        write_sigma_dat(tmp_path / "b_sigma.dat", spec)
    assert not (tmp_path / "b_sigma.dat").exists()
    two = GtszSpec(kb=3, bpg_ref_zlev=(-1.0, -2.0))
    assert _roundtrip(tmp_path, two) == two
