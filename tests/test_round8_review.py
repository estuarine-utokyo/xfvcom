# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Regression tests for review round 8 (gpt-6.1-sol, scratch/reviews/fulldomain/r8_final.md).

#1 depth-weighted element aperture FRAC1, #2 commas normalised once in sigma.dat
values, #3 space-only trimming (tabs / CR rejected), #4 case-sensitive coordinate
type, #5 FVCOM scanner limits (201-line header search, 400-character lines,
150-value lists) in the reader and the writer, #6 GTSZ MASK / masked-column /
BPG REF NZ startup checks, #7 one minimum layer count (two layers), #8 the
full-domain archive root is absolute from construction.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from xfvcom.grid.gtsz import GtszSpec, compute_kbp1_frac1
from xfvcom.grid.gtsz_builder import MeshInputs, build_gtsz, load_mesh
from xfvcom.io.jcope_obc_generator import _validate_sigma
from xfvcom.io.sigma_dat import gtsz_to_lines, read_sigma_dat, write_sigma_dat

sys.path.insert(0, str(Path(__file__).parent))
from test_jcope_fulldomain import archives  # noqa: E402,F401  # isort: skip

_HEAD = "NUMBER OF SIGMA LEVELS = 31\nSIGMA COORDINATE TYPE = SIGMAZ\nGTSZ BASE = 2\n"


def _read(tmp_path, body, head=_HEAD + "GTSZ NZ = 0\n"):
    p = tmp_path / "s_sigma.dat"
    p.write_text(head + body)
    return read_sigma_dat(p)


# ---------------------------------------------------------------------------
# #1 FRAC1 = sum(H_i f_i) / sum(H_i), f_i = -Z_i(KBP1+1)  (mod_setup.F)
# ---------------------------------------------------------------------------


def test_round8_frac1_is_depth_weighted():
    Z = np.array([[0.0, -0.5, -1.0], [0.0, -0.2, -1.0], [0.0, -0.3, -1.0]], dtype=float)
    nv = np.array([[0, 1, 2]])
    kbp = np.array([1, 2, 2])
    H = np.array([10.0, 20.0, 30.0])
    kbp1, frac1 = compute_kbp1_frac1(Z, nv, kbp, H)
    assert kbp1.tolist() == [1]
    # (10*0.5 + 20*0.2 + 30*0.3) / 60 = 0.3; the obsolete plain mean was 1/3
    np.testing.assert_allclose(frac1, [0.3], rtol=0, atol=1e-15)


_TB = Path.home() / "Github" / "TB-FVCOM"
_GRID = _TB / "input/goto2023/grid"
_SPEC = _TB / "hydro/tuning/sigmaz/input/tb_sz_m7prod_k30_s02_sigma.dat"


@pytest.mark.skipif(
    not (_SPEC.is_file() and (_GRID / "TokyoBay_grd.dat").is_file()),
    reason="production Tokyo Bay mesh / SADAPT spec not present",
)
def test_round8_frac1_production_element():
    mesh = load_mesh(
        _GRID / "TokyoBay_grd.dat", _GRID / "TokyoBay_dep_m7001tp_rfac0p2_cap300.dat"
    )
    spec = read_sigma_dat(_SPEC).gtsz
    assert spec is not None
    c = build_gtsz(mesh, spec)
    assert c.frac1 is not None
    e = 5309 - 1
    assert c.frac1[e] == pytest.approx(0.8947629271, abs=1e-9)
    aper = c.frac1[e] * mesh.h[mesh.nv[e]].mean()
    assert aper == pytest.approx(131.373972, abs=1e-5)


# ---------------------------------------------------------------------------
# #2 commas are normalised once, for typing AND conversion
# ---------------------------------------------------------------------------


def test_round8_trailing_commas_keep_their_meaning(tmp_path):
    head = _HEAD.replace("= 31", "= 31,") + "GTSZ NZ = 0,\n"
    g = _read(
        tmp_path,
        "GTSZ SADAPT = T,\nGTSZ LOWSIG = T,\nGTSZ SMAX = 0.02,\nGTSZ K1 = 2,\n",
        head=head,
    )
    assert g.kb == 31
    assert g.gtsz is not None
    assert g.gtsz.sadapt is True and g.gtsz.lowsig is True
    assert g.gtsz.smax == 0.02 and g.gtsz.k1 == 2
    head = _HEAD + "GTSZ NZ = 2\nGTSZ K1 = 2\nGTSZ K2 = 5\n"
    assert _read(tmp_path, "GTSZ ZLEV = -2.0,-4.0\n", head=head).gtsz.zlev == (
        -2.0,
        -4.0,
    )


# ---------------------------------------------------------------------------
# #3 FVCOM trims spaces only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "GTSZ P1 = \t4.0\n",
        "GTSZ P1 = 4.0\t\n",
        "\tGTSZ P1 = 4.0\n",
        "GTSZ P1 = 4.0\r\n",
    ],
)
def test_round8_tabs_and_cr_are_rejected(tmp_path, body):
    with pytest.raises(ValueError, match="non-space whitespace"):
        _read(tmp_path, body)


def test_round8_tabs_in_comments_and_spaces_are_fine(tmp_path):
    g = _read(tmp_path, "   GTSZ P1 =   4.0   ! a\tcomment\n\t! tabbed comment\n")
    assert g.gtsz is not None and g.gtsz.p1 == 4.0


# ---------------------------------------------------------------------------
# #4 the coordinate type is case-sensitive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stype", ["sigmaz", "Sigmaz", "uniform", "SIGMA Z"])
def test_round8_noncanonical_type_is_rejected(tmp_path, stype):
    with pytest.raises(ValueError, match="case-sensitive"):
        _read(tmp_path, "", head=_HEAD.replace("SIGMAZ", stype) + "GTSZ NZ = 0\n")


# ---------------------------------------------------------------------------
# #5 scanner limits
# ---------------------------------------------------------------------------


def test_round8_key_beyond_the_header_search_is_rejected(tmp_path):
    head = _HEAD + "GTSZ NZ = 0\n" + "! filler\n" * 201
    with pytest.raises(ValueError, match="line 206.*201"):
        _read(tmp_path, "GTSZ P1 = 4.0\n", head=head)
    head = _HEAD + "GTSZ NZ = 0\n" + "! filler\n" * 197
    with pytest.raises(ValueError, match="line 202"):
        _read(tmp_path, "GTSZ P1 = 4.0\n", head=head)
    # line 201 is still read by FVCOM
    head = _HEAD + "GTSZ NZ = 0\n" + "! filler\n" * 196
    p = tmp_path / "s_sigma.dat"
    p.write_text(head + "GTSZ P1 = 4.0\n")
    assert len(p.read_text().splitlines()) == 201
    assert read_sigma_dat(p).gtsz.p1 == 4.0


def test_round8_overlong_line_and_list_are_rejected(tmp_path):
    head = _HEAD + "GTSZ NZ = 2\nGTSZ K1 = 2\nGTSZ K2 = 5\n"
    pad = " " * 400
    with pytest.raises(ValueError, match="400"):
        _read(tmp_path, f"GTSZ ZLEV = -2.0{pad}-4.0\n", head=head)
    # a long COMMENT line is harmless
    _read(tmp_path, "!" + "x" * 500 + "\nGTSZ ZLEV = -2.0 -4.0\n", head=head)
    many = " ".join(["1"] * 150) + " 1.0"
    with pytest.raises(ValueError, match="151 values"):
        _read(tmp_path, f"BPG REF ZLEV = {many}\n")


def test_round8_writer_respects_the_line_buffer(tmp_path):
    spec = GtszSpec(
        kb=91, k1=20, k2=91, nz=70, zlev=tuple(-float(i) for i in range(1, 71))
    )
    lines = gtsz_to_lines(spec)
    assert max(len(ln) for ln in lines) <= 400
    p = tmp_path / "w_sigma.dat"
    write_sigma_dat(p, spec)
    assert read_sigma_dat(p).gtsz == spec
    # short lists keep a decimal point on every value
    assert "GTSZ ZLEV = -2.0 -4.0" in gtsz_to_lines(
        GtszSpec(kb=31, k1=10, k2=13, nz=2, zlev=(-2.0, -4.0))
    )


def test_round8_writer_refuses_what_fvcom_cannot_read():
    long = GtszSpec(
        kb=200,
        k1=20,
        k2=160,
        nz=140,
        zlev=tuple(-1000.123456 - i for i in range(140)),
    )
    with pytest.raises(ValueError, match="400"):
        gtsz_to_lines(long)
    many = GtszSpec(
        kb=200, k1=20, k2=180, nz=160, zlev=tuple(-float(i) for i in range(1, 161))
    )
    with pytest.raises(ValueError, match="150"):
        gtsz_to_lines(many)
    with pytest.raises(ValueError, match="line 202"):
        gtsz_to_lines(GtszSpec(kb=31), header_comment="\n".join(["c"] * 200))


# ---------------------------------------------------------------------------
# #6 GTSZ startup constraints
# ---------------------------------------------------------------------------


def _triangle(h=10.0):
    return MeshInputs(
        x=np.array([0.0, 1000.0, 0.0]),
        y=np.array([0.0, 0.0, 1000.0]),
        nv=np.array([[0, 1, 2]]),
        h=np.full(3, h),
    )


def test_round8_mask_needs_a_zband():
    with pytest.raises(ValueError, match="requires a z-band"):
        GtszSpec(kb=5, mask=True).validate()
    with pytest.raises(ValueError, match="requires a z-band"):
        build_gtsz(_triangle(), GtszSpec(kb=5, mask=True))


def test_round8_masked_column_needs_two_active_layers():
    spec = GtszSpec(
        kb=5, base=1, k1=2, k2=5, nz=2, zlev=(-1.0, -2.0), mask=True, sadapt=True
    )
    with pytest.raises(ValueError, match="fewer than two"):
        build_gtsz(_triangle(), spec)


@pytest.mark.parametrize("n", [-1, 501])
def test_round8_bpg_ref_nz_range(tmp_path, n):
    with pytest.raises(ValueError, match="BPG REF NZ"):
        _read(tmp_path, f"BPG REF NZ = {n}\n")


# ---------------------------------------------------------------------------
# #7 one minimum: three interfaces / two layers
# ---------------------------------------------------------------------------


def test_round8_two_layers_accepted_one_layer_refused(tmp_path):
    GtszSpec(kb=3).validate()
    c = build_gtsz(_triangle(), GtszSpec(kb=3, base=1))
    np.testing.assert_allclose(c.Z[0], [0.0, -0.5, -1.0])
    with pytest.raises(ValueError, match="KB=2"):
        GtszSpec(kb=2).validate()
    with pytest.raises(ValueError, match="at least 3 levels"):
        _read(tmp_path, "", head=_HEAD.replace("= 31", "= 2") + "GTSZ NZ = 0\n")
    with pytest.raises(ValueError, match="siglay rows >= 2"):
        _validate_sigma([0.0, -1.0], [-0.5], 1)
    _validate_sigma([0.0, -0.5, -1.0], [-0.25, -0.75], 1)


# ---------------------------------------------------------------------------
# #8 the archive root does not follow the working directory
# ---------------------------------------------------------------------------


def test_round8_relative_root_survives_chdir(
    archives, tmp_path, monkeypatch
):  # noqa: F811
    from xfvcom.io import FullDomainSource

    _, _, root = archives
    monkeypatch.chdir(root.parent)
    src = FullDomainSource(root.name, "2020-01-01", "2020-01-01")
    assert src.root.is_absolute()
    assert src.name.startswith(f"{root.name}[")
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    out = src.read_2d("EGT", [3], [6])
    assert np.asarray(out).shape[0] == 24
    assert os.getcwd() == str(other)
