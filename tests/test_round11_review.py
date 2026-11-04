# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Regression tests for review round 11 (gpt-6.1-sol, scratch/reviews/fulldomain/r11_final.md).

#1 the single-precision LOWSIG startup check evaluates SBAND (A3 shape,
   subtraction, normalisation) in REAL(4), catching an upper-band collapse,
#2 integer values outside FVCOM's 32-bit INTEGER range are rejected by the
   reader (also an integer-form GTSZ DREF), by GtszSpec.validate() and the writer,
#3 is documentation only (xfvcom.grid.gtsz module docstring).
"""

from __future__ import annotations

import numpy as np
import pytest

from xfvcom.grid.gtsz import GtszSpec
from xfvcom.grid.gtsz_builder import MeshInputs, build_gtsz
from xfvcom.io.sigma_dat import gtsz_to_lines, read_sigma_dat

_HEAD = "NUMBER OF SIGMA LEVELS = 5\nSIGMA COORDINATE TYPE = SIGMAZ\nGTSZ BASE = 1\n"


def _tri(h):
    return MeshInputs(
        x=np.array([0.0, 1000.0, 0.0]),
        y=np.array([0.0, 0.0, 1000.0]),
        nv=np.array([[0, 1, 2]], dtype=np.int64),
        h=np.asarray(h, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# #1 upper-band (SBAND) collapse in single precision
# ---------------------------------------------------------------------------
_SATURATE = GtszSpec(kb=31, base=3, k1=10, sadapt=True, lowsig=True, l1=10.0, l2=10.0)


def test_round11_single_precision_sband_saturation_is_refused():
    with pytest.raises(ValueError, match="single precision"):
        build_gtsz(_tri([10.0] * 3), _SATURATE, model_precision="single")


def test_round11_sband_saturation_still_builds_in_double():
    c = build_gtsz(_tri([10.0] * 3), _SATURATE, model_precision="double")
    dz = (c.Z[:, :-1] - c.Z[:, 1:]) * c.H[:, None]
    assert (dz > 0.0).all()


@pytest.mark.parametrize("base", [1, 2, 3])
def test_round11_ordinary_lowsig_builds_in_single(base):
    spec = GtszSpec(kb=31, base=base, k1=10, sadapt=True, lowsig=True)
    a = build_gtsz(_tri([10.0, 80.0, 300.0]), spec, model_precision="single")
    b = build_gtsz(_tri([10.0, 80.0, 300.0]), spec, model_precision="double")
    np.testing.assert_array_equal(a.Z, b.Z)


# ---------------------------------------------------------------------------
# #2 32-bit INTEGER range
# ---------------------------------------------------------------------------
def test_round11_reader_refuses_int32_overflow(tmp_path):
    p = tmp_path / "s_sigma.dat"
    p.write_text(_HEAD + "GTSZ NZ = 0\nGTSZ K1 = 2147483648\n")
    with pytest.raises(ValueError, match="GTSZ K1.*32-bit INTEGER"):
        read_sigma_dat(p)


def test_round11_reader_accepts_int32_max(tmp_path):
    p = tmp_path / "s_sigma.dat"
    p.write_text(_HEAD + "GTSZ NZ = 0\nGTSZ K1 = 2147483647\n")
    assert read_sigma_dat(p).gtsz.k1 == 2147483647


def test_round11_reader_refuses_int32_underflow(tmp_path):
    p = tmp_path / "s_sigma.dat"
    p.write_text(_HEAD + "GTSZ NZ = 0\nGTSZ K2 = -2147483649\n")
    with pytest.raises(ValueError, match="GTSZ K2.*32-bit INTEGER"):
        read_sigma_dat(p)


def test_round11_reader_refuses_integer_form_dref_overflow(tmp_path):
    p = tmp_path / "s_sigma.dat"
    p.write_text(_HEAD + "GTSZ NZ = 0\nGTSZ DREF = 99999999999\n")
    with pytest.raises(ValueError, match="GTSZ DREF.*32-bit INTEGER"):
        read_sigma_dat(p)


def test_round11_reader_keeps_integer_form_dref(tmp_path):
    p = tmp_path / "s_sigma.dat"
    p.write_text(_HEAD + "GTSZ NZ = 0\nGTSZ DREF = 30\n")
    assert read_sigma_dat(p).gtsz.dref == 30.0


def test_round11_spec_and_writer_refuse_int32_overflow():
    spec = GtszSpec(kb=5, base=1, k1=2147483648)
    with pytest.raises(ValueError, match="GTSZ K1.*32-bit INTEGER"):
        spec.validate()
    with pytest.raises(ValueError, match="GTSZ K1.*32-bit INTEGER"):
        gtsz_to_lines(spec)
