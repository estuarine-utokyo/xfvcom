# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Regression tests for review round 12 (gpt-6.1-sol, scratch/reviews/fulldomain/r12_final.md).

#1 the single-precision LOWSIG check uses the C library's tanhf/powf and refuses
   upper-band separations within a few float32 ulps (a runtime/compiler can
   collapse them); double builds are unchanged,
#2 every numeric key/value line FVCOM's SCAN_FILE passes is converted (32-bit
   INTEGER or REAL), so an unreadable or out-of-range token is rejected even
   when its key is inactive,
#3 read_dat accepts 3-field node and 4-field element records,
#5 the decimal-point hint for an exponent literal is valid ('8.0e-4').
"""

from __future__ import annotations

import numpy as np
import pytest

from xfvcom.grid import gtsz_builder
from xfvcom.grid.dat_reader import read_dat
from xfvcom.grid.gtsz import GtszSpec
from xfvcom.grid.gtsz_builder import MeshInputs, build_gtsz
from xfvcom.io.sigma_dat import read_sigma_dat

_HEAD = "NUMBER OF SIGMA LEVELS = 5\nSIGMA COORDINATE TYPE = SIGMAZ\nGTSZ BASE = 1\n"


def _tri(h):
    return MeshInputs(
        x=np.array([0.0, 1000.0, 0.0]),
        y=np.array([0.0, 0.0, 1000.0]),
        nv=np.array([[0, 1, 2]], dtype=np.int64),
        h=np.asarray(h, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# #1 single-precision SBAND near saturation
# ---------------------------------------------------------------------------
_NEAR = GtszSpec(kb=31, base=3, k1=10, sadapt=True, lowsig=True, l1=0.1, l2=8.8)
_SATURATE = GtszSpec(kb=31, base=3, k1=10, sadapt=True, lowsig=True, l1=10.0, l2=10.0)


@pytest.mark.parametrize("spec", [_NEAR, _SATURATE])
def test_round12_near_saturated_sband_is_refused_in_single(spec):
    with pytest.raises(ValueError, match="single precision"):
        build_gtsz(_tri([10.0] * 3), spec, model_precision="single")


@pytest.mark.parametrize("spec", [_NEAR, _SATURATE])
def test_round12_near_saturated_sband_is_refused_without_libm(spec, monkeypatch):
    monkeypatch.setattr(gtsz_builder, "_LIBM_F32", None)
    with pytest.raises(ValueError, match="single precision"):
        build_gtsz(_tri([10.0] * 3), spec, model_precision="single")


@pytest.mark.parametrize("spec", [_NEAR, _SATURATE])
def test_round12_near_saturated_sband_builds_in_double(spec):
    c = build_gtsz(_tri([10.0] * 3), spec, model_precision="double")
    dz = (c.Z[:, :-1] - c.Z[:, 1:]) * c.H[:, None]
    assert (dz > 0.0).all()


def test_round12_libm_tanhf_collapses_the_reviewer_case():
    """The reason for the guard: with the C library's tanhf the second
    interface of the reviewer's case coincides with the surface."""
    if gtsz_builder._LIBM_F32 is None:
        pytest.skip("no C math library through ctypes")
    f32 = np.float32
    t = gtsz_builder._tanh_sp
    arg = (f32(-1) * f32(0.1) + f32(29) * f32(8.8)) / f32(30)
    assert t(arg, f32) == t(f32(8.8), f32)


@pytest.mark.parametrize(
    "kw",
    [
        dict(base=1),
        dict(base=2),
        dict(base=2, p1=3.0),
        dict(base=3),
        dict(base=3, l1=2.0, l2=3.0),
        dict(base=1, smooth=2.0),
    ],
)
def test_round12_ordinary_lowsig_builds_in_both_precisions(kw):
    spec = GtszSpec(kb=31, k1=10, sadapt=True, lowsig=True, **kw)
    a = build_gtsz(_tri([10.0, 80.0, 300.0]), spec, model_precision="single")
    b = build_gtsz(_tri([10.0, 80.0, 300.0]), spec, model_precision="double")
    np.testing.assert_array_equal(a.Z, b.Z)


# ---------------------------------------------------------------------------
# #2 inactive numeric lines are still converted by FVCOM
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "line, key, tok",
    [
        ("GTSZ ZLEV = -2147483649 -2147483650", "GTSZ ZLEV", "-2147483649"),
        ("GTSZ LOWFRAC = 99999999999", "GTSZ LOWFRAC", "99999999999"),
        ("SOME OTHER KEY = 3000000000", "SOME OTHER KEY", "3000000000"),
    ],
)
def test_round12_inactive_int32_overflow_is_refused(tmp_path, line, key, tok):
    p = tmp_path / "s_sigma.dat"
    p.write_text(_HEAD + "GTSZ NZ = 0\n" + line + "\n")
    with pytest.raises(ValueError, match=f"{key}.*'{tok}'.*32-bit INTEGER"):
        read_sigma_dat(p)


@pytest.mark.parametrize(
    "line, key, tok, why",
    [
        ("GTSZ ZLEV = -2.0 1.0e999", "GTSZ ZLEV", "1.0e999", "not finite"),
        ("GTSZ LOWFRAC = 0.5 1-2.", "GTSZ LOWFRAC", "1-2.", "not a valid floating"),
        ("GTSZ LOWFRAC = 5 1-2", "GTSZ LOWFRAC", "1-2", "not a valid integer"),
    ],
)
def test_round12_inactive_unreadable_token_is_refused(tmp_path, line, key, tok, why):
    p = tmp_path / "s_sigma.dat"
    p.write_text(_HEAD + "GTSZ NZ = 0\n" + line + "\n")
    with pytest.raises(ValueError, match=f"{key}.*'{tok}'.*{why}"):
        read_sigma_dat(p)


def test_round12_inactive_valid_lists_still_read(tmp_path):
    p = tmp_path / "s_sigma.dat"
    p.write_text(
        _HEAD + "GTSZ NZ = 0\nGTSZ ZLEV = -2 -4\nGTSZ LOWFRAC = 0.5 0.5\n"
        "OTHER = 2147483647 -2147483648\n"
    )
    s = read_sigma_dat(p)
    assert s.gtsz is not None and s.gtsz.zlev == () and s.gtsz.lowfrac == ()


def test_round12_lines_beyond_the_header_search_are_not_converted(tmp_path):
    p = tmp_path / "s_sigma.dat"
    pad = "".join(f"! comment {i}\n" for i in range(210))
    p.write_text(_HEAD + "GTSZ NZ = 0\n" + pad + "OTHER = 99999999999\n")
    assert read_sigma_dat(p).kb == 5


# ---------------------------------------------------------------------------
# #3 minimal node / element records
# ---------------------------------------------------------------------------
def _grd(tmp_path, elem, node):
    p = tmp_path / "m_grd.dat"
    p.write_text("Node Number = 3\nCell Number = 1\n" + elem + node)
    return p


@pytest.mark.parametrize("elem", ["1 1 2 3\n", "1 1 2 3 1\n", "1 1 2 3 1 9\n"])
@pytest.mark.parametrize(
    "node",
    [
        "1 400000.0 3900000.0\n2 401000.0 3900000.0\n3 400000.0 3901000.0\n",
        "1 400000.0 3900000.0 0.0\n2 401000.0 3900000.0 0.0\n"
        "3 400000.0 3901000.0 0.0\n",
        "1 400000.0 3900000.0 0.0 7\n2 401000.0 3900000.0 0.0 7\n"
        "3 400000.0 3901000.0 0.0 7\n",
    ],
)
def test_round12_read_dat_minimal_and_trailing_columns(tmp_path, elem, node):
    g = read_dat(_grd(tmp_path, elem, node))
    np.testing.assert_array_equal(g["nv"][:, 0], [0, 1, 2])
    np.testing.assert_array_equal(g["x"], [400000.0, 401000.0, 400000.0])
    np.testing.assert_array_equal(g["y"], [3900000.0, 3900000.0, 3901000.0])


def test_round12_read_dat_keeps_order_validation(tmp_path):
    node = "2 400000.0 3900000.0\n1 401000.0 3900000.0\n3 400000.0 3901000.0\n"
    with pytest.raises(ValueError, match="node record 1 carries id 2"):
        read_dat(_grd(tmp_path, "1 1 2 3\n", node))


# ---------------------------------------------------------------------------
# #5 decimal-point hint for an exponent literal
# ---------------------------------------------------------------------------
def test_round12_exponent_hint_is_a_valid_literal(tmp_path):
    p = tmp_path / "s_sigma.dat"
    p.write_text(_HEAD + "GTSZ NZ = 0\nGTSZ SMAX = 8e-4\n")
    with pytest.raises(ValueError, match=r"'8\.0e-4'") as ei:
        read_sigma_dat(p)
    assert "8e-4.0" not in str(ei.value)


def test_round12_get_float_hint_inserts_point_in_mantissa():
    from xfvcom.io.sigma_dat import _with_point

    assert _with_point("8") == "8.0"
    assert _with_point("8e-4") == "8.0e-4"
    assert _with_point("-3E+2") == "-3.0E+2"
    assert float(_with_point("8e-4")) == 8e-4
