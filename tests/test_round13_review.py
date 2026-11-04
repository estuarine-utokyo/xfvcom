# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Regression tests for review round 13 (gpt-6.1-sol, scratch/reviews/fulldomain/r13_final.md).

#1 substituted_obc_depths offers an explicit endpoint rule: "observed" (default,
   endpoints keep their own depth) or "source" (mod_obcs.F literally),
#2 sigma.dat values are converted with Fortran's real-literal syntax
   (implicit exponents such as '1.0-3', D exponents, signs) by ONE helper shared
   by the numeric-line validator and the active readers.
"""

from __future__ import annotations

import re
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

from xfvcom.io.jcope_obc_generator import write_tsobc_nc
from xfvcom.io.obc_depth_control import next_obc_nodes, substituted_obc_depths
from xfvcom.io.sigma_dat import fortran_real, read_sigma_dat

_HEAD = "NUMBER OF SIGMA LEVELS = 5\nSIGMA COORDINATE TYPE = SIGMAZ\nGTSZ BASE = 1\n"


# ---------------------------------------------------------------------------
# #2 Fortran real literals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tok, x",
    [
        ("1.0-3", 1.0e-3),
        ("1.0+3", 1.0e3),
        ("-1.0+2", -100.0),
        (".5-1", 0.05),
        ("1.-3", 1.0e-3),
        ("1-3", 1.0e-3),
        ("1.0d-3", 1.0e-3),
        ("1.0D+3", 1.0e3),
        ("1e5", 1.0e5),
        ("1.0E-3", 1.0e-3),
        ("+.5", 0.5),
        ("5.", 5.0),
        ("0.999", 0.999),
        ("1.0e-400", 0.0),  # underflow reads as zero, as in Fortran
    ],
)
def test_round13_fortran_real_accepts(tok, x):
    assert fortran_real(tok) == x


@pytest.mark.parametrize(
    "tok",
    [
        "1.0.0",
        "abc",
        "1e",
        "1.0e+",
        ".",
        "-.",
        ".e5",
        "e5",
        "--1",
        "1.0-+3",
        "1.0e3.0",
        "1+",
        "1.0e-3-2",
        "1.0e999",
        "1.0+400",
    ],
)
def test_round13_fortran_real_rejects(tok):
    with pytest.raises(ValueError):
        fortran_real(tok)


def test_round13_reviewer_lowfrac_file_reads(tmp_path):
    p = tmp_path / "s_sigma.dat"
    p.write_text(_HEAD + "GTSZ NZ = 0\nGTSZ LOWFRAC = 1.0-3 0.999\n")
    s = read_sigma_dat(p)
    assert s.gtsz is not None and s.gtsz.lowfrac == ()  # inactive: LOWSIG unset


def test_round13_active_lowfrac_reads_implicit_exponent(tmp_path):
    p = tmp_path / "s_sigma.dat"
    p.write_text(
        _HEAD + "GTSZ K1 = 2\nGTSZ NZ = 0\nGTSZ LOWSIG = T\n"
        "GTSZ LOWFRAC = 1.0-3 0.999 1.0+0\n"
    )
    s = read_sigma_dat(p)
    assert s.gtsz is not None and s.gtsz.lowfrac == (1.0e-3, 0.999, 1.0)


@pytest.mark.parametrize(
    "lit, x", [("8.0-4", 8.0e-4), ("1.0+3", 1.0e3), ("8.0e-4", 8.0e-4)]
)
def test_round13_active_float_key_reads_implicit_exponent(tmp_path, lit, x):
    p = tmp_path / "s_sigma.dat"
    p.write_text(_HEAD + f"GTSZ NZ = 0\nGTSZ SMAX = {lit}\n")
    s = read_sigma_dat(p)
    assert s.gtsz is not None and s.gtsz.smax == x


def test_round13_zlev_reads_implicit_exponent(tmp_path):
    p = tmp_path / "s_sigma.dat"
    p.write_text(
        _HEAD + "GTSZ K1 = 1\nGTSZ K2 = 5\nGTSZ NZ = 2\nGTSZ ZLEV = -2.0+0 -4.0+0\n"
    )
    s = read_sigma_dat(p)
    assert s.gtsz is not None and s.gtsz.zlev == (-2.0, -4.0)


@pytest.mark.parametrize(
    "line, tok, why",
    [
        ("GTSZ LOWFRAC = 0.5 1.0.0", "1.0.0", "not a valid floating"),
        ("GTSZ LOWFRAC = 0.5 1e", "1e", "not a valid floating"),
        ("GTSZ LOWFRAC = 0.5 1.0e+", "1.0e+", "not a valid floating"),
        ("GTSZ LOWFRAC = 0.5 1.0e999", "1.0e999", "not finite"),
        ("GTSZ LOWFRAC = 0.5 1.0+999", "1.0+999", "not finite"),
    ],
)
def test_round13_invalid_tokens_still_refused(tmp_path, line, tok, why):
    p = tmp_path / "s_sigma.dat"
    p.write_text(_HEAD + "GTSZ NZ = 0\n" + line + "\n")
    with pytest.raises(ValueError, match=f"'{re.escape(tok)}'.*{why}"):
        read_sigma_dat(p)


def test_round13_letters_make_a_string_value(tmp_path):
    # 'abc' and 'd' are outside FVCOM's numeric characters: the value is a
    # string, never converted on an unused line, and refused for an active key
    p = tmp_path / "s_sigma.dat"
    p.write_text(_HEAD + "GTSZ NZ = 0\nOTHER = abc\nGTSZ ZLEV = 1.0d-3 2.0\n")
    assert read_sigma_dat(p).kb == 5
    p.write_text(_HEAD + "GTSZ NZ = 0\nGTSZ SMAX = 1.0d-3\n")
    with pytest.raises(ValueError, match="GTSZ SMAX"):
        read_sigma_dat(p)


def test_round13_integer_hint_for_implicit_exponent(tmp_path):
    p = tmp_path / "s_sigma.dat"
    p.write_text(_HEAD + "GTSZ NZ = 0\nGTSZ LOWFRAC = 5 1-2\n")
    with pytest.raises(ValueError, match=r"not a valid integer.*'1\.0-2'"):
        read_sigma_dat(p)


# ---------------------------------------------------------------------------
# #1 endpoint rule (strip fixture of tests/test_obc_depth_control.py)
# ---------------------------------------------------------------------------
X = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
Y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
NV = np.array([[0, 1, 1, 2], [1, 4, 2, 5], [3, 3, 4, 4]])
OBC_IDS = np.array([4, 5, 6])
H = np.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0])


def test_round13_observed_rule_is_default():
    h_sub, nxt = substituted_obc_depths(X, Y, NV, OBC_IDS, H)
    np.testing.assert_array_equal(h_sub, [1.0, 20.0, 3.0])
    np.testing.assert_array_equal(nxt, [4, 2, 6])
    h_obs, _ = substituted_obc_depths(X, Y, NV, OBC_IDS, H, endpoint_rule="observed")
    np.testing.assert_array_equal(h_obs, h_sub)


def test_round13_source_rule_substitutes_endpoints():
    h_sub, nxt = substituted_obc_depths(X, Y, NV, OBC_IDS, H, endpoint_rule="source")
    np.testing.assert_array_equal(h_sub, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(nxt, [1, 2, 3])
    np.testing.assert_array_equal(
        next_obc_nodes(X, Y, NV, OBC_IDS, endpoint_rule="source"), [1, 2, 3]
    )


def test_round13_unknown_endpoint_rule_refused():
    with pytest.raises(ValueError, match="endpoint_rule"):
        substituted_obc_depths(X, Y, NV, OBC_IDS, H, endpoint_rule="corner")


def test_round13_history_note_is_recorded(tmp_path: Path):
    out = tmp_path / "ts.nc"
    siglev = np.array([0.0, -0.5, -1.0])
    siglay = np.array([-0.25, -0.75])
    fields: dict[str, np.ndarray] = {"temp": np.full((2, 2, 3), 10.0, np.float32)}
    write_tsobc_nc(
        out,
        time_mjd=np.array([58849.0, 58849.5]),
        obc_nodes=np.array([4, 5, 6], np.int32),
        obc_h=np.array([10.0, 20.0, 30.0], np.float32),
        siglev=siglev,
        siglay=siglay,
        fields=fields,
        source_files=["a.nc"],
        history_note="endpoint rule 'source'",
    )
    with nc.Dataset(out) as ds:
        assert ds.history.endswith("from a.nc; endpoint rule 'source'")
