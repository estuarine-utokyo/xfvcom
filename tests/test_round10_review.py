# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Regression tests for review round 10 (gpt-6.1-sol, scratch/reviews/fulldomain/r10_final.md).

#1 continuation suffixes exposed by FVCOM's 400-character line truncation,
#2 LOWSIG startup checks in the model's precision (``model_precision``),
#3 micrometre rounding of suggested z-levels never breaks the ZLEV contract,
#4 non-finite numeric values (overflowing literals) are rejected,
#5 ``python -m scripts.compare_jcope_sources --help`` works from the repo root.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from xfvcom.grid.gtsz import GtszSpec
from xfvcom.grid.gtsz_builder import MeshInputs, build_gtsz, suggest_zlev
from xfvcom.io.sigma_dat import gtsz_to_lines, read_sigma_dat, write_sigma_dat

_REPO = Path(__file__).resolve().parents[1]
_HEAD = (
    "NUMBER OF SIGMA LEVELS = 5\nSIGMA COORDINATE TYPE = SIGMAZ\nGTSZ BASE = 2\n"
    "GTSZ NZ = 0\n"
)


def _tri(h):
    return MeshInputs(
        x=np.array([0.0, 1000.0, 0.0]),
        y=np.array([0.0, 0.0, 1000.0]),
        nv=np.array([[0, 1, 2]], dtype=np.int64),
        h=np.asarray(h, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# #1 truncation to the 400-character buffer exposes a continuation suffix
# ---------------------------------------------------------------------------


def test_round10_truncated_comment_continuation_is_refused(tmp_path):
    comment = "! " + "x" * 394 + "\\" * 4 + "tail"
    p = tmp_path / "s_sigma.dat"
    p.write_text(_HEAD + comment + "\nGTSZ P1 = 4.0\n")
    with pytest.raises(ValueError, match="continuation"):
        read_sigma_dat(p)


def test_round10_long_comment_without_exposed_suffix_still_reads(tmp_path):
    comment = "! " + "x" * 500
    p = tmp_path / "s_sigma.dat"
    p.write_text(_HEAD + comment + "\nGTSZ P1 = 4.0\n")
    assert read_sigma_dat(p).gtsz.p1 == 4.0


@pytest.mark.parametrize(
    "text", ["x" * 394 + "\\" * 4 + "tail", "y" * 399]  # exposed suffix / plain long
)
def test_round10_writer_refuses_overlong_header_comment(text):
    with pytest.raises(ValueError, match="400"):
        gtsz_to_lines(GtszSpec(kb=5, base=1), header_comment=text)


def test_round10_writer_accepts_comment_at_the_limit(tmp_path):
    p = tmp_path / "s_sigma.dat"
    write_sigma_dat(p, GtszSpec(kb=5, base=1), header_comment="z" * 398)
    assert read_sigma_dat(p).kb == 5


# ---------------------------------------------------------------------------
# #2 LOWSIG startup checks in the model precision
# ---------------------------------------------------------------------------

_COLLAPSE = GtszSpec(
    kb=5, base=1, k1=3, sadapt=True, lowsig=True, lowfrac=(0.99999999, 1e-8)
)


def test_round10_single_precision_collapse_is_refused():
    with pytest.raises(ValueError, match="single precision"):
        build_gtsz(_tri([10.0] * 3), _COLLAPSE, model_precision="single")


def test_round10_double_precision_is_unchanged():
    c = build_gtsz(_tri([10.0] * 3), _COLLAPSE)  # default: double
    dz = (c.Z[:, :-1] - c.Z[:, 1:]) * c.H[:, None]
    assert dz.min() > 0.0
    c2 = build_gtsz(_tri([10.0] * 3), _COLLAPSE, model_precision="double")
    assert np.array_equal(c.Z, c2.Z)


def test_round10_single_precision_accepts_regular_lowsig():
    spec = GtszSpec(
        kb=8,
        base=2,
        k1=4,
        sadapt=True,
        lowsig=True,
        smax=0.05,
        smooth=3.0,
        dref=20.0,
        lowfrac=(0.1, 0.2, 0.3, 0.4),
    )
    a = build_gtsz(_tri([10.0, 80.0, 300.0]), spec, model_precision="single")
    b = build_gtsz(_tri([10.0, 80.0, 300.0]), spec)
    assert np.array_equal(a.Z, b.Z)  # the returned coordinate is float64 either way


def test_round10_unknown_precision_is_refused():
    with pytest.raises(ValueError, match="model_precision"):
        build_gtsz(_tri([10.0] * 3), _COLLAPSE, model_precision="quad")


# ---------------------------------------------------------------------------
# #3 suggested z-levels stay negative and strictly descending
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        dict(h_max=3.0, z_top=-4e-7),
        dict(h_max=4.0, dz_shallow=2e-7, stretch=2.0),
        dict(h_max=300.0),
    ],
)
def test_round10_suggested_levels_keep_the_contract(kw):
    z = suggest_zlev(**kw)
    assert (z < 0.0).all() and (np.diff(z) < 0.0).all()
    GtszSpec(kb=len(z) + 3, k1=2, nz=len(z), zlev=tuple(z)).validate()


def test_round10_regular_levels_are_still_rounded():
    z = suggest_zlev(300.0)
    assert np.array_equal(z, np.round(z, 6))


# ---------------------------------------------------------------------------
# #4 non-finite numeric values
# ---------------------------------------------------------------------------


def test_round10_overflowing_zlev_is_refused(tmp_path):
    p = tmp_path / "s_sigma.dat"
    p.write_text(
        "NUMBER OF SIGMA LEVELS = 6\nSIGMA COORDINATE TYPE = SIGMAZ\nGTSZ BASE = 1\n"
        "GTSZ K1 = 3\nGTSZ K2 = 6\nGTSZ NZ = 2\nGTSZ MASK = T\nGTSZ SADAPT = T\n"
        "GTSZ ZLEV = -1.0e999 -2.0e999\n"
    )
    with pytest.raises(ValueError, match="GTSZ ZLEV.*not finite"):
        read_sigma_dat(p)


def test_round10_overflowing_scalar_is_refused(tmp_path):
    p = tmp_path / "s_sigma.dat"
    p.write_text(_HEAD + "GTSZ P1 = 1.0e999\n")
    with pytest.raises(ValueError, match="GTSZ P1.*not finite"):
        read_sigma_dat(p)


@pytest.mark.parametrize(
    "zlev", [(-np.inf, -np.inf), (-1.0, np.nan), (-np.inf, -2.0), (-1.0, -np.inf)]
)
def test_round10_spec_requires_finite_zlev(zlev):
    spec = GtszSpec(kb=6, base=1, k1=3, k2=6, nz=2, zlev=zlev, mask=True, sadapt=True)
    with pytest.raises(ValueError, match="finite"):
        spec.validate()


def test_round10_spec_requires_finite_lowfrac():
    spec = GtszSpec(kb=5, base=1, k1=3, sadapt=True, lowsig=True, lowfrac=(np.nan, 0.5))
    with pytest.raises(ValueError, match="finite"):
        spec.validate()


# ---------------------------------------------------------------------------
# #5 module invocation of the comparison script
# ---------------------------------------------------------------------------


def test_round10_compare_script_runs_as_module():
    r = subprocess.run(
        [sys.executable, "-m", "scripts.compare_jcope_sources", "--help"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr
    assert "--archive-root" in r.stdout
