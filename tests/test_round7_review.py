# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Regression tests for review round 7 (gpt-6.1-sol, scratch/reviews/fulldomain/r7_final.md).

#1 --siglev-npz identity (node ids, depths), #2 sigma.dat value typing as in
FVCOM's SCAN_FILE, #3 LOWFRAC renormalisation, #4 GTSZ shape-parameter startup
checks, #6 unconverged grassfire sweep, #7 multi-valued start_index attribute.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from xfvcom.grid.connectivity import zero_based_connectivity
from xfvcom.grid.grid_obj import FvcomGrid
from xfvcom.grid.gtsz import GtszSpec, build_column_lowsig
from xfvcom.grid.gtsz_builder import grassfire_href
from xfvcom.io.sigma_dat import gtsz_to_lines, read_sigma_dat, write_sigma_dat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_jcope_obc as r  # noqa: E402  # isort: skip

# ---------------------------------------------------------------------------
# #1 --siglev-npz: node ids and recorded depths
# ---------------------------------------------------------------------------

_IDS = np.array([30, 20, 10])
_H = np.array([100.0, 200.0, 300.0])


def _npz(tmp_path, **arrays):
    p = tmp_path / "c.npz"
    np.savez(p, **arrays)
    return p


def _siglev(n=3, kb=5):
    base = np.linspace(0.0, -1.0, kb)
    # a different (identifiable) column per node
    return np.vstack([base ** (1 + i) * (-1) ** i for i in range(n)])


def test_round7_siglev_rows_are_reordered_by_node_id(tmp_path):
    z = _siglev()
    p = _npz(tmp_path, siglev=z, obc_node=_IDS, h=_H)
    out = r.load_siglev_npz(p, _IDS[::-1], _H[::-1])
    np.testing.assert_array_equal(out, z[::-1])
    np.testing.assert_array_equal(r.load_siglev_npz(p, _IDS, _H), z)


def test_round7_siglev_node_sets_must_agree(tmp_path):
    p = _npz(tmp_path, siglev=_siglev(), obc_node=_IDS, h=_H)
    with pytest.raises(ValueError, match="differs from the OBC list"):
        r.load_siglev_npz(p, np.array([30, 20, 11]), _H)
    p = _npz(tmp_path, siglev=_siglev(), obc_node=np.array([30, 30, 10]), h=_H)
    with pytest.raises(ValueError, match="duplicate"):
        r.load_siglev_npz(p, _IDS, _H)


def test_round7_siglev_depths_must_agree(tmp_path, capsys):
    p = _npz(tmp_path, siglev=_siglev(), obc_node=_IDS, h=_H)
    wrong = _H + np.array([0.0, 47.0, 0.0])
    with pytest.raises(ValueError, match="up to 47 m"):
        r.load_siglev_npz(p, _IDS, wrong)
    r.load_siglev_npz(p, _IDS, _H + 1e-5)  # within tolerance
    r.load_siglev_npz(p, _IDS, wrong, ignore_depth=True)
    assert "--siglev-npz-ignore-depth" in capsys.readouterr().out


def test_round7_siglev_without_identity_warns(tmp_path, capsys):
    z = _siglev()
    p = _npz(tmp_path, siglev=z)
    np.testing.assert_array_equal(r.load_siglev_npz(p, _IDS, _H), z)
    out = capsys.readouterr().out
    assert "identity" in out and "could NOT be verified" in out


_TB = Path.home() / "Github" / "TB-FVCOM"
_REAL_NPZ = _TB / "hydro/tuning/new_bc/obc_regen_lsc2/lsc2_obc_siglev_kb51.npz"
_GRID = _TB / "input/goto2023/grid"


@pytest.mark.skipif(not _REAL_NPZ.is_file(), reason="production npz not present")
def test_round7_siglev_production_file():
    ids = r.parse_obc_dat(_GRID / "TokyoBay_obc.dat")
    good = r.parse_dep_dat(_GRID / "TokyoBay_dep_m7001tp_rfac0p2_cap300.dat")
    z = r.load_siglev_npz(_REAL_NPZ, ids, good[ids - 1])
    assert z.shape == (13, 51)
    np.testing.assert_array_equal(
        r.load_siglev_npz(_REAL_NPZ, ids[::-1], good[ids[::-1] - 1]), z[::-1]
    )
    wrong = r.parse_dep_dat(_GRID / "TokyoBay_dep.dat")
    with pytest.raises(ValueError, match="347.18"):
        r.load_siglev_npz(_REAL_NPZ, ids, wrong[ids - 1])


# ---------------------------------------------------------------------------
# #2 sigma.dat values typed as FVCOM's SCAN_FILE types them
# ---------------------------------------------------------------------------

_HEAD = "NUMBER OF SIGMA LEVELS = 31\nSIGMA COORDINATE TYPE = SIGMAZ\nGTSZ BASE = 2\n"


def _read(tmp_path, body, head=_HEAD + "GTSZ NZ = 0\n"):
    p = tmp_path / "s_sigma.dat"
    p.write_text(head + body)
    return read_sigma_dat(p)


def test_round7_decimal_less_float_differing_from_default_is_rejected(tmp_path):
    with pytest.raises(ValueError, match=r"GTSZ P1 = 4.*default 2.0.*'4.0'"):
        _read(tmp_path, "GTSZ P1 = 4\n")
    with pytest.raises(ValueError, match="GTSZ SMOOTH"):
        _read(tmp_path, "GTSZ SMOOTH = 5\n")
    with pytest.raises(ValueError, match="GTSZ SMAX"):
        _read(tmp_path, "GTSZ SMAX = 1e-3\n")  # no '.' -> integer type in FVCOM


def test_round7_forms_fvcom_reads_as_written_are_accepted(tmp_path):
    g = _read(tmp_path, "GTSZ P1 = 4.0\nGTSZ L1 = 1\nGTSZ SMAX = 1.0e-3\n").gtsz
    assert g.p1 == 4.0 and g.l1 == 1.0 and g.smax == 1.0e-3
    # decimal-less value equal to FVCOM's fallback: same meaning to both readers
    assert _read(tmp_path, "GTSZ P1 = 2\n").gtsz.p1 == 2.0
    # FVCOM re-scans a decimal-less DREF as an integer
    assert _read(tmp_path, "GTSZ DREF = 30\n").gtsz.dref == 30.0


@pytest.mark.parametrize(
    "body",
    [
        "GTSZ MASK = TRUE\n",  # FVCOM: string -> MASK silently F
        "GTSZ K1 = 10.0\n",  # integer key with a decimal point
        "gtsz p1 = 4.0\n",  # FVCOM matches keys exactly -> ignored
        "GTSZ P1 = 4.0\nGTSZ P1 = 3.0\n",  # FVCOM takes the first
    ],
)
def test_round7_other_forms_fvcom_reads_differently_are_rejected(tmp_path, body):
    with pytest.raises(ValueError):
        _read(tmp_path, body)


def test_round7_zlev_needs_a_decimal_point(tmp_path):
    head = _HEAD + "GTSZ NZ = 2\nGTSZ K1 = 2\nGTSZ K2 = 5\n"
    with pytest.raises(ValueError, match="GTSZ ZLEV"):
        _read(tmp_path, "GTSZ ZLEV = -2 -4\n", head=head)
    assert _read(tmp_path, "GTSZ ZLEV = -2 -4.0\n", head=head).gtsz.zlev == (
        -2.0,
        -4.0,
    )
    with pytest.raises(ValueError, match="NUMBER OF SIGMA LEVELS = 31.0"):
        _read(tmp_path, "", head=_HEAD.replace("= 31", "= 31.0") + "GTSZ NZ = 0\n")


def test_round7_writer_always_emits_decimal_points(tmp_path):
    spec = GtszSpec(
        kb=31, k1=10, k2=13, nz=2, zlev=(-2.0, -4.0), p1=4.0, l1=3.0, smooth=5.0
    )
    lines = gtsz_to_lines(spec)
    assert "GTSZ P1 = 4.0" in lines
    assert "GTSZ L1 = 3.0" in lines
    assert "GTSZ SMOOTH = 5.0" in lines
    assert "GTSZ ZLEV = -2.0 -4.0" in lines
    p = tmp_path / "w_sigma.dat"
    write_sigma_dat(p, spec)
    assert read_sigma_dat(p).gtsz == spec
    tiny = GtszSpec(kb=31, p1=1.0e-5)
    assert "GTSZ P1 = 1.0e-05" in gtsz_to_lines(tiny)


# ---------------------------------------------------------------------------
# #3 LOWFRAC is renormalised before use (mod_input.F)
# ---------------------------------------------------------------------------


def test_round7_lowfrac_is_renormalised():
    frac = (0.33331, 0.33331, 0.33331)
    spec = GtszSpec(kb=5, base=1, k1=2, sadapt=True, lowsig=True, lowfrac=frac)
    spec.validate()
    z = build_column_lowsig(300.0, 20.0, spec)
    exact = build_column_lowsig(300.0, 20.0, spec.with_(lowfrac=(1 / 3,) * 3))
    np.testing.assert_allclose(z, exact, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# #4 FVCOM's GTSZ startup checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        {"p1": 0.0},
        {"p1": -1.0},
        {"p1": float("nan")},
        {"p1": float("inf")},
        {"smooth": -1.0},
        {"smooth": float("nan")},
        {"smooth": float("inf")},
        {"base": 3, "l1": float("inf")},
        {"base": 3, "l1": 1.0, "l2": -1.0},
        {"sadapt": True, "lowsig": True, "k1": 1},
    ],
)
def test_round7_model_invalid_parameters_are_refused(kw):
    with pytest.raises(ValueError):
        GtszSpec(kb=31, **kw).validate()


def test_round7_valid_parameters_still_pass():
    GtszSpec(kb=31, p1=0.5, smooth=0.0).validate()
    GtszSpec(kb=31, base=1, p1=0.0).validate()  # P1 unused unless BASE=2
    GtszSpec(kb=31, base=3, l1=2.0, l2=0.5).validate()
    GtszSpec(kb=31, sadapt=True, lowsig=True, k1=2).validate()


# ---------------------------------------------------------------------------
# #6 an exhausted sweep budget is an error
# ---------------------------------------------------------------------------


def _strip(n=20):
    x = np.repeat(np.arange(n, dtype=float), 2) * 1000.0
    y = np.tile([0.0, 1000.0], n)
    tri = []
    for i in range(n - 1):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        tri += [[a, c, b], [b, c, d]]
    h = np.full(2 * n, 300.0)
    h[:2] = 10.0  # shallow end: the front must cross the whole strip
    return h, np.array(tri), x, y


def test_round7_unconverged_grassfire_raises():
    h, nv, x, y = _strip()
    with pytest.raises(RuntimeError, match="did not converge"):
        grassfire_href(h, nv, x, y, 0.01, max_sweep=1)
    href, sweeps = grassfire_href(h, nv, x, y, 0.01, return_sweeps=True)
    assert sweeps > 1
    assert np.all(href <= h)


# ---------------------------------------------------------------------------
# #7 the start_index attribute must hold exactly one value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attr", [np.array([0, 1]), np.array([], dtype=int)])
def test_round7_start_index_must_be_single_valued(attr):
    da = xr.DataArray(np.array([[1], [2], [3]]), dims=("three", "nele"))
    da.attrs["start_index"] = attr
    with pytest.raises(ValueError, match="exactly one value"):
        zero_based_connectivity(da)
    ds = xr.Dataset(
        {
            "x": ("node", np.array([0.0, 1.0, 0.0, 5.0])),
            "y": ("node", np.array([0.0, 0.0, 1.0, 1.0])),
            "nv": da,
        }
    )
    with pytest.raises(ValueError, match="exactly one value"):
        FvcomGrid.from_dataset(ds)


@pytest.mark.parametrize("attr", [0, np.int32(0), np.array([0]), np.array(0)])
def test_round7_single_valued_start_index_still_works(attr):
    da = xr.DataArray(np.array([[1], [2], [3]]), dims=("three", "nele"))
    da.attrs["start_index"] = attr
    assert zero_based_connectivity(da)[:, 0].tolist() == [1, 2, 3]
