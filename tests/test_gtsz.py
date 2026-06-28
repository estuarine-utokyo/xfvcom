"""Tests for the FVCOM sigma-z (GTSZ) coordinate generator (:mod:`xfvcom.grid.gtsz`,
``gtsz_builder``, ``gtsz_diagnostics``, ``xfvcom.io.sigma_dat``, ``xfvcom.io.dep_reader``).

The base / B-sandwich column builder is regression-tested against an inline copy
of the FVCOM reference ``Tests/MultiSigma/bu_shallow/check_coord.py::sigma_gtsz``
(hard-max + monotonic clamp). The SADAPT path is checked by its invariants
(monotone, fixed-depth z-levels in the deep, pure-sigma on gentle columns). The
grassfire H_ref is checked against the Lipschitz property + the flat-min(H)
branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from xfvcom.grid.gtsz import (
    DZ_MIN_KBP,
    GtszSpec,
    base_sigma,
    build_coordinate,
    compute_kbp,
    compute_kbp1_frac1,
    sband_reshape,
)
from xfvcom.grid.gtsz_builder import (
    MeshInputs,
    active_zlev_count,
    auto_kb,
    build_gtsz,
    grassfire_href,
    suggest_zlev,
)
from xfvcom.grid.gtsz_diagnostics import coordinate_summary, rx0
from xfvcom.io.dep_reader import read_dep, write_dep
from xfvcom.io.sigma_dat import read_sigma_dat, write_sigma_dat


# --------------------------------------------------------------------------
#  Reference: an inline copy of FVCOM check_coord.py::sigma_gtsz (the golden)
# --------------------------------------------------------------------------
def _ref_sigma_gtsz(H, kb, base, nz, zlev, k1, k2, p1=2.0, l1=1.0, l2=1.0):
    """Verbatim port of FVCOM/Tests/MultiSigma/bu_shallow/check_coord.py::sigma_gtsz
    (non-mask hard-max B-sandwich + monotonic clamp)."""
    H = np.atleast_1d(np.asarray(H, float))
    M = len(H)
    Z = np.zeros((M, kb))
    khalf = (kb + 1) // 2
    eta0 = 0.0
    have_z = nz > 0
    zlev = np.asarray(zlev, float)
    for i in range(M):
        D = H[i]
        for K in range(1, kb + 1):
            if base == 1:
                SA = (1 - K) / (kb - 1)
            elif base == 2:
                if K <= khalf:
                    SA = -0.5 * ((2 * K - 2) / (kb - 1)) ** p1
                else:
                    SA = 0.5 * ((2 * kb - 2 * K) / (kb - 1)) ** p1 - 1.0
            else:
                SA = (
                    np.tanh(((1 - K) * l1 + (kb - K) * l2) / (kb - 1)) - np.tanh(l2)
                ) / (np.tanh(l1) + np.tanh(l2))
            if (not have_z) or K <= k1 or K >= k2:
                z = SA
            else:
                kz = K - k1
                s4 = (zlev[kz - 1] - eta0) / (D + eta0)
                z = max(SA, s4)
            if K > 1 and z >= Z[i, K - 2]:
                z = Z[i, K - 2] - 1.0e-6
            Z[i, K - 1] = z
        Z[i, 0] = 0.0
        Z[i, kb - 1] = -1.0
    return Z


# --------------------------------------------------------------------------
#  base_sigma / sband
# --------------------------------------------------------------------------
@pytest.mark.parametrize("base", [1, 2, 3])
def test_base_sigma_endpoints(base):
    sa = base_sigma(31, base)
    assert sa[0] == pytest.approx(0.0)
    assert sa[-1] == pytest.approx(-1.0)
    assert np.all(np.diff(sa) < 0)  # strictly decreasing


def test_sband_reshape_endpoints():
    sb = sband_reshape(31, 10, base=2)
    assert sb[0] == pytest.approx(0.0)
    assert sb[-1] == pytest.approx(1.0)


# --------------------------------------------------------------------------
#  base / B-sandwich path vs the FVCOM reference (golden regression)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("base", [1, 2, 3])
def test_pure_base_sigma_matches_reference(base):
    """NZ=0: pure base sigma must equal the reference for every depth + base."""
    H = np.array([5.0, 30.0, 100.0, 647.0])
    kb = 31
    spec = GtszSpec(kb=kb, base=base, nz=0)
    Z = build_coordinate(H, spec)
    Zref = _ref_sigma_gtsz(H, kb, base, 0, [], 1, kb)
    np.testing.assert_allclose(Z, Zref, atol=1e-12, rtol=0)


def test_bsandwich_matches_reference():
    """B-sandwich (hard max, mask off, smooth off) must equal the reference."""
    H = np.array([30.0, 100.0, 300.0, 960.0])
    kb = 41
    zlev = [
        -2.0,
        -4.0,
        -6.0,
        -8.0,
        -10.0,
        -13.0,
        -16.0,
        -20.0,
        -25.0,
        -30.0,
        -35.0,
        -40.0,
        -45.0,
        -52.0,
        -58.0,
        -66.0,
        -74.0,
        -83.0,
        -93.0,
        -105.0,
        -118.0,
        -132.0,
        -149.0,
        -167.0,
        -188.0,
        -211.0,
        -237.0,
        -266.0,
        -299.0,
        -336.0,
        -378.0,
        -424.0,
        -477.0,
        -536.0,
        -602.0,
        -677.0,
        -760.0,
        -854.0,
        -960.0,
    ]
    spec = GtszSpec(kb=kb, base=1, k1=1, k2=kb, nz=len(zlev), zlev=tuple(zlev))
    Z = build_coordinate(H, spec)
    Zref = _ref_sigma_gtsz(H, kb, 1, len(zlev), zlev, 1, kb)
    np.testing.assert_allclose(Z, Zref, atol=1e-12, rtol=0)
    # In the hard-max (non-mask) B-sandwich the z-levels sit at fixed physical
    # depth only where they win the max(SA, S4) -- i.e. mid-column; near the
    # surface/bed the sigma stretch SA dominates. Check a mid-column z-level.
    far = np.argmax(H)  # H = 960
    k_mid = 20  # zlev[19] = -105 m, well inside (SA(K=21) ~ -0.5 -> -480 m)
    assert (Z[far, k_mid] * H[far]) == pytest.approx(zlev[k_mid - 1], abs=1e-6)


# --------------------------------------------------------------------------
#  Coordinate invariants (both paths)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("base", [1, 2, 3])
def test_columns_monotone_and_endpoints(base):
    H = np.array([5.0, 18.0, 50.0, 200.0, 600.0])
    spec = GtszSpec(kb=31, base=base, nz=0)
    Z = build_coordinate(H, spec)
    assert np.allclose(Z[:, 0], 0.0)
    assert np.allclose(Z[:, -1], -1.0)
    assert np.all(np.diff(Z, axis=1) < 0)  # strictly decreasing everywhere


# --------------------------------------------------------------------------
#  SADAPT path
# --------------------------------------------------------------------------
def _make_rect_mesh(nx=21, ny=5, lx=20000.0, ly=4000.0):
    """A small triangulated rectangle; returns x, y, nv (N,3 0-based)."""
    xs = np.linspace(0, lx, nx)
    ys = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(xs, ys)
    x = X.ravel()
    y = Y.ravel()
    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1 = n0 + 1
            n2 = n0 + nx
            n3 = n2 + 1
            tris.append([n0, n1, n3])
            tris.append([n0, n3, n2])
    return x, y, np.asarray(tris, dtype=np.int64)


def _steep_mouth_bathy(x: np.ndarray, lx: float) -> np.ndarray:
    """Head-shallow -> gentle inner bay -> a STEEP deep mouth (slope >> a small
    SMAX), so a SADAPT z-band genuinely activates at the mouth."""
    h = 8.0 + 22.0 * (x / lx)  # gentle 8 -> 30 m (slope ~ 1.4e-3)
    h = h + 270.0 * 0.5 * (1.0 + np.tanh((x - 0.82 * lx) / (0.025 * lx)))  # steep mouth
    return h


def test_grassfire_flat_branch():
    """smax <= 0 -> flat reference at global min(H) (Fortran GTSZ_SMAX<=0 branch)."""
    x, y, nv = _make_rect_mesh()
    H = 5.0 + (x / x.max()) * 100.0  # head-shallow -> mouth-deep
    href = grassfire_href(H, nv, x, y, 0.0)
    assert np.allclose(href, H.min())


def test_grassfire_lipschitz_and_bound():
    """H_ref <= H and edge slope <= smax everywhere; gentle field unchanged."""
    x, y, nv = _make_rect_mesh()
    # a steep deep dip in the middle
    H = 10.0 + 200.0 * np.exp(-(((x - x.max() / 2) / 800.0) ** 2))
    smax = 0.02
    href = grassfire_href(H, nv, x, y, smax)
    assert np.all(href <= H + 1e-9)
    # edge-slope bound
    e = np.vstack([nv[:, [0, 1]], nv[:, [1, 2]], nv[:, [2, 0]]])
    a, b = e[:, 0], e[:, 1]
    dist = np.hypot(x[a] - x[b], y[a] - y[b])
    slope = np.abs(href[a] - href[b]) / np.maximum(dist, 1e-9)
    assert slope.max() <= smax * (1 + 1e-6)
    # a gentle uniform field (slope < smax) is unchanged
    Hg = 10.0 + (x / x.max()) * 50.0  # slope ~ 50/20000 = 2.5e-3 < smax
    href_g = grassfire_href(Hg, nv, x, y, smax)
    np.testing.assert_allclose(href_g, Hg, atol=1e-6)


def test_sadapt_deep_column_has_fixed_depth_zlevels():
    lx = 20000.0
    x, y, nv = _make_rect_mesh(lx=lx)
    H = _steep_mouth_bathy(x, lx)  # gentle inner bay + steep deep mouth
    mesh = MeshInputs(x=x, y=y, nv=nv, h=H)
    zlev = suggest_zlev(
        float(H.max()), z_top=-2.0, dz_shallow=4.0, stretch=1.15, dz_max=40.0
    )
    k1 = 12
    smax = 0.01  # gentle inner bay (1.4e-3) stays sigma; steep mouth becomes z
    href = grassfire_href(H, nv, x, y, smax)
    kb = auto_kb(H, href, k1, zlev)
    spec = GtszSpec(
        kb=kb,
        base=2,
        k1=k1,
        k2=kb,
        nz=len(zlev),
        zlev=tuple(zlev),
        mask=True,
        sadapt=True,
        smax=smax,
    )
    coord = build_gtsz(mesh, spec)
    # every column: endpoints + monotone over the real layers
    for i in range(mesh.n_node):
        z = coord.Z[i]
        assert z[0] == pytest.approx(0.0)
        assert z[-1] == pytest.approx(-1.0)
        kbp = coord.kbp[i]
        if kbp >= 2:
            assert np.all(np.diff(z[:kbp]) < 0)
    # KBP rises from head (gentle, ~K1) to mouth (deep, > K1)
    head = np.argmin(H)
    mouth = np.argmax(H)
    assert coord.kbp[mouth] > coord.kbp[head]
    assert coord.kbp[head] <= k1
    # the appended z-levels sit at fixed physical depth in the deep column
    zc = coord.Z[mouth]
    zphys = zc * H[mouth]
    real = zphys[(zc > -1.0 + DZ_MIN_KBP)]
    # the deep real layers below -H_ref should be a subset of ZLEV depths
    deep = real[real < -coord.href[mouth]]
    for d in deep:
        assert np.min(np.abs(zlev - d)) < 1e-5 or d == pytest.approx(0.0)


def test_sadapt_requires_mask():
    # nz = K2-K1-1 = 9 so the spec passes the z-band-size check and reaches the
    # SADAPT->MASK requirement check.
    with pytest.raises(ValueError, match="requires GTSZ MASK"):
        GtszSpec(
            kb=20,
            k1=10,
            k2=20,
            nz=9,
            zlev=tuple(-np.arange(2.0, 20.0, 2.0)),
            mask=False,
            sadapt=True,
            smax=0.02,
        ).validate()


def test_kbp1_frac1_full_column():
    """A full (un-killed) mesh has KBP=KBM1, KBP1=KBM1, FRAC1=1."""
    x, y, nv = _make_rect_mesh(nx=6, ny=3, lx=1000.0, ly=400.0)
    H = np.full(x.shape, 20.0)
    spec = GtszSpec(kb=11, base=1, nz=0)
    Z = build_coordinate(H, spec)
    kbp = compute_kbp(Z, spec.kbm1)
    assert np.all(kbp == spec.kbm1)
    kbp1, frac1 = compute_kbp1_frac1(Z, nv, kbp)
    assert np.all(kbp1 == spec.kbm1)
    np.testing.assert_allclose(frac1, 1.0)


# --------------------------------------------------------------------------
#  Spec validation (replicates the Fortran FATAL_ERRORs)
# --------------------------------------------------------------------------
def test_validate_zlev_descending():
    with pytest.raises(ValueError, match="strictly descending"):
        GtszSpec(kb=20, k1=10, k2=20, nz=3, zlev=(-2.0, -2.0, -4.0)).validate()


def test_validate_zlev_negative():
    with pytest.raises(ValueError, match="negative"):
        GtszSpec(kb=20, k1=10, k2=20, nz=2, zlev=(2.0, -4.0)).validate()


def test_validate_k1_k2():
    with pytest.raises(ValueError, match="K1"):
        GtszSpec(kb=20, k1=10, k2=11, nz=5, zlev=tuple(-np.arange(2, 12, 2))).validate()


# --------------------------------------------------------------------------
#  sigma.dat / dep.dat round-trips
# --------------------------------------------------------------------------
def test_sigma_dat_roundtrip(tmp_path):
    zlev = tuple(float(z) for z in -np.arange(2.0, 42.0, 2.0))
    spec = GtszSpec(
        kb=31,
        base=2,
        k1=10,
        k2=31,
        nz=len(zlev),
        zlev=zlev,
        mask=True,
        sadapt=True,
        smax=8.0e-4,
    )
    p = tmp_path / "t_sigma.dat"
    write_sigma_dat(p, spec, header_comment="test grid")
    sf = read_sigma_dat(p)
    assert sf.stype == "SIGMAZ"
    assert sf.kb == 31
    g = sf.gtsz
    assert g.base == 2 and g.k1 == 10 and g.k2 == 31 and g.nz == len(zlev)
    assert g.mask and g.sadapt
    assert g.smax == pytest.approx(8.0e-4)
    np.testing.assert_allclose(np.asarray(g.zlev), np.asarray(zlev))


def test_read_canonical_sadapt_header(tmp_path):
    """Parse the exact canonical FVCOM SADAPT header format."""
    text = (
        "NUMBER OF SIGMA LEVELS = 31\n"
        "SIGMA COORDINATE TYPE = SIGMAZ\n"
        "GTSZ BASE = 2\n"
        "GTSZ K1 = 10\n"
        "GTSZ K2 = 31\n"
        "GTSZ P1 = 2.0\n"
        "GTSZ L1 = 1.0\n"
        "GTSZ L2 = 1.0\n"
        "GTSZ NZ = 20\n"
        "GTSZ ZLEV = -2.0 -4.0 -6.0 -8.0 -10.0 -12.0 -14.0 -16.0 -18.0 -20.0 "
        "-22.0 -24.0 -26.0 -28.0 -30.0 -32.0 -34.0 -36.0 -38.0 -40.0\n"
        "GTSZ MASK = T\n"
        "GTSZ SADAPT = T\n"
        "GTSZ SMAX = 0.000800\n"
    )
    p = tmp_path / "slope_sigma.dat"
    p.write_text(text)
    sf = read_sigma_dat(p)
    assert sf.kb == 31 and sf.stype == "SIGMAZ"
    assert sf.gtsz.nz == 20 and len(sf.gtsz.zlev) == 20
    assert sf.gtsz.mask and sf.gtsz.sadapt
    assert sf.gtsz.smax == pytest.approx(8e-4)


def test_dep_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    n = 50
    x = rng.uniform(3.8e5, 3.9e5, n)
    y = rng.uniform(3.94e6, 3.95e6, n)
    h = rng.uniform(3.0, 600.0, n)
    p = tmp_path / "t_dep.dat"
    write_dep(p, x, y, h)
    d = read_dep(p)
    assert d.n_node == n
    np.testing.assert_allclose(d.x, x, atol=1e-4)
    np.testing.assert_allclose(d.h, h, atol=1e-4)


# --------------------------------------------------------------------------
#  Diagnostics
# --------------------------------------------------------------------------
def test_rx0_and_summary():
    lx = 20000.0
    x, y, nv = _make_rect_mesh(lx=lx)
    H = _steep_mouth_bathy(x, lx)
    mesh = MeshInputs(x=x, y=y, nv=nv, h=H)
    r = rx0(H, nv)
    assert 0.0 < r < 1.0
    zlev = suggest_zlev(float(H.max()), dz_shallow=4.0, dz_max=40.0)
    href = grassfire_href(H, nv, x, y, 0.01)
    kb = auto_kb(H, href, 12, zlev)
    spec = GtszSpec(
        kb=kb,
        base=2,
        k1=12,
        k2=kb,
        nz=len(zlev),
        zlev=tuple(zlev),
        mask=True,
        sadapt=True,
        smax=0.01,
    )
    coord = build_gtsz(mesh, spec)
    summ = coordinate_summary(coord)
    assert summ.kbp_min <= summ.kbp_max <= spec.kbm1
    assert 0.0 <= summ.killed_waste <= 1.0
    assert summ.href_min is not None
    assert "KBP" in summ.as_text()
