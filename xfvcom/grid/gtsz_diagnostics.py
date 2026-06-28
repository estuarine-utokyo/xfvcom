"""Diagnostics for an FVCOM sigma-z (GTSZ) coordinate.

Operates on a :class:`~xfvcom.grid.gtsz.GtszCoordinate` (built by
:func:`xfvcom.grid.gtsz_builder.build_gtsz`). Three families:

* **Geometry** -- :func:`rx0` (Beckmann-Haidvogel Haney number), :func:`coordinate_summary`
  (KBP distribution, pure-sigma fraction, z-band depth range, killed-cell waste).
* **Numerical error** -- :func:`offline_sigma_pge`, a faithful port of
  ``FVCOM/Tests/MultiSigma/bu_shallow/analyze_bpg_offline.py``: the spurious
  baroclinic pressure-gradient force per element-layer for an at-rest
  (horizontal-isopycnal) density field, where the analytic BPG is **zero** -- so
  the magnitude *is* the sigma-coordinate PGE. A cheap relative pre-screen before
  the authoritative FVCOM at-rest run.
* **Plots** -- :func:`plot_transect`, :func:`plot_maps` (matplotlib imported lazily).

All metrics are evaluated at ``eta = 0`` (cold-start at-rest).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .gtsz import DZ_MIN_KBP, GtszCoordinate

__all__ = [
    "rx0",
    "coordinate_summary",
    "linear_density",
    "offline_sigma_pge",
    "plot_transect",
    "plot_maps",
]

GRAV = 9.806


# ===========================================================================
#  Geometry
# ===========================================================================
def rx0(
    h: NDArray[np.float64], nv: NDArray[np.int64], *, per_node: bool = False
) -> NDArray[np.float64] | float:
    """Beckmann-Haidvogel r-factor (Haney number) ``rx0 = |h_a-h_b|/(h_a+h_b)``
    over mesh edges. Returns the max (default) or a per-node max array.

    The sigma-coordinate PGE grows with ``rx0``; the literature target is
    ``rx0 <= 0.2`` over the sigma region.
    """
    h = np.asarray(h, dtype=np.float64)
    nv = np.asarray(nv)
    if nv.shape[0] == 3 and nv.shape[1] != 3:
        nv = nv.T
    e = np.vstack([nv[:, [0, 1]], nv[:, [1, 2]], nv[:, [2, 0]]])
    a, b = e[:, 0], e[:, 1]
    r = np.abs(h[a] - h[b]) / (h[a] + h[b])
    if not per_node:
        return float(r.max())
    out = np.zeros(h.shape[0], dtype=np.float64)
    np.maximum.at(out, a, r)
    np.maximum.at(out, b, r)
    return out


@dataclass
class CoordinateSummary:
    """Scalar summary of a built coordinate (see :func:`coordinate_summary`)."""

    kb: int
    n_node: int
    kbp_min: int
    kbp_max: int
    n_reduced: int
    frac_pure_sigma: float  #: fraction of nodes that are full-depth sigma (KBP==KBM1)
    killed_waste: float  #: fraction of (node, layer) cells that are killed/degenerate
    rx0_max: float
    href_min: float | None
    href_max: float | None
    zband_depth_max: float | None  #: deepest physical depth a z-level reaches [m]

    def as_text(self) -> str:
        lines = [
            f"KB = {self.kb} ({self.kb - 1} layers), nodes = {self.n_node}",
            f"KBP (active layers): {self.kbp_min} .. {self.kbp_max} (full = {self.kb - 1})",
            f"reduced (masked) columns : {self.n_reduced} "
            f"({100 * self.n_reduced / self.n_node:.1f}%)",
            f"pure-sigma columns       : {100 * self.frac_pure_sigma:.1f}%",
            f"killed-cell waste        : {100 * self.killed_waste:.1f}% of (node,layer) cells",
            f"rx0 (Haney) max          : {self.rx0_max:.3f}",
        ]
        if self.href_min is not None:
            lines.append(
                f"H_ref range              : {self.href_min:.1f} .. {self.href_max:.1f} m"
            )
        if self.zband_depth_max is not None:
            lines.append(f"deepest z-level reaches  : {self.zband_depth_max:.1f} m")
        return "\n".join(lines)


def coordinate_summary(coord: GtszCoordinate) -> CoordinateSummary:
    """Compute a :class:`CoordinateSummary` for a built coordinate."""
    kbm1 = coord.spec.kbm1
    kbp = coord.kbp
    dz = coord.dz  # (M, KB-1)
    killed = dz < DZ_MIN_KBP
    n_reduced = int((kbp < kbm1).sum())
    frac_pure = float((kbp >= kbm1).mean())
    waste = float(killed.mean())
    rx0_max = float(rx0(coord.H, coord.nv)) if coord.nv is not None else float("nan")

    href_min = href_max = None
    if coord.href is not None:
        href_min = float(coord.href.min())
        href_max = float(coord.href.max())

    zband_depth_max = None
    if coord.spec.have_zband and coord.spec.zlev:
        # deepest z-level that becomes a real layer anywhere
        zmin = min(coord.spec.zlev)  # most negative listed
        # but only count it if it is actually realized (above some node's bed)
        zband_depth_max = float(-zmin)

    return CoordinateSummary(
        kb=coord.spec.kb,
        n_node=coord.H.shape[0],
        kbp_min=int(kbp.min()),
        kbp_max=int(kbp.max()),
        n_reduced=n_reduced,
        frac_pure_sigma=frac_pure,
        killed_waste=waste,
        rx0_max=rx0_max,
        href_min=href_min,
        href_max=href_max,
        zband_depth_max=zband_depth_max,
    )


# ===========================================================================
#  Offline sigma-PGE (BPG) estimator
# ===========================================================================
def linear_density(
    s: NDArray[np.float64],
    t: NDArray[np.float64],
    *,
    rho0: float = 1025.0,
    alpha: float = 2.0e-4,
    beta: float = 7.6e-4,
    t0: float = 10.0,
    s0: float = 34.0,
) -> NDArray[np.float64]:
    """Simple linear equation of state ``rho = rho0*(1 - alpha*(T-T0) + beta*(S-S0))``.

    Adequate for the *relative* sigma-PGE metric (the BPG force is reported in
    arbitrary units, consistent across the compared grids; see the reference
    ``analyze_bpg_offline.py``). For the absolute at-rest spurious current, run
    the FVCOM at-rest test (the authoritative metric).
    """
    return rho0 * (1.0 - alpha * (t - t0) + beta * (s - s0))


def _sinter(X: NDArray, A: NDArray, Y: NDArray) -> NDArray:
    """FVCOM ``SINTER_EXTRP_NONE`` 1-D interpolation of ``A(X)`` onto ``Y``
    (X descending). Clamps outside the range. Port of ``analyze_bpg_offline.py``.
    """
    B = np.empty(len(Y))
    for i, y in enumerate(Y):
        if y > X[0]:
            B[i] = A[0]
        elif y < X[-1]:
            B[i] = A[-1]
        else:
            j = min(int(np.searchsorted(-X, -y)) - 1, len(X) - 2)
            j = max(j, 0)
            B[i] = A[j] - (A[j] - A[j + 1]) * (X[j] - y) / (X[j] - X[j + 1])
    return B


@dataclass
class SigmaPgeResult:
    """Result of :func:`offline_sigma_pge`."""

    bpg: NDArray[
        np.float64
    ]  #: (N, KBM1) |BPG force| per element-layer (arbitrary units)
    max_bpg: float
    max_elem: int
    max_level: int  #: 1-based layer K of the peak
    max_depth: float  #: physical depth [m] of the peak
    mean_bpg: float
    p95_bpg: float

    def as_text(self) -> str:
        return (
            f"offline sigma-PGE (|BPG|, arb. units): max {self.max_bpg:.4e} "
            f"at elem {self.max_elem}, K={self.max_level} (z~{self.max_depth:.1f} m); "
            f"mean {self.mean_bpg:.4e}, p95 {self.p95_bpg:.4e}"
        )


def offline_sigma_pge(
    coord: GtszCoordinate,
    *,
    ts_profile=None,
    salt: NDArray[np.float64] | None = None,
    temp: NDArray[np.float64] | None = None,
    zref: NDArray[np.float64] | None = None,
    near_mask: NDArray[np.bool_] | None = None,
) -> SigmaPgeResult:
    """Offline sigma-coordinate PGE estimate: the spurious BPG force per
    element-layer for an at-rest (horizontal-isopycnal) density field.

    The density is built from a 1-D ``T(z), S(z)`` profile (so isopycnals are
    level and the analytic BPG is **zero**). Any nonzero result is the
    sigma-coordinate truncation error. Faithful port of the FVCOM
    ``analyze_bpg_offline.py`` global-reference ``bpg_elem`` (DRIJK1 + DRIJK2 edge
    projections, ``RHO_PMEAN`` global z-layer reference).

    Parameters
    ----------
    coord : GtszCoordinate
        The built coordinate (needs ``nv``, ``x``, ``y``).
    ts_profile : callable, optional
        ``ts_profile(z) -> (T, S)`` for ``z <= 0`` (m). Default: the PGE-harness
        profile (T 24->14 degC, S 31->34.5 PSU linear over 0..-Hmax).
    salt, temp : (M, KBM1) arrays, optional
        Pre-computed node-layer S/T (overrides ``ts_profile``).
    zref : (NZL,) array, optional
        Reference z-levels for ``RHO_PMEAN`` (default: the coordinate's own
        ``BPG REF ZLEV`` or a 0..-Hmax stretch).
    near_mask : (N,) bool, optional
        Restrict the reported peak to these elements (e.g. the deep mouth).
    """
    if coord.nv is None or coord.x is None or coord.y is None:
        raise ValueError("offline_sigma_pge: coord needs nv, x, y")
    H = coord.H
    nv = np.asarray(coord.nv)
    if nv.shape[0] == 3 and nv.shape[1] != 3:
        nv = nv.T
    xy = np.column_stack([coord.x, coord.y])
    Z = coord.Z
    kbm1 = coord.spec.kbm1
    ZZ = 0.5 * (Z[:, :-1] + Z[:, 1:])  # (M, KBM1) layer-center sigma
    DZ = Z[:, :-1] - Z[:, 1:]
    znode = ZZ * H[:, None]  # (M, KBM1) physical depth

    # --- density field from a level (horizontal-isopycnal) profile ----------
    if salt is None or temp is None:
        hmax = float(H.max())
        if ts_profile is None:

            def ts_profile(z):  # PGE-harness IC (tb_pge_run.nml)
                frac = np.clip(-z / hmax, 0.0, 1.0)
                t = 24.0 + (14.0 - 24.0) * frac
                s = 31.0 + (34.5 - 31.0) * frac
                return t, s

        temp, salt = ts_profile(znode)
    RHO1 = linear_density(np.asarray(salt), np.asarray(temp))

    # --- global RHO_PMEAN reference on z-layers ------------------------------
    if zref is None:
        if coord.spec.bpg_ref_zlev:
            zref = np.asarray(coord.spec.bpg_ref_zlev, dtype=np.float64)
        else:
            zref = -np.linspace(0.0, float(H.max()), 60)
    zref = np.sort(np.asarray(zref, dtype=np.float64))[::-1]  # descending
    NZL = len(zref)

    acc = np.zeros(NZL)
    cnt = np.zeros(NZL)
    for i in range(len(H)):
        rz = _sinter(znode[i], RHO1[i], zref)
        m = zref >= -H[i]
        acc[m] += rz[m]
        cnt[m] += 1
    kfull = int(np.max(np.where(cnt > 0))) + 1
    rhoa = acc / np.maximum(cnt, 1)
    RMEAN1 = np.array(
        [_sinter(zref[:kfull], rhoa[:kfull], znode[i]) for i in range(len(H))]
    )
    res_node = RHO1 - RMEAN1  # (M, KBM1)

    # --- per-element BPG force (DRIJK1 + DRIJK2 edge projection) -------------
    EDGE = [(1, 2), (2, 0), (0, 1)]
    bpg = np.zeros((nv.shape[0], kbm1))
    for I in range(nv.shape[0]):
        nn = nv[I]
        ZZ1 = ZZ[nn].mean(0)
        DZ1 = DZ[nn].mean(0)
        DT1 = H[nn].mean()
        zedge_node = znode[nn]
        resn = res_node[nn]
        RERES = resn.mean(0)
        drhox = np.zeros(kbm1)
        drhoy = np.zeros(kbm1)
        vx, vy = xy[nn, 0], xy[nn, 1]
        for j1, j2 in EDGE:
            RIJK = 0.5 * (resn[j1] + resn[j2])
            ZEDGE = 0.5 * (zedge_node[j1] + zedge_node[j2])
            DRIJK1 = np.zeros(kbm1)
            DRIJK2 = np.zeros(kbm1)
            DRIJK1[0] = RIJK[0] * (-ZZ1[0])
            for K in range(1, kbm1):
                DRIJK1[K] = DRIJK1[K - 1] + 0.5 * (RIJK[K - 1] + RIJK[K]) * (
                    ZZ1[K - 1] - ZZ1[K]
                )
                DRIJK2[K] = DRIJK2[K - 1] + 0.5 * (ZEDGE[K - 1] + ZEDGE[K]) * (
                    RERES[K] - RERES[K - 1]
                )
            drhox += (vy[j1] - vy[j2]) * DRIJK1 * DT1 + (vy[j1] - vy[j2]) * DRIJK2
            drhoy += (vx[j2] - vx[j1]) * DRIJK1 * DT1 + (vx[j2] - vx[j1]) * DRIJK2
        drhox *= DT1 * DZ1 * GRAV
        drhoy *= DT1 * DZ1 * GRAV
        bpg[I] = np.hypot(drhox, drhoy)

    # Zero the killed (degenerate) element-layers (K > KBP1) so the reported peak
    # is a real layer, not a masked-tail artifact (DZ1~1e-6 already suppresses
    # them, but be explicit). NOTE the BPG magnitude is in arbitrary units and
    # scales with depth (DT1^2) + density structure -- it is a same-grid
    # SADAPT-vs-UNIFORM *relative* pre-screen, NOT comparable across bathymetries.
    # The authoritative metric is the at-rest FVCOM run.
    if coord.kbp1 is not None:
        kbp1 = np.asarray(coord.kbp1)[: nv.shape[0]]
        layer_idx = np.arange(kbm1)[None, :]
        bpg[layer_idx >= kbp1[:, None]] = 0.0

    sel = (
        np.ones(nv.shape[0], dtype=bool) if near_mask is None else np.asarray(near_mask)
    )
    bsel = bpg[sel]
    flat_idx = int(np.argmax(bsel))
    ei = np.where(sel)[0][flat_idx // kbm1]
    lvl = flat_idx % kbm1
    zc = znode[nv[ei]].mean(0)[lvl]
    return SigmaPgeResult(
        bpg=bpg,
        max_bpg=float(bsel.max()),
        max_elem=int(ei),
        max_level=int(lvl + 1),
        max_depth=float(zc),
        mean_bpg=float(bsel.mean()),
        p95_bpg=float(np.percentile(bsel, 95)),
    )


# ===========================================================================
#  Plots (matplotlib imported lazily; English-only labels per project rules)
# ===========================================================================
def plot_transect(
    coord: GtszCoordinate,
    *,
    along: str = "x",
    y0: float | None = None,
    x0: float | None = None,
    width: float | None = None,
    section: tuple[tuple[float, float], tuple[float, float]] | None = None,
    n_section: int = 200,
    ax=None,
    title: str | None = None,
):
    """Plot a vertical transect of the coordinate: every level's physical depth
    along a section. Sigma-band levels in blue, z-band levels in red, bed + H_ref
    overlaid.

    Two sampling modes:

    * ``section=((x0,y0),(x1,y1))`` -- sample ``n_section`` points evenly along the
      straight line and take the nearest mesh node at each (the clean way to cut an
      unstructured mesh, e.g. head -> deep mouth). The x-axis is along-line distance.
    * otherwise -- the legacy near-constant ``y`` (``along='x'``) / ``x``
      (``along='y'``) band (jagged on an unstructured mesh; use ``section`` for TB).
    """
    import matplotlib.pyplot as plt

    if coord.x is None or coord.y is None:
        raise ValueError("plot_transect: coord needs node coordinates x, y")
    H, Z = coord.H, coord.Z
    x, y = coord.x, coord.y
    if section is not None:
        from scipy.spatial import cKDTree

        (sx0, sy0), (sx1, sy1) = section
        t = np.linspace(0.0, 1.0, n_section)
        px = sx0 + (sx1 - sx0) * t
        py = sy0 + (sy1 - sy0) * t
        _, idx = cKDTree(np.column_stack([x, y])).query(np.column_stack([px, py]))
        # de-duplicate consecutive repeats while keeping order
        keep = np.concatenate([[True], np.diff(idx) != 0])
        nodes = idx[keep]
        s = np.hypot(px - sx0, py - sy0)[keep] / 1000.0
    else:
        if along == "x":
            coord_along, coord_across = x, y
            c0 = y0 if y0 is not None else float(np.median(y))
        else:
            coord_along, coord_across = y, x
            c0 = x0 if x0 is not None else float(np.median(x))
        if width is None:
            width = 0.02 * (coord_across.max() - coord_across.min())
        sel = np.abs(coord_across - c0) <= width
        order = np.argsort(coord_along[sel])
        nodes = np.where(sel)[0][order]
        s = coord_along[nodes] / 1000.0
    zphys = (Z[nodes] * H[nodes, None]).T  # (KB, n)
    k1 = coord.spec.k1

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 6))
    if coord.href is not None:
        ax.fill_between(
            s,
            -coord.href[nodes],
            -H[nodes],
            color="gold",
            alpha=0.25,
            label="z-region (H_ref..H)",
        )
    for k in range(Z.shape[1]):
        col = "tab:blue" if (k + 1) <= k1 else "tab:red"
        lw = 1.3 if k in (0, k1 - 1) else 0.5
        ax.plot(s, zphys[k], color=col, lw=lw)
    ax.plot(s, -H[nodes], "k-", lw=1.6, label="true bed -H")
    if coord.href is not None:
        ax.plot(s, -coord.href[nodes], "g--", lw=1.3, label="reference -H_ref")
    ax.set_xlabel(
        "distance along section [km]" if section is not None else f"{along} [km]"
    )
    ax.set_ylabel("depth z [m]")
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def plot_maps(coord: GtszCoordinate, *, figsize=(15, 4.2)):
    """Three node maps: depth H, reference H_ref, and the active-layer count KBP
    (turbo). Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    if coord.x is None or coord.y is None or coord.nv is None:
        raise ValueError(
            "plot_maps: coord needs node coordinates x, y and connectivity nv"
        )
    x, y, H = coord.x / 1000.0, coord.y / 1000.0, coord.H
    nv = coord.nv
    if nv.shape[0] == 3 and nv.shape[1] != 3:
        nv = nv.T
    tri = mtri.Triangulation(x, y, nv)
    fields = [
        (H, "depth H [m]", "viridis_r"),
        (
            coord.href if coord.href is not None else H,
            "reference H_ref [m]",
            "viridis_r",
        ),
        (coord.kbp.astype(float), "active-layer count KBP", "turbo"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (F, ttl, cm) in zip(axes, fields):
        im = ax.tripcolor(tri, F, cmap=cm, shading="gouraud")
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, pad=0.01)
    fig.tight_layout()
    return fig
