"""Build an FVCOM sigma-z (GTSZ) vertical coordinate over a real mesh.

This module is the *grid builder*: it ties the mesh I/O (``*_grd.dat`` +
``*_dep.dat``) to the coordinate model (:mod:`xfvcom.grid.gtsz`) and provides the
two pieces the model needs but cannot get from a single column:

* :func:`grassfire_href` -- the gentle Lipschitz lower envelope ``H_ref`` for the
  slope-adaptive (``GTSZ_SADAPT``) coordinate, computed by edge relaxation over
  the unstructured mesh. This is the same fixed point as the in-Fortran grassfire
  (``mod_setup.F::SIGMA_GTSZ`` lines 1162-1203); the Lipschitz envelope is
  order-independent, so the vectorized (Jacobi/Gauss-Seidel-hybrid) sweep here
  converges to the Fortran's converged ``GTSZ_HREF`` (the design doc reports
  agreement < 1e-3 m).

* :func:`build_gtsz` -- assemble a :class:`~xfvcom.grid.gtsz.GtszCoordinate`
  (sigma field + KBP/KBP1/FRAC1 masking maps) from a :class:`MeshInputs` + a
  :class:`~xfvcom.grid.gtsz.GtszSpec`.

Plus helpers to *design* a spec from a target geometry: :func:`active_zlev_count`,
:func:`auto_kb`, :func:`suggest_zlev`.
"""

from __future__ import annotations

# jsasaki 2026-11-05: review round 12 #1: ctypes for the C library's tanhf / powf
import ctypes
import ctypes.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from .dat_reader import read_dat
from .gtsz import (
    DZ_MIN_KBP,
    SMALL,
    GtszCoordinate,
    GtszSpec,
    build_coordinate,
    compute_kbp,
    compute_kbp1_frac1,
)

__all__ = [
    "MeshInputs",
    "load_mesh",
    "grassfire_href",
    "active_zlev_count",
    "auto_kb",
    "suggest_zlev",
    "build_gtsz",
]


@dataclass
class MeshInputs:
    """A horizontal mesh + bathymetry for the sigma-z builder."""

    x: NDArray[np.float64]  #: (M,) node x
    y: NDArray[np.float64]  #: (M,) node y
    nv: NDArray[np.int64]  #: (N, 3) 0-based element->node table
    h: NDArray[np.float64]  #: (M,) node depth [m], positive down

    @property
    def n_node(self) -> int:
        return self.h.shape[0]

    @property
    def n_elem(self) -> int:
        return self.nv.shape[0]


def load_mesh(grd_path: str | Path, dep_path: str | Path) -> MeshInputs:
    """Load a mesh from an FVCOM ``*_grd.dat`` + ``*_dep.dat`` pair.

    The grid file supplies node coordinates + connectivity; the depth file
    supplies ``H``. The two node orderings are cross-checked.
    """
    # Lazy import so that ``import xfvcom.grid`` (which pulls in this module) does
    # NOT eagerly trigger ``xfvcom.io.__init__`` -> ``input_loader`` ->
    # ``xfvcom.grid.FvcomGrid`` (a circular import). Keeping it function-local also
    # makes the module-level import order irrelevant (isort-safe).
    from ..io.dep_reader import read_dep

    g = read_dat(grd_path)
    d = read_dep(dep_path)
    x, y = np.asarray(g["x"], float), np.asarray(g["y"], float)
    nv = np.asarray(g["nv"])  # (3, N)
    # ---> jsasaki 2026-11-05: review round 6 #3: read_dat always returns (3, N); transposing only
    # when N != 3 left a three-element mesh in the wrong orientation
    if nv.ndim != 2 or nv.shape[0] != 3:
        raise ValueError(f"load_mesh: read_dat returned nv of shape {nv.shape}")
    nv = nv.T  # -> (N, 3)
    # <--- jsasaki 2026-11-05
    if d.h.shape[0] != x.shape[0]:
        raise ValueError(
            f"load_mesh: grid has {x.shape[0]} nodes but dep has {d.h.shape[0]}"
        )
    # cross-check coordinates (the dep file repeats X/Y); tolerate tiny rounding
    dxy = np.hypot(d.x - x, d.y - y)
    if np.nanmax(dxy) > 1.0:
        bad = int(np.argmax(dxy))
        raise ValueError(
            f"load_mesh: grd.dat and dep.dat node coordinates disagree "
            f"(max {np.nanmax(dxy):.3g} m at node {bad}); are they the same mesh?"
        )
    return MeshInputs(x=x, y=y, nv=nv.astype(np.int64), h=d.h)


# ===========================================================================
#  H_ref: gentle Lipschitz lower envelope (grassfire)
# ===========================================================================
def _undirected_edges(nv: NDArray[np.int64]) -> NDArray[np.int64]:
    """Unique undirected edges (E, 2) from an (N, 3) connectivity table."""
    e = np.vstack([nv[:, [0, 1]], nv[:, [1, 2]], nv[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def grassfire_href(
    h: NDArray[np.float64],
    nv: NDArray[np.int64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    smax: float,
    *,
    max_sweep: int = 20000,
    tol: float = 1.0e-6,
    return_sweeps: bool = False,
):
    """Reference depth ``H_ref(x) = min_k[ H(k) + smax*dist(k,x) ]``: the largest
    field ``<= H`` with edge slope ``|dH_ref| <= smax`` along every mesh edge.

    Parameters
    ----------
    h : (M,) depth
    nv : (N, 3) 0-based connectivity
    x, y : (M,) node coordinates [m]
    smax : max sigma-slope [m/m]. ``smax <= 0`` -> flat reference ``min(H)``
        (matches the Fortran ``GTSZ_SMAX <= 0`` branch).

    Notes
    -----
    Vectorized edge relaxation (in-place ``np.minimum.at``; a Jacobi/Gauss-Seidel
    hybrid). Converges to the same Lipschitz envelope as the Fortran
    ``mod_setup.F`` Gauss-Seidel loop -- the envelope is the unique fixed point,
    independent of relaxation order.
    """
    h = np.asarray(h, dtype=np.float64)
    if smax <= 0.0:
        href = np.full_like(h, float(h.min()))
        return (href, 0) if return_sweeps else href

    nv = np.asarray(nv)
    if nv.shape[0] == 3 and nv.shape[1] != 3:
        nv = nv.T
    edges = _undirected_edges(nv.astype(np.int64))
    a, b = edges[:, 0], edges[:, 1]
    w = np.hypot(x[a] - x[b], y[a] - y[b]) * float(smax)  # smax * edge length

    href = h.copy()
    sweeps = 0
    # jsasaki 2026-11-05: review round 7 #6: track convergence; an exhausted
    # sweep budget is an error, as in mod_setup.F (FATAL_ERROR), not a result
    dchg = np.inf
    for sweeps in range(1, max_sweep + 1):
        before = href.copy()
        np.minimum.at(href, b, href[a] + w)  # relax a -> b
        np.minimum.at(href, a, href[b] + w)  # relax b -> a (sees the a->b update)
        dchg = float(np.max(before - href)) if href.size else 0.0
        if dchg < tol:
            break
    if not dchg < tol:
        raise RuntimeError(
            f"grassfire_href: H_ref did not converge within max_sweep={max_sweep} "
            f"sweeps (last sweep changed H_ref by {dchg:.6g} m >= tol={tol:g} m); "
            "the result would violate the SMAX slope limit -- check smax/the mesh "
            "or raise max_sweep"
        )
    np.minimum(href, h, out=href)  # safety: H_ref <= H
    return (href, sweeps) if return_sweeps else href


# ===========================================================================
#  Spec design helpers
# ===========================================================================
def active_zlev_count(
    h: NDArray[np.float64],
    href: NDArray[np.float64],
    zlev: NDArray[np.float64],
    *,
    eta0: float = 0.0,
) -> NDArray[np.int64]:
    """Per-node count of z-levels that become *real* layers in the SADAPT column
    (those strictly between ``-H_ref`` and the bed).

    Mirrors the acceptance test in ``build_column_sadapt`` /
    ``mod_setup.F`` lines 1247-1252: a level ``ZLEV`` is active at node ``i`` when
    ``S4 < SIGREF - DZ_MIN_KBP`` and ``S4 > -1 + DZ_MIN_KBP`` with
    ``S4 = ZLEV/(H+eta0)`` and ``SIGREF = -H_ref/(H+eta0)``.
    """
    h = np.asarray(h, dtype=np.float64)
    href = np.minimum(np.asarray(href, dtype=np.float64), h)
    zlev = np.asarray(zlev, dtype=np.float64)
    D = h + eta0
    sigref = np.maximum(-(href + eta0) / D, -1.0)  # (M,)
    s4 = (zlev[None, :] - eta0) / D[:, None]  # (M, NZ)
    active = (s4 < sigref[:, None] - DZ_MIN_KBP) & (s4 > -1.0 + DZ_MIN_KBP)
    return active.sum(axis=1).astype(np.int64)


def auto_kb(
    h: NDArray[np.float64],
    href: NDArray[np.float64],
    k1: int,
    zlev: NDArray[np.float64],
    *,
    eta0: float = 0.0,
) -> int:
    """Smallest ``KB`` that fits the deepest SADAPT column:
    ``KB = K1 + max_i(active_zlev_count_i) + 1`` (room for the degenerate bed).

    Raises if no z-level activates anywhere (the domain is gentler than ``SMAX``
    so ``H_ref == H`` everywhere -> the z-band is empty and SADAPT is degenerate;
    use a smaller ``SMAX``, a deeper / steeper bathymetry, or a pure-sigma
    coordinate instead).
    """
    nmax = int(active_zlev_count(h, href, zlev, eta0=eta0).max())
    if nmax < 1:
        raise ValueError(
            "auto_kb: no z-level activates in any column (H_ref == H everywhere: "
            "the domain is gentler than SMAX, or ZLEV does not reach below H_ref). "
            "SADAPT would be degenerate -- lower SMAX, deepen the bathymetry, or "
            "use a pure-sigma coordinate."
        )
    return int(k1) + nmax + 1


def suggest_zlev(
    h_max: float,
    *,
    z_top: float = -2.0,
    dz_shallow: float = 2.0,
    stretch: float = 1.12,
    dz_max: float = 60.0,
    href_min: float | None = None,
) -> NDArray[np.float64]:
    """Suggest a geometrically-stretched ``ZLEV`` set spanning ``z_top`` down to
    just past ``-h_max``.

    The first interval is ``dz_shallow`` m (fine near the σ/z interface where the
    estuarine pycnocline + deep inflow live), growing by ``stretch`` each step up
    to ``dz_max`` m in the deep where vertical structure is weak. Levels are
    negative and strictly descending (the FVCOM ZLEV contract). ``href_min`` (if
    given) is only used to advise -- the deepest level just needs to reach the
    bed.
    """
    levels = [float(z_top)]
    dz = float(dz_shallow)
    while levels[-1] - dz > -float(h_max):
        levels.append(levels[-1] - dz)
        dz = min(dz * stretch, dz_max)
    # jsasaki 2026-11-05: review round 9 #5: micrometre-rounded levels, so that the
    # exact sigma.dat spelling of a long stretched set still fits FVCOM's 400-char line
    # ---> jsasaki 2026-11-05: review round 10 #3: rounding must not break the
    # ZLEV contract (a level rounded to -0.0, or two levels rounded together);
    # in that case the unrounded levels are kept (the writer still raises its
    # 400-character error if their exact spelling does not fit)
    rounded = [round(v, 6) for v in levels]
    r = np.asarray(rounded, dtype=np.float64)
    if (r < 0.0).all() and (np.diff(r) < 0.0).all():
        levels = rounded
    # <--- jsasaki 2026-11-05
    # ensure the last level is at least as deep as the bed
    if levels[-1] > -float(h_max):
        levels.append(-float(h_max))
    return np.asarray(levels, dtype=np.float64)


# ===========================================================================
#  Assemble the coordinate
# ===========================================================================
#: FVCOM ``REAL(SP)`` precision -> numpy dtype (``SP`` is double with ``-DDOUBLE_PRECISION``)
_MODEL_DTYPES: dict[str, type[np.floating]] = {
    "double": np.float64,
    "single": np.float32,
}


# ---> jsasaki 2026-11-05: review round 12 #1: the single-precision check uses the
# C math library's tanhf / powf (what a single-precision FVCOM build calls) when
# it can be loaded, and numpy's float32 functions otherwise
def _load_libm_f32() -> (
    tuple[Callable[[float], float], Callable[[float, float], float]] | None
):
    """``(tanhf, powf)`` from the C math library via :mod:`ctypes`, or ``None``."""
    names = [ctypes.util.find_library("m"), "libm.so.6", "libm.dylib"]
    for name in names:
        if not name:
            continue
        try:
            lib = ctypes.CDLL(name)
            tanhf, powf = lib.tanhf, lib.powf
        except (OSError, AttributeError):
            continue
        tanhf.restype, tanhf.argtypes = ctypes.c_float, [ctypes.c_float]
        powf.restype, powf.argtypes = ctypes.c_float, [ctypes.c_float, ctypes.c_float]
        return tanhf, powf
    return None


_LIBM_F32 = _load_libm_f32()

#: interface separations below this many float32 units in the last place (of
#: the operands of the subtraction that forms them) are refused in a
#: single-precision check: the runtime's tanhf/powf and the compiler (operation
#: order, fused multiply-add) can move a value by an ulp or more, so such a
#: separation may be zero or negative in the model
SP_MIN_ULP = 16


def _tanh_sp(x: Any, dt: type[np.floating]) -> Any:
    """``TANH`` of ``REAL(SP)`` values (scalar or array) in precision ``dt``."""
    if dt is np.float32 and _LIBM_F32 is not None:
        f = _LIBM_F32[0]
        a = np.asarray(x, dtype=dt)
        out = np.array([f(float(v)) for v in a.ravel()], dtype=dt).reshape(a.shape)
        return dt(out[()]) if out.ndim == 0 else out
    return np.tanh(np.asarray(x, dtype=dt)) if np.ndim(x) else dt(np.tanh(dt(x)))


def _pow_sp(x: Any, y: Any, dt: type[np.floating]) -> Any:
    """Scalar ``X**Y`` (real exponent) in precision ``dt``."""
    if dt is np.float32 and _LIBM_F32 is not None:
        return dt(_LIBM_F32[1](float(x), float(y)))
    return dt(dt(x) ** dt(y))


def _ulp(x: Any) -> Any:
    """float32 spacing at ``|x|`` (elementwise)."""
    return np.spacing(np.abs(np.asarray(x, dtype=np.float32)))


# <--- jsasaki 2026-11-05


# jsasaki 2026-11-05: review round 11 #1: new helper (SBAND in model precision)
def _sband_sp(spec: GtszSpec, dtype: type[np.floating]) -> NDArray[np.floating]:
    """``SBAND(1:K1)`` of ``mod_setup.F::SIGMA_GTSZ`` evaluated in ``REAL(SP)``
    precision ``dtype``: the base shape, ``SAK1`` with its ``|SAK1| < SMALL``
    guard, and the normalisation ``SBAND(K)/SAK1``, operation by operation in the
    Fortran order with ``GTSZ P1/L1/L2`` converted to ``dtype``.

    In float32 the C library's ``tanhf`` / ``powf`` are used when available
    (numpy's float32 functions otherwise).  Another runtime can still differ by
    an ulp, so in float32 each upper-band separation ``SBAND(K-1) - SBAND(K)``
    must exceed :data:`SP_MIN_ULP` ulps of the operands it is formed from (for
    the A3 shape the ``tanh`` values near 1); a :class:`ValueError` otherwise.
    """
    dt = dtype
    kb, k1 = spec.kb, spec.k1
    p1, l1, l2 = dt(spec.p1), dt(spec.l1), dt(spec.l2)
    khalf = (kb + 1) // 2
    sb = np.empty(k1, dtype=dt)
    # jsasaki 2026-11-05: review round 12 #1: magnitude of the operands behind each SBAND(K)
    mag = np.empty(k1, dtype=dt)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # jsasaki 2026-11-05: review round 12 #1: tanhf / powf of the C library
        tl1, tl2 = _tanh_sp(l1, dt), _tanh_sp(l2, dt)
        for k in range(1, k1 + 1):
            if spec.base == 1:
                sb[k - 1] = dt(1 - k) / dt(kb - 1)
                mag[k - 1] = abs(sb[k - 1])
            elif spec.base == 2:
                if k <= khalf:
                    pw = _pow_sp(dt(2 * k - 2) / dt(kb - 1), p1, dt)
                    sb[k - 1] = dt(-0.5) * pw
                    mag[k - 1] = abs(sb[k - 1])
                else:
                    pw = _pow_sp(dt(2 * kb - 2 * k) / dt(kb - 1), p1, dt)
                    sb[k - 1] = dt(0.5) * pw - dt(1.0)
                    mag[k - 1] = dt(1.0)
            else:
                num = _tanh_sp((dt(1 - k) * l1 + dt(kb - k) * l2) / dt(kb - 1), dt)
                sb[k - 1] = (num - tl2) / (tl1 + tl2)
                mag[k - 1] = max(abs(num), abs(tl2))
        sak1 = sb[k1 - 1]
        if abs(sak1) < dt(SMALL):
            sak1 = dt(-SMALL)
        out: NDArray[np.floating] = (sb / sak1).astype(dt)
    # ---> jsasaki 2026-11-05: review round 12 #1: a separation within a few ulps
    # of the operands can vanish under another tanhf/powf or compiler
    if dt is np.float32 and k1 >= 2:
        # interface 1 is set to exactly 0 by FVCOM (no rounding uncertainty)
        unc = (SP_MIN_ULP // 2) * _ulp(mag)
        unc[0] = 0.0
        gap = np.abs(np.diff(sb.astype(np.float64)))
        lim = unc[:-1].astype(np.float64) + unc[1:].astype(np.float64)
        bad = np.flatnonzero(~(gap > lim))
        if bad.size:
            k = int(bad[0]) + 1
            raise ValueError(
                f"build_gtsz: in single precision (model_precision='single') the "
                f"upper-band interfaces {k} and {k + 1} (GTSZ BASE = {spec.base}) are "
                f"separated by {gap[k - 1]:.3g}, within {SP_MIN_ULP} float32 ulps of the "
                "values they are computed from; the runtime's tanhf/powf or the "
                "compiler can collapse them (a single-precision FVCOM then stops at "
                "startup in mod_setup.F SIGMA_GTSZ) -- use a less saturated shape "
                "(smaller GTSZ L1/L2 or P1), fewer upper layers (GTSZ K1) or a "
                "double-precision build"
            )
    # <--- jsasaki 2026-11-05
    return out


def _lowsig_thickness_sp(
    h: NDArray[np.float64],
    href: NDArray[np.float64],
    spec: GtszSpec,
    dtype: type[np.floating],
) -> NDArray[np.floating]:
    """Physical LOWSIG layer thicknesses ``(Z(K)-Z(K+1))*H``, shape ``(M, KB-1)``,
    evaluated in FVCOM's ``REAL(SP)`` precision ``dtype``.

    Port of the ``GTSZ_LOWSIG`` branch of ``mod_setup.F::SIGMA_GTSZ`` and of the
    ``GTSZ LOWFRAC`` reading in ``mod_input.F`` (conversion to ``SP``, positivity,
    ``|SUM-1| <= 1e-4``, renormalisation by ``SUM`` and the sequential ``FRCUM``
    accumulation), at ``ETA0 = 0``.  ``SBAND`` comes from :func:`_sband_sp`.
    """
    # jsasaki 2026-11-05: review round 10 #2: new helper (startup check in model precision)
    dt = dtype
    kb, k1 = spec.kb, spec.k1
    H = np.asarray(h, dtype=dt)
    hr = np.minimum(np.asarray(href, dtype=dt), H)
    if spec.dref > 0.0:
        hr = np.minimum(hr, dt(spec.dref))
    beta_up = dt(k1 - 1) / dt(kb - 1)
    dup = beta_up * H
    if spec.smooth > 0.0:
        sm = dt(spec.smooth)
        # jsasaki 2026-11-05: review round 12 #1: the C library's tanhf in float32
        w = dt(0.5) * (dt(1.0) + _tanh_sp((dup - hr) / sm, dt))
        dup = w * hr + (dt(1.0) - w) * dup
    else:
        dup = np.minimum(hr, dup)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sigref = -dup / H
        # jsasaki 2026-11-05: review round 11 #1: SBAND itself in model precision
        sband = _sband_sp(spec, dt)
        Z = np.empty((H.shape[0], kb), dtype=dt)
        Z[:, :k1] = sigref[:, None] * sband[None, :]
        if spec.lowfrac:
            lf = np.asarray(spec.lowfrac, dtype=dt)
            if not (np.isfinite(lf).all() and (lf > dt(0.0)).all()):
                raise ValueError(
                    "build_gtsz: a GTSZ LOWFRAC fraction is not positive in the "
                    f"model's {np.dtype(dt).name} precision; FVCOM stops at startup"
                )
            fsum = dt(0.0)
            for f in lf:
                fsum = dt(fsum + f)
            if abs(fsum - dt(1.0)) > dt(1.0e-4):
                raise ValueError(
                    "build_gtsz: GTSZ LOWFRAC fractions do not sum to 1 in the "
                    f"model's {np.dtype(dt).name} precision; FVCOM stops at startup"
                )
            lf = lf / fsum
            frcum = dt(0.0)
            for k in range(k1 + 1, kb + 1):
                frcum = dt(frcum + lf[k - k1 - 1])
                Z[:, k - 1] = sigref + (dt(-1.0) - sigref) * frcum
        else:
            for k in range(k1 + 1, kb + 1):
                Z[:, k - 1] = sigref + (dt(-1.0) - sigref) * (dt(k - k1) / dt(kb - k1))
        Z[:, 0] = dt(0.0)
        Z[:, kb - 1] = dt(-1.0)
        dz: NDArray[np.floating] = (Z[:, :-1] - Z[:, 1:]) * H[:, None]
    # ---> jsasaki 2026-11-05: review round 12 #1: a positive float32 separation of
    # only a few ulps of the interface values may vanish in the model (another
    # tanhf, operation order, fused multiply-add); non-positive ones are left to
    # the caller's message
    if dt is np.float32 and Z.size:
        gap = Z[:, :-1].astype(np.float64) - Z[:, 1:].astype(np.float64)
        ref = np.maximum(np.abs(Z[:, :-1]), np.abs(Z[:, 1:]))
        lim = SP_MIN_ULP * _ulp(ref).astype(np.float64)
        thin = (gap > 0.0) & ~(gap > lim)
        if thin.any():
            badn = np.flatnonzero(thin.any(axis=1))
            kk = int(np.flatnonzero(thin[badn[0]])[0]) + 1
            raise ValueError(
                f"build_gtsz: in single precision (model_precision='single') "
                f"GTSZ_LOWSIG interfaces {kk} and {kk + 1} are within {SP_MIN_ULP} "
                f"float32 ulps of each other at {badn.size} node(s) (first node "
                f"{int(badn[0]) + 1} 1-based); rounding differences of the model's "
                "runtime or compiler can collapse the layer and a single-precision "
                "FVCOM then stops at startup (mod_setup.F SIGMA_GTSZ) -- use thicker "
                "layers or a double-precision build"
            )
    # <--- jsasaki 2026-11-05
    return dz


def build_gtsz(
    mesh: MeshInputs, spec: GtszSpec, *, model_precision: str = "double"
) -> GtszCoordinate:
    """Build a :class:`~xfvcom.grid.gtsz.GtszCoordinate` over ``mesh`` for ``spec``.

    Computes ``H_ref`` (SADAPT only), the sigma field ``Z``, and the masking maps
    ``KBP`` / ``KBP1`` / ``FRAC1`` (when ``spec.mask``).

    ``model_precision`` is the ``REAL(SP)`` precision of the FVCOM build that will
    read the coordinate: ``"double"`` (default; the TB-FVCOM build, compiled with
    ``-DDOUBLE_PRECISION``) or ``"single"``.  FVCOM's LOWSIG startup checks
    (LOWFRAC normalisation and accumulation, every layer thicker than 0) are
    evaluated in that precision, so a layer that collapses only in a
    single-precision build is rejected here.  The returned coordinate is always
    float64.
    """
    # ---> jsasaki 2026-11-05: review round 10 #2: explicit model precision
    if model_precision not in _MODEL_DTYPES:
        raise ValueError(
            f"build_gtsz: model_precision must be one of {sorted(_MODEL_DTYPES)}, "
            f"got {model_precision!r}"
        )
    # <--- jsasaki 2026-11-05
    spec.validate()
    # ---> jsasaki 2026-11-05: review round 9 #1: the depths must be finite, and a
    # LOWSIG coordinate needs H > 0 at every node (FVCOM's domain-wide check below)
    h = np.asarray(mesh.h, dtype=np.float64)
    if h.shape != (mesh.n_node,) or not np.isfinite(h).all():
        bad = np.flatnonzero(~np.isfinite(h)) if h.ndim == 1 else np.array([0])
        raise ValueError(
            f"build_gtsz: mesh depths must be a finite ({mesh.n_node},) array; "
            f"{bad.size} non-finite value(s), first node {int(bad[0]) + 1} (1-based)"
            if bad.size
            else f"build_gtsz: mesh depths have shape {h.shape}, expected ({mesh.n_node},)"
        )
    if spec.lowsig and (h <= 0.0).any():
        bad = np.flatnonzero(h <= 0.0)
        raise ValueError(
            f"build_gtsz: GTSZ_LOWSIG needs positive depths, but {bad.size} node(s) "
            f"have H <= 0 (first node {int(bad[0]) + 1} 1-based, H = {h[bad[0]]:g} m); "
            "FVCOM stops at startup (mod_setup.F SIGMA_GTSZ: 'GTSZ_LOWSIG produced "
            "a non-positive layer thickness') -- use a depth file with H > 0"
        )
    # <--- jsasaki 2026-11-05
    href = None
    if spec.sadapt:
        href = grassfire_href(mesh.h, mesh.nv, mesh.x, mesh.y, spec.smax)
    Z = build_coordinate(mesh.h, spec, href=href)
    # ---> jsasaki 2026-11-05: review round 9 #1: mod_setup.F SIGMA_GTSZ checks the
    # thinnest physical LOWSIG layer over the whole domain, min((Z(K)-Z(K+1))*H),
    # and stops when it is <= 0 (not only at the open boundary)
    if spec.lowsig and h.size:
        dz = (Z[:, :-1] - Z[:, 1:]) * h[:, None]
        if not np.isfinite(dz).all() or float(dz.min()) <= 0.0:
            badn = np.flatnonzero(~(np.isfinite(dz) & (dz > 0.0)).all(axis=1))
            raise ValueError(
                f"build_gtsz: GTSZ_LOWSIG produced {int((~(dz > 0.0)).sum())} "
                f"non-positive layer thickness(es) at {badn.size} node(s) (first "
                f"node {int(badn[0]) + 1} 1-based, thinnest {float(np.nanmin(dz)):g} "
                "m); FVCOM stops at startup (mod_setup.F SIGMA_GTSZ) -- check H "
                "and GTSZ K1 / KB / LOWFRAC"
            )
    # <--- jsasaki 2026-11-05
    # ---> jsasaki 2026-11-05: review round 10 #2: a single-precision FVCOM build
    # evaluates the same LOWSIG construction in REAL(4), where a thin layer can
    # collapse to zero thickness (startup FATAL_ERROR) although it is positive here
    if spec.lowsig and h.size and model_precision == "single":
        assert href is not None  # LOWSIG requires SADAPT (validated)
        dz32 = _lowsig_thickness_sp(h, href, spec, _MODEL_DTYPES[model_precision])
        good = np.isfinite(dz32) & (dz32 > 0.0)
        if not good.all():
            badn = np.flatnonzero(~good.all(axis=1))
            raise ValueError(
                f"build_gtsz: in single precision (model_precision='single') GTSZ_LOWSIG "
                f"produces {int((~good).sum())} non-positive layer thickness(es) at "
                f"{badn.size} node(s) (first node {int(badn[0]) + 1} 1-based); a "
                "single-precision FVCOM stops at startup (mod_setup.F SIGMA_GTSZ) -- "
                "use thicker GTSZ LOWFRAC fractions or a double-precision build"
            )
    # <--- jsasaki 2026-11-05
    if spec.mask:
        kbp = compute_kbp(Z, spec.kbm1)
        # ---> jsasaki 2026-11-05: review round 8 #6: mod_setup.F SIGMA_GTSZ
        # refuses a masked column with fewer than two active layers (KBP = 1:
        # the vertical diffusion solvers need two; KBP = 0: division by zero)
        if kbp.size and int(kbp.min()) < 2:
            bad = np.flatnonzero(kbp < 2)
            raise ValueError(
                f"build_gtsz: {bad.size} GTSZ_MASK column(s) keep fewer than two "
                f"active layers (min KBP = {int(kbp.min())}, first node "
                f"{int(bad[0]) + 1} 1-based); FVCOM stops at startup -- raise "
                "GTSZ K1 or review ZLEV so every column keeps two active layers"
            )
        # <--- jsasaki 2026-11-05
        # jsasaki 2026-11-05: review round 8 #1: node depths for the depth-weighted FRAC1
        kbp1, frac1 = compute_kbp1_frac1(Z, mesh.nv, kbp, mesh.h)
    else:
        kbp = np.full(mesh.n_node, spec.kbm1, dtype=np.int64)
        kbp1 = np.full(mesh.n_elem, spec.kbm1, dtype=np.int64)
        frac1 = np.ones(mesh.n_elem, dtype=np.float64)
    return GtszCoordinate(
        spec=spec,
        H=np.asarray(mesh.h, dtype=np.float64),
        Z=Z,
        kbp=kbp,
        href=href,
        nv=mesh.nv,
        x=mesh.x,
        y=mesh.y,
        kbp1=kbp1,
        frac1=frac1,
    )
