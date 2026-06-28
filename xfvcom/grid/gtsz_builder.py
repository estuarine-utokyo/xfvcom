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

from dataclasses import dataclass
from pathlib import Path

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
    if nv.shape[0] == 3 and nv.shape[1] != 3:
        nv = nv.T  # -> (N, 3)
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
    for sweeps in range(1, max_sweep + 1):
        before = href.copy()
        np.minimum.at(href, b, href[a] + w)  # relax a -> b
        np.minimum.at(href, a, href[b] + w)  # relax b -> a (sees the a->b update)
        if float(np.max(before - href)) < tol:
            break
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
    # ensure the last level is at least as deep as the bed
    if levels[-1] > -float(h_max):
        levels.append(-float(h_max))
    return np.asarray(levels, dtype=np.float64)


# ===========================================================================
#  Assemble the coordinate
# ===========================================================================
def build_gtsz(mesh: MeshInputs, spec: GtszSpec) -> GtszCoordinate:
    """Build a :class:`~xfvcom.grid.gtsz.GtszCoordinate` over ``mesh`` for ``spec``.

    Computes ``H_ref`` (SADAPT only), the sigma field ``Z``, and the masking maps
    ``KBP`` / ``KBP1`` / ``FRAC1`` (when ``spec.mask``).
    """
    spec.validate()
    href = None
    if spec.sadapt:
        href = grassfire_href(mesh.h, mesh.nv, mesh.x, mesh.y, spec.smax)
    Z = build_coordinate(mesh.h, spec, href=href)
    if spec.mask:
        kbp = compute_kbp(Z, spec.kbm1)
        kbp1, frac1 = compute_kbp1_frac1(Z, mesh.nv, kbp)
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
