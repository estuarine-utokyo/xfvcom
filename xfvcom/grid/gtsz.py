"""FVCOM generalized terrain-following sigma-z ("GTSZ" / "B-sandwich") vertical
coordinate -- a bit-faithful Python port of ``SIGMA_GTSZ`` (FVCOM ``src/mod_setup.F``).

This is the *coordinate model*: given a per-node depth ``H`` and a set of header
knobs (the ``<casename>_sigma.dat`` ``GTSZ ...`` keys), it reproduces -- column by
column, exactly as the Fortran does -- the non-dimensional sigma field
``Z(node, K) in [0, -1]`` (fraction of ``H+eta``; physical depth ``= Z*(H+eta)``).

Two coordinate families are supported, matching the Fortran's two code paths:

* **Base / B-sandwich** (``GTSZ_SADAPT = F``): a terrain-following base stretch
  (A1 uniform / A2 double-exponential / A3 hyperbolic) optionally with a *fixed*
  middle z-band between layers ``K1`` and ``K2`` (Bu et al. 2025). The z-band is
  the hard ``max(SA, S4)`` (``GTSZ_SMOOTH=0``), a physical-depth tanh blend
  (``GTSZ_SMOOTH>0``), or -- with ``GTSZ_MASK=T`` -- aligned fixed-depth z-levels
  with a degenerate (killed) at-bed tail. Reference Python:
  ``FVCOM/Tests/MultiSigma/bu_shallow/check_coord.py``.

* **Slope-adaptive** (``GTSZ_SADAPT = T``, *requires* ``GTSZ_MASK = T``): the
  reference-surface hybrid. A smooth gentle lower envelope ``H_ref(x) <= H``
  (``|grad H_ref| <= SMAX``, built by :func:`xfvcom.grid.gtsz_builder.grassfire_href`)
  carries ``K1`` terrain-following sigma layers; the fixed z-levels that fall
  strictly between ``-H_ref`` and the bed are **appended in order** (concatenated,
  *not* a per-K cap) so the real layers stay contiguous from the surface and
  ``DZ > 0`` everywhere. Gentle (``H_ref = H``) -> pure sigma; steep/deep
  (``H_ref < H``) -> a z-staircase that removes the sigma-PGE by construction.
  Reference Python (design, H_ref grassfire): ``FVCOM/Tests/SlopeAdaptive/prototype_sadapt.py``.

The authoritative spec is ``FVCOM/src/mod_setup.F::SIGMA_GTSZ`` and the
``<casename>_sigma.dat`` parser ``FVCOM/src/mod_input.F`` (``READ_COLDSTART_SIGMA``,
``case(STYPE_SIGMAZ)``); design docs ``FVCOM/docs/sigma-z-slope-adaptive-design.md``
and ``FVCOM/docs/sigma-z-physics-consistency.md``.

The coordinate is evaluated at ``eta = 0`` (cold-start at-rest), so the result
matches the ``siglay`` / ``siglev`` written into an FVCOM output NetCDF (the
bit-faithful check). FVCOM uses single precision internally; this port uses
float64, so agreement is to single-precision round-off (~1e-5), not bit-exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

# --- Fortran constants (mod_setup.F::SIGMA_GTSZ) ----------------------------
SMALL = 1.0e-6  #: degenerate-tail ramp step (Fortran ``SMALL``)
DZ_MIN_KBP = 1.0e-4  #: sigma-thickness below this => killed (degenerate) layer

__all__ = [
    "SMALL",
    "DZ_MIN_KBP",
    "GtszSpec",
    "base_sigma",
    "sband_reshape",
    "build_column_base",
    "build_column_sadapt",
    "build_coordinate",
    "compute_kbp",
    "element_average",
    "compute_kbp1_frac1",
    "GtszCoordinate",
]


# ===========================================================================
#  Header spec (one ``<casename>_sigma.dat`` SIGMAZ block)
# ===========================================================================
@dataclass(frozen=True)
class GtszSpec:
    """The ``GTSZ ...`` knobs of a SIGMAZ ``<casename>_sigma.dat`` header.

    Field names mirror the Fortran module variables / the ``GTSZ <KEY>`` header
    lines (see :mod:`xfvcom.io.sigma_dat` for the file reader/writer). Defaults
    match the ``mod_input.F`` ``SCAN_FILE`` defaults so an omitted key behaves
    exactly as in FVCOM.
    """

    kb: int  #: ``NUMBER OF SIGMA LEVELS`` (levels; KB-1 layers)
    base: int = 2  #: ``GTSZ BASE``: 1=A1 uniform, 2=A2 double-exp, 3=A3 hyperbolic
    k1: int = 1  #: ``GTSZ K1``: surface sigma band (K<=K1)
    k2: int | None = None  #: ``GTSZ K2``: bottom sigma band (K>=K2); default KB
    nz: int = 0  #: ``GTSZ NZ``: number of fixed z-levels (0 = pure base sigma)
    zlev: tuple[
        float, ...
    ] = ()  #: ``GTSZ ZLEV``: NZ levels, negative, strictly descending
    p1: float = 2.0  #: ``GTSZ P1`` (A2 exponent)
    l1: float = 1.0  #: ``GTSZ L1`` (A3)
    l2: float = 1.0  #: ``GTSZ L2`` (A3)
    smooth: float = 0.0  #: ``GTSZ SMOOTH`` z<->sigma blend width [m]; 0 = hard max
    mask: bool = False  #: ``GTSZ MASK`` (variable active layers + lateral wall)
    sadapt: bool = False  #: ``GTSZ SADAPT`` (slope-adaptive reference surface)
    smax: float = (
        0.0  #: ``GTSZ SMAX`` max sigma-slope for H_ref [m/m]; <=0 => flat min(H)
    )
    dye_nowall: bool = False  #: ``GTSZ DYE_NOWALL`` (diagnostic)
    bpg_ref_zlev: tuple[
        float, ...
    ] = ()  #: ``BPG REF ZLEV`` (optional, RHO_PMEAN reference)

    def __post_init__(self) -> None:
        if self.k2 is None:
            object.__setattr__(self, "k2", self.kb)
        object.__setattr__(self, "zlev", tuple(float(z) for z in self.zlev))
        object.__setattr__(
            self, "bpg_ref_zlev", tuple(float(z) for z in self.bpg_ref_zlev)
        )

    @property
    def kbm1(self) -> int:
        return self.kb - 1

    @property
    def have_zband(self) -> bool:
        return self.nz > 0

    def validate(self) -> None:
        """Replicate the Fortran startup ``FATAL_ERROR`` checks (mod_setup.F /
        mod_input.F), so a malformed spec is rejected here rather than at FVCOM
        startup."""
        if self.kb < 4:
            raise ValueError(f"GtszSpec: KB={self.kb} too small (need >= 4)")
        if self.base not in (1, 2, 3):
            raise ValueError(f"GtszSpec: GTSZ BASE must be 1/2/3, got {self.base}")
        if not (0 <= self.nz <= 500):
            raise ValueError(f"GtszSpec: require 0 <= GTSZ NZ <= 500, got {self.nz}")
        if len(self.zlev) != self.nz:
            raise ValueError(f"GtszSpec: len(ZLEV)={len(self.zlev)} != NZ={self.nz}")
        # mod_input.F:5166-5173 ZLEV contract (checked at parse time in FVCOM,
        # i.e. BEFORE the K1/K2/NZ setup checks below): negative, strictly descending.
        if self.zlev:
            z = np.asarray(self.zlev, dtype=float)
            if np.any(z >= 0.0):
                raise ValueError("GtszSpec: all GTSZ ZLEV must be negative")
            if np.any(np.diff(z) >= 0.0):
                raise ValueError("GtszSpec: GTSZ ZLEV must be strictly descending")
        if self.have_zband:
            assert self.k2 is not None  # always set in __post_init__ (None -> kb)
            # mod_setup.F:1072-1075 (only enforced when a z-band exists)
            if not (1 <= self.k1 and self.k1 + 1 < self.k2 <= self.kb):
                raise ValueError(
                    "GtszSpec: require 1 <= GTSZ K1, GTSZ K1+1 < GTSZ K2 <= KB "
                    f"(K1={self.k1}, K2={self.k2}, KB={self.kb})"
                )
            if self.nz < (self.k2 - self.k1 - 1):
                raise ValueError(
                    "GtszSpec: GTSZ NZ < (K2-K1-1): too few z-levels for the z-band "
                    f"(NZ={self.nz}, K2-K1-1={self.k2 - self.k1 - 1})"
                )
        if self.sadapt:
            # mod_setup.F:1143-1146
            if not self.mask:
                raise ValueError(
                    "GtszSpec: GTSZ SADAPT=T requires GTSZ MASK=T (the z-region uses the wall)"
                )
            if not self.have_zband:
                raise ValueError(
                    "GtszSpec: GTSZ SADAPT=T requires a z-band (GTSZ NZ > 0)"
                )

    def with_(self, **kw) -> "GtszSpec":
        """Return a copy with fields replaced (e.g. for a parameter sweep)."""
        return replace(self, **kw)


# ===========================================================================
#  Base stretches + surface-band reshape
# ===========================================================================
def base_sigma(
    kb: int, base: int = 2, *, p1: float = 2.0, l1: float = 1.0, l2: float = 1.0
) -> NDArray[np.float64]:
    """Base terrain-following stretch ``SA(K)`` for ``K = 1 .. KB`` (returned as a
    length-``KB`` array, index ``k`` = Fortran ``K = k+1``).

    Exact port of ``mod_setup.F::SIGMA_GTSZ`` lines 1263-1278:

    * ``base=1`` (A1 uniform): ``SA = (1-K)/(KB-1)``
    * ``base=2`` (A2 double-exp, ``P1``): split at ``KHALF=(KB+1)//2``
    * ``base=3`` (A3 hyperbolic, ``L1``/``L2``)

    All bases give ``SA(1)=0`` and ``SA(KB)=-1``.
    """
    if base not in (1, 2, 3):
        raise ValueError(f"base_sigma: GTSZ BASE must be 1/2/3, got {base}")
    K: NDArray[np.float64] = np.arange(1, kb + 1, dtype=np.float64)
    khalf = (kb + 1) // 2
    if base == 1:
        sa = (1.0 - K) / (kb - 1.0)
    elif base == 2:
        sa = np.empty(kb, dtype=np.float64)
        upper = K <= khalf
        sa[upper] = -0.5 * (((2.0 * K[upper] - 2.0) / (kb - 1.0)) ** p1)
        sa[~upper] = 0.5 * (((2.0 * kb - 2.0 * K[~upper]) / (kb - 1.0)) ** p1) - 1.0
    else:  # base == 3
        sa = (np.tanh(((1.0 - K) * l1 + (kb - K) * l2) / (kb - 1.0)) - np.tanh(l2)) / (
            np.tanh(l1) + np.tanh(l2)
        )
    return sa


def sband_reshape(
    kb: int,
    k1: int,
    base: int = 2,
    *,
    p1: float = 2.0,
    l1: float = 1.0,
    l2: float = 1.0,
) -> NDArray[np.float64]:
    """Surface-sigma-band reshape ``SBAND(K)`` for ``K = 1 .. K1`` (length-``K1``
    array), normalized so ``SBAND(1)=0`` and ``SBAND(K1)=1``.

    Port of ``mod_setup.F::SIGMA_GTSZ`` lines 1204-1225: take the base shape over
    ``1..K1`` and divide by ``SBAND(K1)`` (with the ``|SAK1|<SMALL -> -SMALL``
    guard). Used by the SADAPT path to spread ``K1`` sigma layers over ``[0,
    sigma_ref]``.
    """
    sb = base_sigma(kb, base, p1=p1, l1=l1, l2=l2)[:k1].copy()
    sak1 = sb[k1 - 1]
    if abs(sak1) < SMALL:
        sak1 = -SMALL
    return sb / sak1


# ===========================================================================
#  Per-column builders
# ===========================================================================
def build_column_base(
    H: float, spec: GtszSpec, *, eta0: float = 0.0
) -> NDArray[np.float64]:
    """Build one node's column for the **non-SADAPT** path (pure base sigma, or
    the fixed K1<K<K2 B-sandwich; hard-max / SMOOTH-blend / MASK fixed-z).

    Exact port of ``mod_setup.F::SIGMA_GTSZ`` lines 1261-1334.
    """
    kb = spec.kb
    sa = base_sigma(kb, spec.base, p1=spec.p1, l1=spec.l1, l2=spec.l2)  # SA(1..KB)
    z: NDArray[np.float64] = np.empty(kb, dtype=np.float64)
    D = H + eta0
    have_z = spec.have_zband
    k1, k2 = spec.k1, spec.k2
    assert k2 is not None  # always set in __post_init__
    zlev = np.asarray(spec.zlev, dtype=np.float64)
    for k in range(1, kb + 1):  # Fortran K
        sak = sa[k - 1]
        if (not have_z) or k <= k1 or k >= k2:
            zk = sak
        else:
            kz = k - k1  # 1 .. K2-K1-1
            s4 = (zlev[kz - 1] - eta0) / D
            if spec.mask:
                zk = max(s4, -1.0)
            elif spec.smooth > 0.0:
                wblend = 0.5 * (1.0 + np.tanh((s4 - sak) * D / spec.smooth))
                zk = wblend * s4 + (1.0 - wblend) * sak
            else:
                zk = max(sak, s4)
        # robustness guard: strict monotonic decrease (skip for MASK kill-tail)
        if k > 1 and not spec.mask:
            if zk >= z[k - 2]:
                zk = z[k - 2] - SMALL
        z[k - 1] = zk
    # masking: ramp the killed at-bed tail monotonically down to -1
    if spec.mask:
        kbed = 1
        for k in range(1, kb + 1):
            if z[k - 1] > -1.0 + DZ_MIN_KBP:
                kbed = k
        for k in range(kbed + 1, kb + 1):
            z[k - 1] = -1.0 + (kb - k) * SMALL
    z[0] = 0.0
    z[kb - 1] = -1.0
    return z


def build_column_sadapt(
    H: float, href: float, spec: GtszSpec, *, eta0: float = 0.0
) -> NDArray[np.float64]:
    """Build one node's column for the **slope-adaptive** path (concatenated:
    ``K1`` sigma layers over ``[0, -H_ref]`` + the fixed z-levels strictly below
    ``-H_ref`` and above the bed, appended in order, + a degenerate tail).

    Exact port of ``mod_setup.F::SIGMA_GTSZ`` lines 1239-1258.
    """
    kb = spec.kb
    k1 = spec.k1
    zlev = np.asarray(spec.zlev, dtype=np.float64)
    sband = sband_reshape(kb, k1, spec.base, p1=spec.p1, l1=spec.l1, l2=spec.l2)
    z: NDArray[np.float64] = np.empty(kb, dtype=np.float64)
    D = H + eta0
    sigref = -(href + eta0) / D
    sigref = max(sigref, -1.0 + kb * SMALL)  # leave room for the tail
    for k in range(1, k1 + 1):
        z[k - 1] = sigref * sband[k - 1]  # surface sigma band over [0, sigref]
    kbed = k1
    for kz in range(1, spec.nz + 1):
        s4 = (zlev[kz - 1] - eta0) / D
        if (s4 < sigref - DZ_MIN_KBP) and (s4 > -1.0 + DZ_MIN_KBP):
            kbed += 1
            if kbed <= kb - 1:
                z[kbed - 1] = s4  # real z-level below -H_ref
    for k in range(kbed + 1, kb + 1):
        z[k - 1] = -1.0 + (kb - k) * SMALL  # degenerate tail -> bed
    z[0] = 0.0
    z[kb - 1] = -1.0
    return z


# ===========================================================================
#  Whole-grid coordinate
# ===========================================================================
def build_coordinate(
    H: NDArray[np.float64],
    spec: GtszSpec,
    *,
    href: NDArray[np.float64] | None = None,
    eta0: float = 0.0,
) -> NDArray[np.float64]:
    """Build the per-node sigma field ``Z(M, KB)`` for every node.

    Parameters
    ----------
    H : (M,) array
        Per-node depth (positive, metres).
    spec : GtszSpec
        The coordinate knobs.
    href : (M,) array, optional
        Per-node reference depth ``H_ref`` (required when ``spec.sadapt``; build
        it with :func:`xfvcom.grid.gtsz_builder.grassfire_href`).

    Returns
    -------
    Z : (M, KB) array
        ``Z in [0, -1]`` per node, evaluated at ``eta = eta0`` (default 0).
    """
    spec.validate()
    H = np.asarray(H, dtype=np.float64)
    M = H.shape[0]
    Z = np.empty((M, spec.kb), dtype=np.float64)
    if spec.sadapt:
        if href is None:
            raise ValueError("build_coordinate: spec.sadapt=True requires href")
        href = np.asarray(href, dtype=np.float64)
        href = np.minimum(href, H)  # safety: H_ref <= H (mod_setup.F:1202)
        for i in range(M):
            Z[i] = build_column_sadapt(H[i], href[i], spec, eta0=eta0)
    else:
        # the base / B-sandwich path is per-column independent; loop for a faithful
        # match to the Fortran (M ~ a few thousand -> sub-second).
        for i in range(M):
            Z[i] = build_column_base(H[i], spec, eta0=eta0)
    return Z


def compute_kbp(Z: NDArray[np.float64], kbm1: int | None = None) -> NDArray[np.int64]:
    """Per-node active-LAYER count ``GTSZ_KBP`` = number of contiguous
    real-thickness layers from the surface (``DZ > DZ_MIN_KBP``).

    Port of ``mod_setup.F::SIGMA_GTSZ`` lines 1346-1355. ``KBP = KBM1`` for a
    full (un-killed) column.
    """
    Z = np.asarray(Z, dtype=np.float64)
    kb = Z.shape[1]
    if kbm1 is None:
        kbm1 = kb - 1
    dz = Z[:, :kbm1] - Z[:, 1 : kbm1 + 1]  # (M, KBM1): layer K=1..KBM1
    killed = dz < DZ_MIN_KBP
    kbp = np.full(Z.shape[0], kbm1, dtype=np.int64)
    any_killed = killed.any(axis=1)
    # first killed layer index k (0-based) => Fortran KBP = k
    first = np.argmax(killed, axis=1)
    kbp[any_killed] = first[any_killed]
    return kbp


def element_average(
    Z: NDArray[np.float64], nv: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Element-centred sigma ``Z1(N, KB)`` = mean of the 3 node columns.

    ``nv`` is the (N, 3) or (3, N) **0-based** element->node table.
    """
    nv = np.asarray(nv)
    if nv.shape[0] == 3 and nv.shape[1] != 3:
        nv = nv.T  # accept (3, N) too
    return Z[nv].mean(axis=1)


def compute_kbp1_frac1(
    Z: NDArray[np.float64], nv: NDArray[np.int64], kbp: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Per-element active-layer count ``GTSZ_KBP1`` (= min of the element's 3
    node KBP) and active sigma-fraction ``GTSZ_FRAC1 = -Z1(KBP1+1)``.

    Port of ``mod_setup.F::SIGMA_GTSZ`` lines 1400-1407.
    """
    nv = np.asarray(nv)
    if nv.shape[0] == 3 and nv.shape[1] != 3:
        nv = nv.T
    Z1 = element_average(Z, nv)  # (N, KB)
    kbp1 = kbp[nv].min(axis=1)  # (N,)
    # Fortran FRAC1 = -Z1(I, KBP1+1) (1-indexed level KBP1+1 == 0-indexed KBP1,
    # since KBP1 is the layer count). Z1(1)=0 -> FRAC1=1 for a full column.
    frac1 = -Z1[np.arange(Z1.shape[0]), kbp1]
    return kbp1.astype(np.int64), frac1.astype(np.float64)


@dataclass
class GtszCoordinate:
    """A built sigma-z coordinate over a mesh: the sigma field + masking maps.

    Produced by :func:`xfvcom.grid.gtsz_builder.build_gtsz`. Holds everything a
    diagnostic or writer needs.
    """

    spec: GtszSpec
    H: NDArray[np.float64]  #: (M,) node depth
    Z: NDArray[np.float64]  #: (M, KB) node sigma
    kbp: NDArray[np.int64]  #: (M,) node active-layer count
    href: NDArray[np.float64] | None = None  #: (M,) reference depth (SADAPT only)
    nv: NDArray[np.int64] | None = None  #: (N, 3) 0-based connectivity
    x: NDArray[np.float64] | None = None  #: (M,) node x
    y: NDArray[np.float64] | None = None  #: (M,) node y
    kbp1: NDArray[np.int64] | None = field(
        default=None
    )  #: (N,) element active-layer count
    frac1: NDArray[np.float64] | None = field(
        default=None
    )  #: (N,) active sigma-fraction

    @property
    def z_phys(self) -> NDArray[np.float64]:
        """Physical depth ``Z * H`` (m, negative down) at ``eta = 0``."""
        return self.Z * self.H[:, None]

    @property
    def dz(self) -> NDArray[np.float64]:
        """Per-node layer sigma-thickness ``Z[K]-Z[K+1]`` (M, KB-1)."""
        return self.Z[:, :-1] - self.Z[:, 1:]

    @property
    def n_reduced(self) -> int:
        """Number of nodes with a reduced (masked) column (``KBP < KBM1``)."""
        return int((self.kbp < self.spec.kbm1).sum())
