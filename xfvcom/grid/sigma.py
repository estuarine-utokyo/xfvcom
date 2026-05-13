# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Sigma-coordinate / z-coordinate conversion helpers.

These functions are deliberately small and side-effect free; they are the
shared vocabulary used by, e.g., the JCOPE-T DA → FVCOM OBC generator
and any other code that needs to move between σ and z.

Sign convention used throughout: ``z`` is **negative downward** (0 at the
free surface, ``-h`` at the seabed). σ is in ``[-1, 0]`` with σ = 0 at the
surface and σ = -1 at the bottom.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "z_from_sigma",
    "sigma_from_z",
    "vertical_interp",
]


def z_from_sigma(
    sigma: NDArray | float,
    h: NDArray | float,
    eta: NDArray | float = 0.0,
) -> NDArray:
    """Return depth ``z = sigma * (h + eta)`` (negative downward).

    Broadcasting follows NumPy rules; pass ``sigma`` as the leading axis
    (e.g. ``(km,)`` or ``(km, n_node)``) and ``h``/``eta`` as the trailing
    axes (e.g. ``(n_node,)``) for the common per-station case.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    return sigma * (h + eta)


def sigma_from_z(
    z: NDArray | float,
    h: NDArray | float,
    eta: NDArray | float = 0.0,
) -> NDArray:
    """Return ``sigma = z / (h + eta)``; the inverse of :func:`z_from_sigma`.

    The result is **not** clipped to ``[-1, 0]``; the caller decides what
    to do with depths that fall outside the water column.
    """
    z = np.asarray(z, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    return z / (h + eta)


def vertical_interp(
    profile: NDArray,
    src_z: NDArray,
    dst_z: NDArray,
) -> NDArray:
    """Linear interpolation of a 1-D vertical profile in z-space.

    Parameters
    ----------
    profile
        ``(n_src,)`` array of values defined at ``src_z``. NaN entries
        are treated as missing and skipped.
    src_z
        ``(n_src,)`` source depths (negative downward). Need not be
        sorted; this function will sort internally.
    dst_z
        ``(n_dst,)`` target depths (negative downward).

    Returns
    -------
    ndarray
        ``(n_dst,)`` array of float64 interpolated values. Targets that
        fall above the shallowest valid source depth or below the
        deepest valid source depth take the nearest valid value
        (constant extrapolation) — the conservative choice for ocean
        boundary forcing where T/S/U/V vary smoothly near the limits.
    """
    profile_arr = np.asarray(profile, dtype=np.float64)
    if hasattr(profile, "filled"):
        profile_arr = profile.filled(np.nan)  # type: ignore[union-attr]
    src_z_arr = np.asarray(src_z, dtype=np.float64)
    dst_z_arr = np.asarray(dst_z, dtype=np.float64)

    if profile_arr.shape != src_z_arr.shape:
        raise ValueError(
            f"profile shape {profile_arr.shape} != src_z shape {src_z_arr.shape}"
        )

    valid = np.isfinite(profile_arr) & np.isfinite(src_z_arr)
    if valid.sum() == 0:
        return np.full(dst_z_arr.shape, np.nan, dtype=np.float64)
    if valid.sum() == 1:
        return np.full(dst_z_arr.shape, profile_arr[valid][0], dtype=np.float64)

    vz = src_z_arr[valid]
    vp = profile_arr[valid]
    # np.interp requires monotonically increasing x — sort by depth ascending
    # (most negative first, i.e. bottom to surface for the negative-downward
    # convention used here).
    order = np.argsort(vz)
    return np.interp(dst_z_arr, vz[order], vp[order])
