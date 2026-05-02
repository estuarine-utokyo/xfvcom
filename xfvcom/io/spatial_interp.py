"""Spatial interpolation helpers for sparse station data onto FVCOM mesh.

Used to project a small set of meteorological observations (typically the
3-station MPOS network in inner Tokyo Bay) onto the unstructured FVCOM mesh
nodes (``node`` dimension) and elements (``nele`` dimension).

The default scheme is inverse-distance weighting (IDW) with power 2 because:

- With only 3-4 source stations a triangulation degenerates to "nearest" or
  "linear-on-triangle" with no smoothness, which is a poor match for a
  spatially-coherent quantity like wind.
- IDW gives a smooth, non-extrapolating field that is exact at the source
  stations and interpolates monotonically between them.

Distances are computed with the haversine formula on the spherical earth so
the helpers work directly on geographic coordinates without requiring the
caller to project to UTM. For Tokyo Bay the distance error of treating the
earth as locally flat is < 0.1 % at 50 km, well within the noise of the
station data, but the haversine cost is negligible.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

EARTH_RADIUS_KM: Final[float] = 6371.0088


def haversine_km(
    lon1: NDArray[np.float64],
    lat1: NDArray[np.float64],
    lon2: NDArray[np.float64],
    lat2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Great-circle distance between two coordinate arrays.

    Parameters
    ----------
    lon1, lat1 : ndarray
        First coordinate set, broadcastable shape.
    lon2, lat2 : ndarray
        Second coordinate set, broadcastable shape.

    Returns
    -------
    ndarray
        Distances in kilometres, shape from broadcasting.
    """
    rlon1 = np.deg2rad(lon1)
    rlon2 = np.deg2rad(lon2)
    rlat1 = np.deg2rad(lat1)
    rlat2 = np.deg2rad(lat2)
    dlon = rlon2 - rlon1
    dlat = rlat2 - rlat1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_RADIUS_KM * c


def idw_weights(
    src_lon: NDArray[np.float64],
    src_lat: NDArray[np.float64],
    tgt_lon: NDArray[np.float64],
    tgt_lat: NDArray[np.float64],
    *,
    power: float = 2.0,
    eps_km: float = 1.0e-3,
) -> NDArray[np.float64]:
    """Pre-compute IDW weight matrix from source to target points.

    Parameters
    ----------
    src_lon, src_lat : ndarray, shape (n_src,)
        Source-point coordinates (degrees).
    tgt_lon, tgt_lat : ndarray, shape (n_tgt,)
        Target-point coordinates (degrees).
    power : float, optional
        IDW power (default 2.0).
    eps_km : float, optional
        Numerical floor on distance to avoid division by zero when a target
        coincides with a source. Targets within this distance are treated
        as exactly coincident with the nearest source.

    Returns
    -------
    ndarray, shape (n_tgt, n_src)
        Row-normalised weights; each row sums to 1.
    """
    src_lon = np.asarray(src_lon, dtype=np.float64).reshape(1, -1)
    src_lat = np.asarray(src_lat, dtype=np.float64).reshape(1, -1)
    tgt_lon = np.asarray(tgt_lon, dtype=np.float64).reshape(-1, 1)
    tgt_lat = np.asarray(tgt_lat, dtype=np.float64).reshape(-1, 1)

    d = haversine_km(src_lon, src_lat, tgt_lon, tgt_lat)  # (n_tgt, n_src)
    coincident = d < eps_km

    with np.errstate(divide="ignore"):
        w = 1.0 / np.power(np.maximum(d, eps_km), power)

    # If a target is coincident with one or more sources, give those sources
    # all the weight and zero to the rest. This is the IDW limit and avoids
    # spurious contributions from far-away stations when the target sits
    # exactly on an observation point.
    has_match = coincident.any(axis=1)
    if has_match.any():
        match_w = coincident.astype(np.float64)
        w = np.where(has_match[:, None], match_w, w)

    row_sum = w.sum(axis=1, keepdims=True)
    return w / row_sum


def idw_apply(
    weights: NDArray[np.float64],
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply pre-computed IDW weights to a station-time array.

    NaNs in ``values`` are handled by re-normalising the weights over the
    available stations on a per-time basis. If all sources at a given time
    are NaN, the result is NaN at every target.

    Parameters
    ----------
    weights : ndarray, shape (n_tgt, n_src)
        Row-normalised weight matrix from :func:`idw_weights`.
    values : ndarray, shape (n_time, n_src) or (n_src,)
        Source values per time and station.

    Returns
    -------
    ndarray
        Interpolated values, shape (n_time, n_tgt) or (n_tgt,).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        # Single time-step shortcut.
        return idw_apply(weights, values[None, :])[0]

    if values.shape[1] != weights.shape[1]:
        raise ValueError(
            f"values has {values.shape[1]} stations but weights expect "
            f"{weights.shape[1]}"
        )

    valid = ~np.isnan(values)  # (n_time, n_src)
    # Effective weights per time-step: zero out where source is NaN, then
    # row-normalise across surviving stations. Use clean values (NaN -> 0)
    # in the weighted sum.
    w_eff = weights[None, :, :] * valid[:, None, :]  # (n_time, n_tgt, n_src)
    w_sum = w_eff.sum(axis=2, keepdims=True)
    safe_sum = np.where(w_sum > 0, w_sum, 1.0)
    w_norm = w_eff / safe_sum

    clean = np.where(valid, values, 0.0)
    out = np.einsum("tns,ts->tn", w_norm, clean)

    # NaN for time steps where every station was NaN.
    no_data = w_sum.squeeze(-1) == 0.0
    if no_data.any():
        out[no_data] = np.nan
    return out
