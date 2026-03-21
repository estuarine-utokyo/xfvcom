"""Vertical coordinate utilities for FVCOM sigma coordinates.

This module provides functions for working with FVCOM's sigma coordinate system,
including depth calculations at sigma layers and depth-range averaging.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    pass


def compute_depth_at_siglay(siglay: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Compute depth at each sigma layer for each node.

    Parameters
    ----------
    siglay : np.ndarray
        Sigma layer values, shape (nsiglay,). Values are negative
        (0 at surface, -1 at bottom).
    H : np.ndarray
        Total water column depth at each node, shape (nnode,).

    Returns
    -------
    np.ndarray
        Depth at each sigma layer and node, shape (nsiglay, nnode).
        Depth is positive downward from surface.

    Examples
    --------
    >>> siglay = np.array([-0.05, -0.15, -0.25])  # 3 sigma layers
    >>> H = np.array([10.0, 20.0])  # 2 nodes with depths 10m and 20m
    >>> depths = compute_depth_at_siglay(siglay, H)
    >>> depths.shape
    (3, 2)
    >>> depths[:, 0]  # Depths at node 0 (10m water column)
    array([0.5, 1.5, 2.5])
    """
    # depth = -sigma * H
    # siglay is (nsiglay,), H is (nnode,)
    # Result should be (nsiglay, nnode)
    return -siglay[:, np.newaxis] * H[np.newaxis, :]


def compute_depth_range_average(
    data: xr.DataArray,
    ds: xr.Dataset,
    depth_min: float | str,
    depth_max: float | str,
    is_hab: bool = False,
    verbose: bool = True,
) -> xr.DataArray:
    """Compute depth-range averaged values for all nodes and times.

    Vectorized implementation for performance. Computes the average of values
    within a specified depth range at each node.

    Parameters
    ----------
    data : xr.DataArray
        Data with dims (time, siglay, node) or (siglay, node)
    ds : xr.Dataset
        Dataset containing 'siglay' and 'h' (bathymetry) variables
    depth_min : float or str
        Minimum depth in meters, or "surface" for 0
    depth_max : float or str
        Maximum depth in meters, or "bottom" for full water column
    is_hab : bool
        If True, interpret as Height Above Bottom range.
        E.g., depth_min=0, depth_max=2 means bottom 2 meters.
    verbose : bool
        If True, print progress messages

    Returns
    -------
    xr.DataArray
        Depth-averaged values with dims (time, node) or (node,)

    Notes
    -----
    When depth_min is "surface" and water depth is shallower than depth_max,
    the average is computed from surface to bottom (full water column).

    Examples
    --------
    >>> # Average from surface to 2m depth
    >>> avg = compute_depth_range_average(dye_data, ds, "surface", 2)

    >>> # Average from 5m to 10m depth
    >>> avg = compute_depth_range_average(dye_data, ds, 5, 10)

    >>> # Average bottom 2m (Height Above Bottom)
    >>> avg = compute_depth_range_average(dye_data, ds, 0, 2, is_hab=True)
    """
    if verbose:
        print("    Computing depth-range average (vectorized)...", flush=True)

    # Get sigma layer values
    siglay = ds["siglay"].values
    if siglay.ndim == 2:
        siglay = siglay.mean(axis=1)  # (siglay,)
    h = ds["h"].values  # (node,)

    # Handle input with or without time dimension
    has_time = "time" in data.dims
    if has_time:
        n_time = data.shape[0]
        n_siglay = data.shape[1]
        n_node = data.shape[2]
    else:
        n_siglay = data.shape[0]
        n_node = data.shape[1]

    # Compute depths at sigma layers: depths[siglay, node]
    depths_all = compute_depth_at_siglay(siglay, h)  # (siglay, node)

    # Parse depth range bounds
    d_min_val = np.zeros(n_node)  # (node,)
    d_max_val = h.copy()  # (node,)

    if is_hab:
        # Height Above Bottom mode
        if depth_max != "bottom":
            d_min_val = h - float(depth_max)
        if depth_min != "surface" and depth_min != 0:
            d_max_val = h - float(depth_min)
    else:
        # Normal depth mode
        if depth_min != "surface":
            d_min_val[:] = float(depth_min)
        if depth_max != "bottom":
            d_max_val = np.minimum(d_max_val, float(depth_max))

    d_min_val = np.maximum(d_min_val, 0.0)  # Ensure non-negative

    # Create mask for sigma layers within depth range: (siglay, node)
    layer_in_range = (depths_all >= d_min_val[np.newaxis, :]) & (
        depths_all <= d_max_val[np.newaxis, :]
    )

    # Get data values
    data_values = data.values

    if has_time:
        # Expand mask to time dimension: (time, siglay, node)
        mask = np.broadcast_to(layer_in_range[np.newaxis, :, :], data_values.shape)

        # Compute masked average along siglay dimension
        masked_values = np.where(mask, data_values, np.nan)
        with np.errstate(all="ignore"):
            result = np.nanmean(masked_values, axis=1)  # (time, node)

        # Handle nodes where all layers were masked (shallow water)
        all_masked = np.all(~mask, axis=1)  # (time, node)
        result[all_masked] = data_values[:, 0, :][all_masked]

        result_da = xr.DataArray(
            result,
            dims=("time", "node"),
            coords={
                "time": data.time,
                "lon": ("node", ds["lon"].values),
                "lat": ("node", ds["lat"].values),
            },
            attrs=data.attrs,  # Preserve original attributes (long_name, units, etc.)
        )
    else:
        # No time dimension: (siglay, node)
        masked_values = np.where(layer_in_range, data_values, np.nan)
        with np.errstate(all="ignore"):
            result = np.nanmean(masked_values, axis=0)  # (node,)

        # Handle nodes where all layers were masked
        all_masked = np.all(~layer_in_range, axis=0)  # (node,)
        result[all_masked] = data_values[0, :][all_masked]

        result_da = xr.DataArray(
            result,
            dims=("node",),
            coords={
                "lon": ("node", ds["lon"].values),
                "lat": ("node", ds["lat"].values),
            },
            attrs=data.attrs,  # Preserve original attributes (long_name, units, etc.)
        )

    if verbose:
        print("    Done.", flush=True)

    return result_da


def parse_depth_range(depth_range_str: str) -> tuple[float | str, float | str]:
    """Parse depth range string into min/max values.

    Parameters
    ----------
    depth_range_str : str
        Depth range in format "min,max", e.g., "surface,2" or "5,10"

    Returns
    -------
    tuple
        (depth_min, depth_max) where each is float or "surface"/"bottom"

    Examples
    --------
    >>> parse_depth_range("surface,2")
    ('surface', 2.0)
    >>> parse_depth_range("5,10")
    (5.0, 10.0)
    >>> parse_depth_range("0,bottom")
    (0.0, 'bottom')
    """
    parts = depth_range_str.split(",")
    if len(parts) != 2:
        raise ValueError(f"Expected 'min,max' format, got: {depth_range_str}")

    def parse_value(s: str) -> float | str:
        s = s.strip().lower()
        if s == "surface":
            return "surface"
        if s == "bottom":
            return "bottom"
        return float(s)

    return (parse_value(parts[0]), parse_value(parts[1]))
