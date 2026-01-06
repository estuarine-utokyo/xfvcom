"""
xfvcom.grid.interpolation - Interpolation utilities for FVCOM unstructured grids
---------------------------------------------------------------------------------

Functions for interpolating between element centers and nodes.

Reference:
    PyFVCOM elems2nodes implementation:
    https://github.com/pwcazenave/PyFVCOM/blob/master/PyFVCOM/grid/_grid.py
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def elems2nodes(
    elem_data: NDArray,
    triangles: NDArray,
    n_nodes: int | None = None,
) -> NDArray:
    """
    Interpolate element-centered data to nodes using area-weighted averaging.

    Each node value is computed as the mean of all elements that share that node.
    This is the standard approach for unstructured triangular grids.

    At boundary/coastline nodes, the algorithm naturally handles fewer contributing
    elements (1-3 instead of ~6 for interior nodes) because it tracks the actual
    count per node.

    Parameters
    ----------
    elem_data : NDArray
        Element-centered data. Can be:
        - 1D: (n_elem,) - single field
        - 2D: (n_time, n_elem) or (n_siglay, n_elem)
        - 3D: (n_time, n_siglay, n_elem)
    triangles : NDArray
        Connectivity array of shape (3, n_elem) or (n_elem, 3).
        Each column/row contains the 3 node indices for an element.
        Node indices should be 0-based.
    n_nodes : int, optional
        Number of nodes in the grid. If None, inferred from max(triangles) + 1.

    Returns
    -------
    NDArray
        Node-centered data with same leading dimensions as input,
        but last dimension is n_nodes instead of n_elem.

    Notes
    -----
    This implementation uses vectorized numpy operations for performance,
    avoiding explicit Python loops over elements.

    The interpolation is not reversible: converting from elements to nodes
    and back will not recover the original element values due to averaging.

    Examples
    --------
    >>> import numpy as np
    >>> from xfvcom.grid.interpolation import elems2nodes
    >>> # Simple 2-element mesh: 4 nodes, 2 triangles
    >>> triangles = np.array([[0, 1, 2], [1, 2, 3]]).T  # shape (3, 2)
    >>> elem_values = np.array([1.0, 2.0])  # value at each element
    >>> node_values = elems2nodes(elem_values, triangles, n_nodes=4)
    >>> # Node 0: only in elem 0 -> 1.0
    >>> # Node 1: in elem 0 and 1 -> (1+2)/2 = 1.5
    >>> # Node 2: in elem 0 and 1 -> (1+2)/2 = 1.5
    >>> # Node 3: only in elem 1 -> 2.0

    Reference
    ---------
    Based on PyFVCOM implementation:
    https://github.com/pwcazenave/PyFVCOM/blob/master/PyFVCOM/grid/_grid.py
    """
    elem_data = np.asarray(elem_data)
    triangles = np.asarray(triangles)

    # Ensure triangles is (3, n_elem)
    if triangles.shape[0] != 3:
        if triangles.shape[1] == 3:
            triangles = triangles.T
        else:
            raise ValueError(
                f"triangles must have shape (3, n_elem) or (n_elem, 3), "
                f"got {triangles.shape}"
            )

    n_elem = triangles.shape[1]
    if n_nodes is None:
        n_nodes = int(np.max(triangles)) + 1

    # Validate element dimension
    if elem_data.shape[-1] != n_elem:
        raise ValueError(
            f"Last dimension of elem_data ({elem_data.shape[-1]}) "
            f"must match number of elements ({n_elem})"
        )

    # Count how many elements share each node (vectorized)
    count = np.zeros(n_nodes, dtype=np.int32)
    np.add.at(count, triangles[0], 1)
    np.add.at(count, triangles[1], 1)
    np.add.at(count, triangles[2], 1)

    # Avoid division by zero for unused nodes
    count = np.maximum(count, 1)

    # Handle different input dimensions
    if elem_data.ndim == 1:
        # 1D: (n_elem,) -> (n_nodes,)
        node_data = np.zeros(n_nodes, dtype=elem_data.dtype)
        np.add.at(node_data, triangles[0], elem_data)
        np.add.at(node_data, triangles[1], elem_data)
        np.add.at(node_data, triangles[2], elem_data)
        node_data /= count

    elif elem_data.ndim == 2:
        # 2D: (n_time, n_elem) -> (n_time, n_nodes)
        n_time = elem_data.shape[0]
        node_data = np.zeros((n_time, n_nodes), dtype=elem_data.dtype)
        for t in range(n_time):
            np.add.at(node_data[t], triangles[0], elem_data[t])
            np.add.at(node_data[t], triangles[1], elem_data[t])
            np.add.at(node_data[t], triangles[2], elem_data[t])
        node_data /= count[np.newaxis, :]

    elif elem_data.ndim == 3:
        # 3D: (n_time, n_siglay, n_elem) -> (n_time, n_siglay, n_nodes)
        n_time, n_siglay = elem_data.shape[:2]
        node_data = np.zeros((n_time, n_siglay, n_nodes), dtype=elem_data.dtype)
        for t in range(n_time):
            for k in range(n_siglay):
                np.add.at(node_data[t, k], triangles[0], elem_data[t, k])
                np.add.at(node_data[t, k], triangles[1], elem_data[t, k])
                np.add.at(node_data[t, k], triangles[2], elem_data[t, k])
        node_data /= count[np.newaxis, np.newaxis, :]

    else:
        raise ValueError(f"elem_data must be 1D, 2D, or 3D, got {elem_data.ndim}D")

    return node_data


def nodes2elems(
    node_data: NDArray,
    triangles: NDArray,
) -> NDArray:
    """
    Interpolate node-centered data to element centers.

    Each element value is computed as the mean of its three corner nodes.

    Parameters
    ----------
    node_data : NDArray
        Node-centered data. Can be:
        - 1D: (n_nodes,)
        - 2D: (n_time, n_nodes) or (n_siglay, n_nodes)
        - 3D: (n_time, n_siglay, n_nodes)
    triangles : NDArray
        Connectivity array of shape (3, n_elem) or (n_elem, 3).

    Returns
    -------
    NDArray
        Element-centered data.

    Examples
    --------
    >>> node_values = np.array([1.0, 2.0, 3.0, 4.0])
    >>> triangles = np.array([[0, 1, 2], [1, 2, 3]]).T
    >>> elem_values = nodes2elems(node_values, triangles)
    >>> # Element 0: (1+2+3)/3 = 2.0
    >>> # Element 1: (2+3+4)/3 = 3.0
    """
    node_data = np.asarray(node_data)
    triangles = np.asarray(triangles)

    # Ensure triangles is (3, n_elem)
    if triangles.shape[0] != 3:
        if triangles.shape[1] == 3:
            triangles = triangles.T
        else:
            raise ValueError(
                f"triangles must have shape (3, n_elem) or (n_elem, 3), "
                f"got {triangles.shape}"
            )

    # Vectorized: average of three corner nodes
    if node_data.ndim == 1:
        elem_data = (
            node_data[triangles[0]] + node_data[triangles[1]] + node_data[triangles[2]]
        ) / 3.0
    elif node_data.ndim == 2:
        elem_data = (
            node_data[..., triangles[0]]
            + node_data[..., triangles[1]]
            + node_data[..., triangles[2]]
        ) / 3.0
    elif node_data.ndim == 3:
        elem_data = (
            node_data[..., triangles[0]]
            + node_data[..., triangles[1]]
            + node_data[..., triangles[2]]
        ) / 3.0
    else:
        raise ValueError(f"node_data must be 1D, 2D, or 3D, got {node_data.ndim}D")

    return elem_data
