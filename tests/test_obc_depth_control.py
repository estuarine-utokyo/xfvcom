r"""Tests for the a-priori OBC_DEPTH_CONTROL_ON substitution (mod_obcs.F port).

Synthetic strip mesh (Cartesian):

    3 --- 4 --- 5      y = 1  (top row: the open boundary, obc.dat order 4,5,6)
    | \   | \   |
    0 --- 1 --- 2      y = 0  (bottom row: interior/solid boundary)

Triangles (0-based, CCW): (0,1,3), (1,4,3), (1,2,4), (2,5,4).

For the MIDDLE top node 4 (1-based 5): two adjacent OBC nodes (3, 5), the
inward normal points to -y, and the non-OBC edge-neighbours are {1, 2}
(0-based). Node 1 sits straight below (unit dot 1.0) -> NEXT_OBC = 2
(1-based). The two top ENDPOINTS keep their own depth (observed FVCOM
runtime behaviour at open/solid corners; see module docstring).
"""

import numpy as np

from xfvcom.io.obc_depth_control import next_obc_nodes, substituted_obc_depths

X = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
Y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
NV = np.array([  # (3, nele), 0-based
    [0, 1, 1, 2],
    [1, 4, 2, 5],
    [3, 3, 4, 4],
])
OBC_IDS = np.array([4, 5, 6])          # 1-based: the whole top row
H = np.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0])


def test_middle_node_picks_straight_inward_neighbour():
    nxt = next_obc_nodes(X, Y, NV, OBC_IDS)
    assert nxt[1] == 2                  # node 5 (middle top) -> node 2 below


def test_endpoints_keep_own_depth():
    h_sub, nxt = substituted_obc_depths(X, Y, NV, OBC_IDS, H)
    assert nxt[0] == 4 and nxt[2] == 6  # endpoints map to themselves
    assert h_sub[0] == H[3] and h_sub[2] == H[5]


def test_substituted_depth_is_neighbours():
    h_sub, _ = substituted_obc_depths(X, Y, NV, OBC_IDS, H)
    assert h_sub[1] == H[1]             # middle top gets node 2's depth (20.0)
