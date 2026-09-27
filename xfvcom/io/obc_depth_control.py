"""A-priori FVCOM OBC depth-control substitution.

With FVCOM's default ``OBC_DEPTH_CONTROL_ON = T``, ``SET_WATER_DEPTH``
(``mod_startup.F``) replaces every open-boundary node's depth (and sigma
column) with its inward neighbour's ``NEXT_OBC``.  An OBC T/S file must
therefore be sampled at the *substituted* depths to be imposed at the depths
the model actually uses (FVCOM does no vertical re-interpolation of the OBC
file; see TB-FVCOM ``hydro/docs/obc_depth_consistency.md``).

This module ports the ``NEXT_OBC`` determination from ``mod_obcs.F``
``SETUP_OBC`` so the substitution is computed **a priori** from the grid and
OBC-node list alone — no FVCOM run is needed:

1. For each OBC node, its adjacent OBC nodes along the boundary line are the
   edge-neighbours that are themselves OBC nodes (``ISONB == 2``).
2. The inward normal is accumulated over the (up to two) boundary edges to
   those adjacent OBC nodes: for each edge, the cell containing the edge
   supplies the orientation (``CROSS = sign((c-p) x (q-p))``) and the edge
   normal ``(CROSS*dyn, -CROSS*dxn)/|edge|`` points *into* the domain.
3. ``NEXT_OBC`` is the edge-neighbour that is NOT an OBC node maximising the
   dot product of its unit direction with the inward normal.

Cartesian (UTM) meshes only — mirrors the ``#else`` (non-SPHERICAL) branch.

**Boundary-string ENDPOINTS (open/solid corner nodes) are NOT substituted.**
Empirical basis (TB-FVCOM b9 mesh, 60-rank production run, 2026-08-03): the
runtime ``h`` shows the endpoint node 3150 KEPT its own dep-file depth
(117.277 m) even though the ported ``NEXT_OBC`` rule would substitute a
neighbour's (107.30/160.95/78.18 — none matches), while the interior OBC
nodes follow the ported rule exactly (3142 -> h(3151), 3137 -> h(3146); the
other 10 have equal-depth neighbours so substitution is a no-op). The code
path that skips corners at runtime has not been pinned down yet (candidate:
a parallel NBSN/halo subtlety at open-solid junctions; the FVCOM repo was
read-only during this arc) — see TB-FVCOM
``hydro/docs/obc_depth_consistency.md`` §6.

Because of that open corner mechanism, treat this module as the PRE-RUN
constructor and ALWAYS contract-check against a run: compare the depths
assumed here with the output NetCDF ``h`` at the OBC nodes (the
harvest-based ``TB-FVCOM/input/scripts/build_dep_obcsub.py`` route) before
trusting a production OBC file on a new mesh or a new rank count.

Validated against the TB-FVCOM b9 production mesh: with the endpoint rule,
the substituted depths reproduce the runtime ``h`` at all 13 OBC nodes.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray


def _topology(nv: NDArray[np.int_]):
    """Node->neighbour-set and node->incident-cell-list from (3, nele) 0-based nv."""
    neighbours: dict[int, set[int]] = defaultdict(set)
    cells_of: dict[int, list[int]] = defaultdict(list)
    for c in range(nv.shape[1]):
        n1, n2, n3 = int(nv[0, c]), int(nv[1, c]), int(nv[2, c])
        neighbours[n1] |= {n2, n3}
        neighbours[n2] |= {n1, n3}
        neighbours[n3] |= {n1, n2}
        cells_of[n1].append(c)
        cells_of[n2].append(c)
        cells_of[n3].append(c)
    return neighbours, cells_of


def next_obc_nodes(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    nv: NDArray[np.int_],
    obc_ids: NDArray[np.int_],
) -> NDArray[np.int_]:
    """Return the 1-based ``NEXT_OBC`` node for each 1-based OBC node.

    Parameters
    ----------
    x, y
        Node coordinates (UTM metres), full mesh.
    nv
        Connectivity ``(3, nele)``, zero-based (``FvcomGrid.nv``).
    obc_ids
        1-based OBC node IDs in ``*_obc.dat`` order.
    """
    nv = np.asarray(nv)
    if nv.shape[0] != 3:
        raise ValueError(f"nv must be (3, nele); got {nv.shape}")
    neighbours, cells_of = _topology(nv)
    xc = x[nv].mean(axis=0)
    yc = y[nv].mean(axis=0)

    obc0 = [int(i) - 1 for i in obc_ids]
    obcset = set(obc0)
    out: NDArray[np.int64] = np.empty(len(obc0), dtype=np.int64)

    for k, p in enumerate(obc0):
        adj_obc = [j for j in neighbours[p] if j in obcset]
        if not adj_obc:
            raise ValueError(
                f"OBC node {p + 1}: no adjacent OBC node found "
                "(mod_obcs.F would PSTOP here)"
            )
        if len(adj_obc) > 2:
            raise ValueError(
                f"OBC node {p + 1}: {len(adj_obc)} adjacent OBC "
                "nodes; boundary line is not simple"
            )
        if len(adj_obc) == 1:
            # Boundary-string ENDPOINT (open/solid corner): observed runtime
            # behaviour is NO substitution (see module docstring) — return the
            # node itself so h_substituted == own depth.
            out[k] = p + 1
            continue
        acc_x = acc_y = 0.0
        for q in adj_obc:
            edge_cells = [
                c
                for c in cells_of[p]
                if q in (int(nv[0, c]), int(nv[1, c]), int(nv[2, c]))
            ]
            if len(edge_cells) != 1:
                raise ValueError(
                    f"edge ({p + 1},{q + 1}) is shared by {len(edge_cells)} "
                    "cells; expected a boundary edge (exactly 1)"
                )
            c = edge_cells[0]
            dxn = x[q] - x[p]
            dyn = y[q] - y[p]
            dxc = xc[c] - x[p]
            dyc = yc[c] - y[p]
            cross = 1.0 if (dxc * dyn - dyc * dxn) >= 0.0 else -1.0
            ln = float(np.hypot(dxn, dyn))
            acc_x += cross * dyn / ln
            acc_y += -cross * dxn / ln
        nrm = float(np.hypot(acc_x, acc_y))
        nx, ny = acc_x / nrm, acc_y / nrm

        best_dot, best = -2.0, -1
        for j in neighbours[p]:
            if j in obcset:
                continue
            dx = x[j] - x[p]
            dy = y[j] - y[p]
            ln = float(np.hypot(dx, dy))
            dot = (dx * nx + dy * ny) / ln
            if dot > best_dot:
                best_dot, best = dot, j
        if best < 0:
            raise ValueError(f"OBC node {p + 1}: no non-OBC neighbour found")
        out[k] = best + 1
    return out


def substituted_obc_depths(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    nv: NDArray[np.int_],
    obc_ids: NDArray[np.int_],
    h_all: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.int_]]:
    """Depths FVCOM actually uses at the OBC nodes under depth control.

    Returns ``(h_substituted, next_ids)`` where ``h_substituted[k] =
    h_all[next_ids[k] - 1]`` — feed these to the OBC generator so the file is
    sampled at the depths ``OBC_DEPTH_CONTROL_ON = T`` imposes it at.
    """
    nxt = next_obc_nodes(x, y, nv, obc_ids)
    return np.asarray(h_all)[nxt - 1], nxt
