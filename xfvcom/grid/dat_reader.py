from __future__ import annotations

"""Robust reader for FVCOM `*_grd.dat` mesh files (no UTM info)."""

import re
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

__all__ = ["read_dat"]


class _GridData(TypedDict):
    x: np.ndarray
    y: np.ndarray
    nv: np.ndarray  # (3, nele)
    zone: int | None
    northern: bool


def _is_int(tok: str) -> bool:
    return tok.lstrip("+-").isdigit()


# jsasaki 2026-11-05: review round 9 #2: FVCOM places records by row position; strict by default
def read_dat(path: str | Path, *, allow_unordered: bool = False) -> _GridData:
    """Read an FVCOM ``*_grd.dat`` (connectivity block, then node block).

    FVCOM's ``READ_COLDSTART_CONNECTIVITY`` / ``READ_COLDSTART_COORDS``
    (``mod_input.F``) assign each element and node record by its ROW POSITION
    and ignore the record's own id.  By default (``allow_unordered=False``)
    the ids must therefore run 1..n in file order, so that the mesh read here
    is the mesh FVCOM uses; a file that breaks this raises :class:`ValueError`.
    ``allow_unordered=True`` instead places each record by its own id (unique,
    1..n), which reconstructs a reordered file but NOT what FVCOM reads.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    # -------------------------------------------------- read non-blank lines
    with path.open() as fp:
        lines = [ln.rstrip("\n") for ln in fp if ln.strip()]

    # ---------------------------------------- locate counts (node / nele)
    node = nele = None  # type: ignore[assignment]
    idx = 0

    while idx < len(lines):
        ln = lines[idx]
        low = ln.lower()

        if low.startswith("node number"):
            # e.g. "Node Number = 3210"
            digits = re.findall(r"\d+", ln)
            if digits:
                node = int(digits[0])
            idx += 1
            continue
        if low.startswith("cell number") or low.startswith("elem number"):
            digits = re.findall(r"\d+", ln)
            if digits:
                nele = int(digits[0])
            idx += 1
            break
        idx += 1

    if node is None or nele is None:
        raise ValueError("Node/Cell counts not found in DAT file")

    # ----------------------------------------------------- element block (comes first)
    nv: NDArray[np.int_] = np.empty((3, nele), dtype=np.int_)
    read_elem = 0
    # jsasaki 2026-11-05: review round 5 #6: element records are placed by their own id (1..nele, unique)
    seen_elem: NDArray[np.bool_] = np.zeros(nele, dtype=bool)
    while read_elem < nele and idx < len(lines):
        parts = lines[idx].split()
        idx += 1
        if len(parts) != 5 or not _is_int(parts[0]):
            continue
        ids = [int(parts[1]), int(parts[2]), int(parts[3])]
        # jsasaki 2026-11-05: review round 3 #19: node ids are 1-based; 0 would become index -1
        if min(ids) < 1 or max(ids) > node:
            raise ValueError(
                f"element {parts[0]}: node id(s) {ids} outside 1..{node} (1-based)"
            )
        # ---> jsasaki 2026-11-05: review round 5 #6: the record id, not the row position, selects the column
        eid = int(parts[0])
        if eid < 1 or eid > nele:
            raise ValueError(f"element record id {eid} outside 1..{nele} (1-based)")
        if seen_elem[eid - 1]:
            raise ValueError(f"element record id {eid} appears more than once")
        # jsasaki 2026-11-05: review round 9 #2: FVCOM reads element row k as element k
        if not allow_unordered and eid != read_elem + 1:
            raise ValueError(
                f"element record {read_elem + 1} carries id {eid}; FVCOM assigns "
                "connectivity by row position, so the ids must run 1..n in order "
                "(read_dat(..., allow_unordered=True) places records by id instead)"
            )
        seen_elem[eid - 1] = True
        nv[:, eid - 1] = [ids[0] - 1, ids[1] - 1, ids[2] - 1]
        # <--- jsasaki 2026-11-05
        read_elem += 1
    if read_elem != nele:
        raise ValueError(f"Element block ended prematurely ({read_elem}/{nele})")

    # -------------------------------------------------------- node block (after elem)
    x: NDArray[np.float64] = np.empty(node, dtype=np.float64)
    y: NDArray[np.float64] = np.empty(node, dtype=np.float64)
    read_node = 0
    # jsasaki 2026-11-05: review round 4 #3: coordinates are placed by the record's own node id
    # (validated 1..node and unique), not by row position
    seen: NDArray[np.bool_] = np.zeros(node, dtype=bool)
    while read_node < node and idx < len(lines):
        parts = lines[idx].split()
        idx += 1
        if len(parts) != 4 or not _is_int(parts[0]):
            continue
        try:
            xv = float(parts[1])
            yv = float(parts[2])
        except ValueError:
            continue
        nid = int(parts[0])
        if nid < 1 or nid > node:
            raise ValueError(f"node record id {nid} outside 1..{node} (1-based)")
        if seen[nid - 1]:
            raise ValueError(f"node record id {nid} appears more than once")
        # jsasaki 2026-11-05: review round 9 #2: FVCOM reads node row k as node k (ignores J)
        if not allow_unordered and nid != read_node + 1:
            raise ValueError(
                f"node record {read_node + 1} carries id {nid}; FVCOM assigns "
                "coordinates by row position (mod_input.F READ_COLDSTART_COORDS), "
                "so the ids must run 1..n in order "
                "(read_dat(..., allow_unordered=True) places records by id instead)"
            )
        seen[nid - 1] = True
        x[nid - 1] = xv
        y[nid - 1] = yv
        read_node += 1
    if read_node != node:
        raise ValueError(f"Node block ended prematurely ({read_node}/{node})")

    return _GridData(x=x, y=y, nv=nv, zone=None, northern=True)
