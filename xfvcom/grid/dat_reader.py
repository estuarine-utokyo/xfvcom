from __future__ import annotations

"""Robust reader for FVCOM `*_grd.dat` mesh files (no UTM info)."""

# jsasaki 2026-11-05: review round 18 #1: 're' no longer needed (header regex replaced)
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

# jsasaki 2026-11-05: review round 18 #1 #2 #5 #6: shared FVCOM header scanner
# jsasaki 2026-11-05: review round 19 #1 #3 #4 #5: READ-based header search and records
from ..fvcom_input import INT, REAL, list_read, locate_records, scan_count_headers

# jsasaki 2026-11-05: review round 14 #2: FVCOM reads coordinates list-directed (D/E/Q exponents)
# jsasaki 2026-11-05: review round 17 #4: list-directed fields (commas separate like blanks)
# jsasaki 2026-11-05: review round 19 #4: conversion now inside the shared list_read


__all__ = ["read_dat"]


class _GridData(TypedDict):
    x: np.ndarray
    y: np.ndarray
    nv: np.ndarray  # (3, nele)
    zone: int | None
    northern: bool


# ---> jsasaki 2026-11-05: review round 15 #2 #3 #5: strict count headers, records after the blocks
# jsasaki 2026-11-05: review round 18 #1 #2 #5 #6: the count headers are looked up by the shared
# FVCOM scanner (xfvcom/fvcom_input.py: 400-byte buffer in UTF-8 bytes, harmless long comments,
# space-only padding, GET_VALUE conversion of every traversed line); the local regex is gone
#: The count-header keys FVCOM looks up (SCAN_FILE(..., "Node Number", ...) in mod_input.F), compared
#: case-sensitively and in full with TRIM(VARNAME) == TRIM(VNAME) (mod_utils.F).
_COUNT_KEYS = {"Node Number": "node", "Cell Number": "nele"}
# <--- jsasaki 2026-11-05


# jsasaki 2026-11-05: review round 16 #3: a record is the text before its first '!' (attached comments too)
def _body(raw: str) -> str:
    """The record part of a line: everything before the first ``!`` comment marker."""
    return raw.split("!", 1)[0]


def _is_comment(raw: str) -> bool:
    """A comment-only line (nothing but blanks before ``!``); blank lines are dropped before this is asked."""
    # jsasaki 2026-11-05: review round 16 #3: same definition as the record body
    return not _body(raw).strip()


# jsasaki 2026-11-05: review round 19 #4: _fields/_is_int replaced by the shared list_read
# <--- jsasaki 2026-11-05


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
    # jsasaki 2026-11-05: review round 15 #3: keep physical line numbers for the error messages
    # jsasaki 2026-11-05: review round 19 #4 #5: all physical lines, blanks included (READs span them)
    with path.open() as fp:
        lines = [(n, ln.rstrip("\n")) for n, ln in enumerate(fp, 1)]

    # ---------------------------------------- locate counts (node / nele)
    # ---> jsasaki 2026-11-05: review round 15 #5: FVCOM's SCAN_FILE looks each header up on its own
    # (rewinding), so 'Cell Number' may precede 'Node Number'.  The connectivity block starts at the
    # first record that reads as four integers (READ_COLDSTART_GRID); both headers must come before it
    # and within the 200 lines SCAN_FILE scans.
    # jsasaki 2026-11-05: review round 16 #3: tokenise the record without its '!' comment
    # jsasaki 2026-11-05: review round 17 #4: commas separate fields like blanks
    # jsasaki 2026-11-05: review round 19 #1 #4 #5: the block starts where FVCOM's header search first
    # READs J,N1,N2,N3 successfully (possibly over several lines), its NHEADER rewind checked
    elem_types = (INT, INT, INT, INT)
    start = locate_records(path, lines, elem_types, block="connectivity block")
    # jsasaki 2026-11-05: review round 18 #1 #2 #5 #6: one shared FVCOM-aware header scanner
    counts = scan_count_headers(
        path, lines, _COUNT_KEYS, start, block="connectivity block"
    )
    if "node" not in counts or "nele" not in counts:
        # jsasaki 2026-11-05: review round 17 #2: name the exact keys FVCOM looks up
        raise ValueError(
            "Node/Cell counts not found in DAT file (FVCOM looks up the exact, "
            "case-sensitive header keys 'Node Number' and 'Cell Number')"
        )
    node, nele = counts["node"], counts["nele"]
    idx = start
    # <--- jsasaki 2026-11-05

    # ----------------------------------------------------- element block (comes first)
    nv: NDArray[np.int_] = np.empty((3, nele), dtype=np.int_)
    read_elem = 0
    # jsasaki 2026-11-05: review round 5 #6: element records are placed by their own id (1..nele, unique)
    seen_elem: NDArray[np.bool_] = np.zeros(nele, dtype=bool)
    while read_elem < nele and idx < len(lines):
        # jsasaki 2026-11-05: review round 19 #4: one record = one list-directed READ of J,N1,N2,N3
        # (it may continue onto further lines); a READ that fails is passed over as before
        rd = list_read(lines, idx, elem_types)
        if rd.status == "eof":
            break
        idx = rd.end + 1
        # jsasaki 2026-11-05: review round 12 #3: FVCOM reads J,N1,N2,N3; extra columns are unused
        if rd.status != "ok":
            continue
        parts = [str(v) for v in rd.values]
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
    # ---> jsasaki 2026-11-05: review round 15 #3: FVCOM ends the connectivity block at the first
    # record whose id is 1 (the first node record); any other record there is one more element
    # (READ_COLDSTART_GRID stops: rows exceed the stated number of cells).  A further record that
    # reads as an element (four integers) and is not node 1 is reported here; any other misplaced
    # record is reported by the node block's own order check below.
    if not allow_unordered:
        # jsasaki 2026-11-05: review round 19 #4: the next READ of J,N1,N2,N3, as FVCOM's loop does it
        nxt = list_read(lines, idx, elem_types)
        if nxt.status == "ok" and nxt.values[0] != 1:
            text = " / ".join(raw.strip() for _n, raw in lines[nxt.start : nxt.end + 1])
            raise ValueError(
                f"{path}: line {nxt.last_line}: record {text.strip(' /')!r} follows the {nele} "
                "declared connectivity records, but the node block must start with "
                "node 1 (more records than the Cell Number header declares?)"
            )
    # <--- jsasaki 2026-11-05

    # -------------------------------------------------------- node block (after elem)
    x: NDArray[np.float64] = np.empty(node, dtype=np.float64)
    y: NDArray[np.float64] = np.empty(node, dtype=np.float64)
    read_node = 0
    # jsasaki 2026-11-05: review round 4 #3: coordinates are placed by the record's own node id
    # (validated 1..node and unique), not by row position
    seen: NDArray[np.bool_] = np.zeros(node, dtype=bool)
    while read_node < node and idx < len(lines):
        # jsasaki 2026-11-05: review round 19 #4: one record = one list-directed READ of J,X1,Y1 (it
        # may continue onto further lines; Fortran real literals as in review round 14 #2); a READ
        # that fails is passed over as before
        rd = list_read(lines, idx, (INT, REAL, REAL))
        if rd.status == "eof":
            break
        idx = rd.end + 1
        # jsasaki 2026-11-05: review round 12 #3: FVCOM reads J,X1,Y1; extra columns are unused
        if rd.status != "ok":
            continue
        xv, yv = float(rd.values[1]), float(rd.values[2])  # type: ignore[arg-type]
        nid = int(rd.values[0])  # type: ignore[arg-type]
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
    # ---> jsasaki 2026-11-05: review round 15 #3: FVCOM reads every further row as one more node
    # and stops (READ_COLDSTART_COORDS); only blank and comment-only lines may follow
    for lineno, raw in lines[idx:]:
        if not _is_comment(raw):
            raise ValueError(
                f"{path}: line {lineno}: record {raw.strip()!r} follows the {node} "
                "declared node records (more records than the Node Number header declares?)"
            )
    # <--- jsasaki 2026-11-05

    return _GridData(x=x, y=y, nv=nv, zone=None, northern=True)
