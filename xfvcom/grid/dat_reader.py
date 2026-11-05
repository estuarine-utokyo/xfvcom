from __future__ import annotations

"""Robust reader for FVCOM `*_grd.dat` mesh files (no UTM info)."""

# jsasaki 2026-11-05: review round 18 #1: 're' no longer needed (header regex replaced)
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

# jsasaki 2026-11-05: review round 18 #1 #2 #5 #6: shared FVCOM header scanner
# jsasaki 2026-11-05: review round 19 #1 #3 #4 #5: READ-based header search and records
# jsasaki 2026-11-05: review round 20 #3 #4: whole-block READ counting and the transition model
# jsasaki 2026-11-05: review round 21 #1: Fortran's record splitting (physical_lines) and blank lines
from ..fvcom_input import (
    INT,
    REAL,
    ListRead,
    is_blank,
    list_read,
    locate_records,
    physical_lines,
    read_all,
    scan_count_headers,
)

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


# jsasaki 2026-11-05: review round 20 #1 #4: _body/_is_comment removed: '!' is not stripped before
# tokenising (list-directed input has no comments) and a trailing comment line is a counted READ
# jsasaki 2026-11-05: review round 19 #4: _fields/_is_int replaced by the shared list_read
# <--- jsasaki 2026-11-05


# ---> jsasaki 2026-11-05: review round 20 #3 #4: the failed-READ message and FVCOM's
# connectivity-to-coordinate transition
def _span(rd: ListRead) -> str:
    return (
        f"line {rd.first_line}"
        if rd.first_line == rd.last_line
        else f"lines {rd.first_line}-{rd.last_line}"
    )


def _bad_read(
    path: Path,
    lines: list[tuple[int, str]],
    rd: ListRead,
    block: str,
    row: int,
    count: int,
) -> ValueError:
    """A failed / unset READ inside a mesh block: FVCOM counts it as a row and keeps the values
    of the previous READ, so the mesh it builds is not the file's."""
    text = " / ".join(raw.strip() for _n, raw in lines[rd.start : rd.end + 1])
    return ValueError(
        f"{path}: {_span(rd)}: {block} record {row} of {count} {text!r} cannot be read: "
        f"{rd.message}; FVCOM's list-directed READ fails or leaves items unset there and "
        "still counts it as one row (keeping the previous record's values) -- a comment-only "
        "line inside the block is such a READ: move it above the first record"
    )


# ---> jsasaki 2026-11-05: review round 21 #3: FVCOM cannot read a one-element mesh (checked with
# gfortran 8.5 and ifx 2025.1 replicas of READ_COLDSTART_GRID on a three-node, one-element mesh)
_ONE_ELEMENT = (
    "{path}: the mesh has one element, whose id is 1: FVCOM's READ_COLDSTART_GRID leaves the "
    "connectivity block only at a record with J = 1 that follows an element whose id is not 1 "
    "(J == 1 .AND. LM1 /= 1), so it reads the node-1 record at line {line} as one more "
    "connectivity row and stops ('Number of rows of data in the grid file CONNECTIVITY data "
    "exceeds the stated number of Cells'); FVCOM cannot read this mesh -- "
    "read_dat(..., allow_unordered=True) reads it without FVCOM's checks"
)
# <--- jsasaki 2026-11-05


# jsasaki 2026-11-05: review round 21 #3: lm1 = the last element's id (FVCOM's LM1)
def _coordinate_start(
    path: Path, lines: list[tuple[int, str]], idx: int, nele: int, lm1: int
) -> int:
    """Index of the line where FVCOM's READ_COLDSTART_COORDS starts, or ValueError.

    After the last element FVCOM READs ``J,N1,N2,N3`` once more (the first node record); with
    J = 1 and LM1 (the last element's id) /= 1 it BACKSPACEs one record, READs ``J`` (no
    IOSTAT: a failure stops FVCOM) and BACKSPACEs J records (mod_input.F ~4657).  That lands on the node-1 record only for a node-1 record
    holding J, X and Y on one line followed directly by node 2's line.  The sequence is modelled
    with gfortran's conversion (a real fails an INTEGER item) and with Intel Fortran's (a real is
    truncated into it, so the READ takes N3 from the next line); the file is refused unless both
    land on the node-1 record (or on blank lines just before it).  With LM1 = 1 (a one-element
    mesh) FVCOM never transitions (review round 21 #3): ``idx`` is returned unchanged and
    :func:`read_dat` refuses the mesh once its other checks have run.
    """
    # jsasaki 2026-11-05: review round 21 #1: a blank line holds list-directed blanks only
    first = next((k for k in range(idx, len(lines)) if not is_blank(lines[k][1])), None)
    if first is None:
        return idx  # no node records: the node block reports it
    elem_types = (INT, INT, INT, INT)
    for compiler, int_reals in (("gfortran", False), ("Intel Fortran", True)):
        nxt = list_read(lines, idx, elem_types, int_reals=int_reals)
        if nxt.status == "eof":
            return idx  # an incomplete node block: the node block reports it
        # a READ that sets no J keeps the last element's id (LM1): no transition, FVCOM counts one
        # more cell row; so does a READ with J /= 1.  Unless the record reads as an element, the
        # node block's own checks name it.
        j = nxt.values[0] if nxt.values else None
        if j is None or (j != 1 and nxt.status != "ok"):
            return idx
        if j != 1:
            text = " / ".join(raw.strip() for _n, raw in lines[nxt.start : nxt.end + 1])
            raise ValueError(
                f"{path}: line {nxt.last_line}: record {text.strip(' /')!r} follows the "
                f"{nele} declared connectivity records, but the node block must start with "
                "node 1 (more records than the Cell Number header declares?)"
            )
        # jsasaki 2026-11-05: review round 21 #3: FVCOM transitions only when J == 1 .AND. LM1 /= 1;
        # with LM1 = 1 it never does (read_dat refuses such a mesh after reading it, _ONE_ELEMENT)
        if lm1 == 1:
            return idx
        rj = list_read(lines, nxt.end, (INT,), int_reals=int_reals)
        why = ""
        if rj.status in ("error", "eof"):
            why = (
                f"its READ(GRIDUNIT,*) J at line {lines[nxt.end][0]} fails "
                f"({rj.message or 'end of file'}) and FVCOM stops"
            )
        else:
            jj = rj.values[0] if rj.status == "ok" else 1
            assert isinstance(jj, int)
            land = rj.end + 1 - jj if jj > 0 else rj.end + 1
            land = max(land, 0)
            # jsasaki 2026-11-05: review round 21 #1: blank = list-directed blanks only
            if land > first or any(not is_blank(raw) for _n, raw in lines[land:first]):
                why = (
                    f"it reads J = {jj} at line {lines[nxt.end][0]} and BACKSPACEs to line "
                    f"{lines[land][0] if land < len(lines) else 'EOF'} instead of the node-1 "
                    f"record at line {lines[first][0]}"
                )
        if why:
            raise ValueError(
                f"{path}: the first node record (line {lines[first][0]}) cannot be replayed by "
                f"FVCOM's connectivity-to-coordinate transition under {compiler}: {why} "
                "(mod_input.F READ_COLDSTART_GRID) -- write node 1 as 'J X Y' on one line, "
                "directly followed by node 2's line"
            )
    return idx


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

    The strict mode also refuses what FVCOM's READ_COLDSTART_GRID stops on after the
    connectivity block: a one-element mesh (its transition test ``J == 1 .AND. LM1 /= 1`` never
    fires) and a largest referenced node id that differs from ``Node Number``.
    ``allow_unordered=True`` skips these checks too (review round 21 #3 #4).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    # -------------------------------------------------- read non-blank lines
    # jsasaki 2026-11-05: review round 15 #3: keep physical line numbers for the error messages
    # jsasaki 2026-11-05: review round 19 #4 #5: all physical lines, blanks included (READs span them)
    # jsasaki 2026-11-05: review round 21 #1: split at '\n' only (one CRLF '\r' removed), not with
    # Python's universal newlines
    lines = physical_lines(path)

    # ---------------------------------------- locate counts (node / nele)
    # ---> jsasaki 2026-11-05: review round 21 #6: the current rules (rounds 15-21).  FVCOM's
    # SCAN_FILE looks each header up on its own (rewinding), so 'Cell Number' may precede
    # 'Node Number'; both must come before the connectivity block and at or before physical line
    # 201 (SCAN_FILE tests NLINE > 200 before reading, so line 201 is the last it reads).  The
    # block starts where FVCOM's header search first READs J,N1,N2,N3 successfully (possibly over
    # several lines), its NHEADER rewind checked.  Records are tokenised from their ORIGINAL text:
    # '!' is kept (list-directed input has no comments, so '1!x' is one token that fails),
    # commas separate values like spaces and tabs, and no other whitespace does.
    # <--- jsasaki 2026-11-05
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
        # (it may continue onto further lines)
        rd = list_read(lines, idx, elem_types)
        if rd.status == "eof":
            break
        idx = rd.end + 1
        # jsasaki 2026-11-05: review round 12 #3: FVCOM reads J,N1,N2,N3; extra columns are unused
        # jsasaki 2026-11-05: review round 20 #4: READ_COLDSTART_GRID counts a failed or unset READ as
        # a connectivity row (keeping the previous values), so it is refused, not passed over
        if rd.status != "ok":
            raise _bad_read(path, lines, rd, "connectivity", read_elem + 1, nele)
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
    # (READ_COLDSTART_GRID stops: rows exceed the stated number of cells).
    # jsasaki 2026-11-05: review round 20 #3: FVCOM then BACKSPACEs, READs J and BACKSPACEs J records
    # (mod_input.F ~4657); the coordinate block starts where that sequence leaves the file
    node_line = 0
    if not allow_unordered:
        # jsasaki 2026-11-05: review round 21 #3: LM1 = the last element's id (= nele in order)
        idx = _coordinate_start(path, lines, idx, nele, nele)
        node_line = next((n for n, raw in lines[idx:] if not is_blank(raw)), 0)
        # ---> jsasaki 2026-11-05: review round 21 #4: FVCOM's NODECOUNT check (READ_COLDSTART_GRID):
        # the largest node id the connectivity references must equal the declared Node Number
        nodecount = int(nv.max()) + 1
        if nodecount != node:
            raise ValueError(
                f"{path}: the largest node id referenced by the connectivity is {nodecount}, but "
                f"'Node Number = {node}'; FVCOM's READ_COLDSTART_GRID stops on it ('The number "
                "of nodes in the grid file CONNECTIVITY does not equal the stated number') -- "
                "every declared node must be used by an element "
                "(read_dat(..., allow_unordered=True) skips this check)"
            )
        # <--- jsasaki 2026-11-05
    # <--- jsasaki 2026-11-05

    # -------------------------------------------------------- node block (after elem)
    x: NDArray[np.float64] = np.empty(node, dtype=np.float64)
    y: NDArray[np.float64] = np.empty(node, dtype=np.float64)
    # jsasaki 2026-11-05: review round 4 #3: coordinates are placed by the record's own node id
    # (validated 1..node and unique), not by row position
    seen: NDArray[np.bool_] = np.zeros(node, dtype=bool)
    # ---> jsasaki 2026-11-05: review round 20 #4: READ_COLDSTART_COORDS counts EVERY READ that does
    # not hit end of file as one node row (a failed or unset one included, keeping the previous
    # values) and stops when the rows exceed the stated number; blank lines only continue a READ
    reads, eof = read_all(lines, idx, (INT, REAL, REAL))
    for read_node, rd in enumerate(reads):
        if read_node == node:
            text = " / ".join(raw.strip() for _n, raw in lines[rd.start : rd.end + 1])
            raise ValueError(
                f"{path}: line {rd.last_line}: record {text.strip(' /')!r} follows the "
                f"{node} declared node records; FVCOM's coordinate loop counts every READ "
                "that does not reach end of file as one more node row, a failed one (e.g. a "
                "comment-only line) included (more records than the Node Number header "
                "declares?)"
            )
        # jsasaki 2026-11-05: review round 12 #3: FVCOM reads J,X1,Y1; extra columns are unused
        if rd.status != "ok":
            raise _bad_read(path, lines, rd, "node", read_node + 1, node)
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
    if len(reads) != node:
        raise ValueError(f"Node block ended prematurely ({len(reads)}/{node})")
    # a partial record before end of file (FVCOM's READ hits EOF and stops quietly) is a stray
    if eof.values:
        raise ValueError(
            f"{path}: lines {eof.first_line}-{eof.last_line}: an incomplete record follows the "
            f"{node} declared node records"
        )
    # <--- jsasaki 2026-11-05
    # jsasaki 2026-11-05: review round 21 #3: refused after the other checks so that their messages
    # (which name the record at fault) come first
    if not allow_unordered and nele == 1:
        raise ValueError(_ONE_ELEMENT.format(path=path, line=node_line))

    return _GridData(x=x, y=y, nv=nv, zone=None, northern=True)
