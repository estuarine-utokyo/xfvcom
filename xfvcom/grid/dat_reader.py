from __future__ import annotations

"""Robust reader for FVCOM `*_grd.dat` mesh files (no UTM info)."""

import re
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

# jsasaki 2026-11-05: review round 14 #2: FVCOM reads coordinates list-directed (D/E/Q exponents)
# jsasaki 2026-11-05: review round 17 #4: list-directed fields (commas separate like blanks)
from ..fortran_numbers import fortran_real, list_fields

__all__ = ["read_dat"]


class _GridData(TypedDict):
    x: np.ndarray
    y: np.ndarray
    nv: np.ndarray  # (3, nele)
    zone: int | None
    northern: bool


def _is_int(tok: str) -> bool:
    return tok.lstrip("+-").isdigit()


# ---> jsasaki 2026-11-05: review round 15 #2 #3 #5: strict count headers, records after the blocks
# jsasaki 2026-11-05: review round 16 #2: SCAN_FILE tests NLINE > 200 BEFORE reading, so line 201 is read
#: The last physical line FVCOM's SCAN_FILE reads (mod_utils.F checks NLINE > 200, then reads line NLINE+1).
_SCAN_FILE_LINES = 201
#: The value of a count header: ONE unsigned integer, optional '+', spaces and a trailing '!'
#: comment (FVCOM's GET_VALUE strips the comment and reads an integer; '3.5' is typed 'float' and
#: refused by SCAN_FILE(ISCAL=...)).
# jsasaki 2026-11-05: review round 17 #2: the '!' comment is removed before the value is read
_COUNT_VALUE = re.compile(r"\s*\+?([0-9]+)\s*")
# ---> jsasaki 2026-11-05: review round 17 #2 #3: SCAN_FILE's exact keys and its line buffer
#: The count-header keys FVCOM looks up (SCAN_FILE(..., "Node Number", ...) in mod_input.F), compared
#: case-sensitively and in full with TRIM(VARNAME) == TRIM(VNAME) (mod_utils.F).
_COUNT_KEYS = {"Node Number": "node", "Cell Number": "nele"}
#: GET_VALUE's VARNAME is CHARACTER(LEN=40): the key is cut to 40 characters before the comparison.
_VARNAME_LEN = 40
#: SCAN_FILE reads each header line into CHARACTER(LEN=400) :: INPLINE; later characters are lost.
_INPLINE_LEN = 400


def _split_header(text: str) -> tuple[str, str] | None:
    """``(key, value)`` of a header line as GET_VALUE sees it, or ``None`` (no ``=``: no data).

    ADJUSTL (spaces only), then commas become blanks, then the ``!`` comment is removed; the key
    is everything before the first ``=``, cut to 40 characters, trailing blanks trimmed.
    """
    t = text.lstrip(" ")
    if not t or t.startswith("!"):
        return None
    t = t.replace(",", " ").split("!", 1)[0]
    key, eq, val = t.partition("=")
    if not eq:
        return None
    return key[:_VARNAME_LEN].rstrip(" "), val


def _header_key(raw: str, lineno: int, path: Path) -> str | None:
    """``"node"`` / ``"nele"`` for a ``Node Number`` / ``Cell Number`` header line, else ``None``.

    A count header whose reading depends on characters beyond column 400 (where FVCOM's
    SCAN_FILE cuts the line) is refused instead of being read differently from FVCOM.
    """
    full = _split_header(raw)
    kf = _COUNT_KEYS.get(full[0]) if full else None
    view = _split_header(raw[:_INPLINE_LEN])
    kv = _COUNT_KEYS.get(view[0]) if view else None
    if (kf or kv) and len(raw.rstrip(" ")) > _INPLINE_LEN:
        raise ValueError(
            f"{path}: line {lineno}: count header is {len(raw.rstrip(' '))} characters long; "
            f"FVCOM's SCAN_FILE reads only the first {_INPLINE_LEN} characters of a header line"
        )
    return kf


def _header_count(raw: str, lineno: int, path: Path) -> int:
    """The count N of a ``Node Number = N`` / ``Cell Number = N`` header line (N >= 1)."""
    kv = _split_header(raw)
    m = _COUNT_VALUE.fullmatch(kv[1]) if kv else None
    if m is None or int(m[1]) < 1:
        raise ValueError(
            f"{path}: line {lineno}: count header {raw.strip()!r} must be "
            "'<name> = N' with N one positive integer"
        )
    return int(m[1])


# <--- jsasaki 2026-11-05


# jsasaki 2026-11-05: review round 16 #3: a record is the text before its first '!' (attached comments too)
def _body(raw: str) -> str:
    """The record part of a line: everything before the first ``!`` comment marker."""
    return raw.split("!", 1)[0]


def _is_comment(raw: str) -> bool:
    """A comment-only line (nothing but blanks before ``!``); blank lines are dropped before this is asked."""
    # jsasaki 2026-11-05: review round 16 #3: same definition as the record body
    return not _body(raw).strip()


# jsasaki 2026-11-05: review round 17 #4: FVCOM reads records list-directed; commas separate fields
def _fields(raw: str) -> list[str]:
    """The list-directed fields of a record (comment removed, commas and blanks as separators)."""
    return list_fields(_body(raw))


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
    with path.open() as fp:
        lines = [(n, ln.rstrip("\n")) for n, ln in enumerate(fp, 1) if ln.strip()]

    # ---------------------------------------- locate counts (node / nele)
    # ---> jsasaki 2026-11-05: review round 15 #5: FVCOM's SCAN_FILE looks each header up on its own
    # (rewinding), so 'Cell Number' may precede 'Node Number'.  The connectivity block starts at the
    # first record that reads as four integers (READ_COLDSTART_GRID); both headers must come before it
    # and within the 200 lines SCAN_FILE scans.
    # jsasaki 2026-11-05: review round 16 #3: tokenise the record without its '!' comment
    # jsasaki 2026-11-05: review round 17 #4: commas separate fields like blanks
    start = next(
        (
            k
            for k, (_n, ln) in enumerate(lines)
            if len(_fields(ln)) >= 4 and all(_is_int(t) for t in _fields(ln)[:4])
        ),
        len(lines),
    )
    counts: dict[str, int] = {}
    for k, (lineno, raw) in enumerate(lines):
        # jsasaki 2026-11-05: review round 17 #2 #3: the complete case-sensitive key of ADJUSTL(TLINE),
        # as SCAN_FILE compares it ('node number', 'Node Number unused', 'Elem Number' are no headers)
        key = _header_key(raw, lineno, path)
        if key is None:
            continue
        if k >= start:
            raise ValueError(
                f"{path}: line {lineno}: count header {raw.strip()!r} follows the start "
                f"of the connectivity block (line {lines[start][0]})"
            )
        if lineno > _SCAN_FILE_LINES:
            # jsasaki 2026-11-05: review round 16 #2: name the last line SCAN_FILE reads (201)
            raise ValueError(
                f"{path}: line {lineno}: count header {raw.strip()!r} lies beyond line "
                f"{_SCAN_FILE_LINES}, the last line that FVCOM's SCAN_FILE reads"
            )
        if key in counts:
            raise ValueError(
                f"{path}: line {lineno}: duplicate count header {raw.strip()!r}"
            )
        counts[key] = _header_count(raw, lineno, path)
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
        # jsasaki 2026-11-05: review round 15 #3: lines carry their line numbers
        # jsasaki 2026-11-05: review round 17 #4: list-directed fields of the comment-free record
        parts = _fields(lines[idx][1])
        idx += 1
        # jsasaki 2026-11-05: review round 12 #3: FVCOM reads J,N1,N2,N3; extra columns are unused
        if len(parts) < 4 or not all(_is_int(t) for t in parts[:4]):
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
    # ---> jsasaki 2026-11-05: review round 15 #3: FVCOM ends the connectivity block at the first
    # record whose id is 1 (the first node record); any other record there is one more element
    # (READ_COLDSTART_GRID stops: rows exceed the stated number of cells).  A further record that
    # reads as an element (four integers) and is not node 1 is reported here; any other misplaced
    # record is reported by the node block's own order check below.
    if not allow_unordered:
        nxt = next((ln for ln in lines[idx:] if not _is_comment(ln[1])), None)
        # jsasaki 2026-11-05: review round 16 #3: tokenise the record without its '!' comment
        # jsasaki 2026-11-05: review round 17 #4: list-directed fields of the comment-free record
        tok = _fields(nxt[1])[:4] if nxt is not None else []
        if len(tok) == 4 and all(_is_int(t) for t in tok) and int(tok[0]) != 1:
            assert nxt is not None
            raise ValueError(
                f"{path}: line {nxt[0]}: record {nxt[1].strip()!r} follows the {nele} "
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
        # jsasaki 2026-11-05: review round 15 #3: lines carry their line numbers
        # jsasaki 2026-11-05: review round 17 #4: list-directed fields of the comment-free record
        parts = _fields(lines[idx][1])
        idx += 1
        # jsasaki 2026-11-05: review round 12 #3: FVCOM reads J,X1,Y1; extra columns are unused
        if len(parts) < 3 or not _is_int(parts[0]):
            continue
        try:
            # jsasaki 2026-11-05: review round 14 #2: Fortran real literals ('4.0D+5', '1.0-3')
            xv = fortran_real(parts[1])
            yv = fortran_real(parts[2])
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
