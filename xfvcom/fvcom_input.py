"""FVCOM input-file header and record scanning shared by the mesh, depth and OBC readers.

One model of how FVCOM reads its ASCII input files (``mod_utils.F`` ``SCAN_FILE`` /
``GET_VALUE``, ``mod_input.F`` ``READ_COLDSTART_*``), used by
:func:`xfvcom.grid.read_dat`, :func:`xfvcom.io.read_dep`, the strict depth / OBC parsers
of ``scripts/run_jcope_obc.py`` and the numeric-line validation of
:func:`xfvcom.io.read_sigma_dat` (review round 18: the readers used to carry their own
regular expressions, each with different gaps).

Headers (``Node Number = 3210``, ``OBC Node Number = 13``):

* ``SCAN_FILE`` reads each physical line into ``CHARACTER(LEN=400) :: INPLINE``; a longer
  line is cut at byte 400.  Lines are modelled as their UTF-8 bytes (one ``str`` character
  per byte, via latin-1), so a multibyte character counts as Fortran counts it.
* ``GET_VALUE`` gets ``ADJUSTL`` of the line (leading SPACES removed), turns commas into
  blanks, removes the ``!`` comment, cuts the key before ``=`` to 40 bytes (``VARNAME``) and
  types the value: ``string`` when it holds any character outside ``0123456789+-Ee. `` (a
  tab included), ``float`` with a ``.``, ``integer`` otherwise -- and converts every token
  of a numeric value with an internal ``READ`` without ``IOSTAT``, for EVERY line it passes,
  before comparing the key.
* A count header is accepted only when the 400-byte cut does not change what it says: an
  overlong comment after the value is harmless, a key, ``=`` or value beyond byte 400 is
  refused.

Records (list-directed ``READ(unit,*)``, review round 19): modelled as READs, not lines
(:func:`list_read`).  Blanks and commas separate values; ``r*value`` repeats a value (up to
gfortran's limit, consumed lazily); a null value or an early ``/`` leaves an item unset; a READ
continues onto further lines until its list is satisfied and ignores the rest of that line.  The
data block starts where FVCOM's header search first READs successfully, and its NHEADER rewind is
checked (:func:`locate_records`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# jsasaki 2026-11-05: review round 19 #2 #4: unexpanded list items and the repeat-count limit
from .fortran_numbers import (
    _FORTRAN_REAL,
    MAX_REPEAT,
    fortran_real,
    list_fields,
    list_items,
)

__all__ = [
    "SCAN_FILE_LINES",
    "INPLINE_BYTES",
    "VARNAME_BYTES",
    "fortran_bytes",
    "get_value",
    "fvcom_type",
    "check_numeric_line",
    "scan_count_headers",
    # jsasaki 2026-11-05: review round 19 #1 #4 #5: READ-based record scanning
    "INT",
    "REAL",
    "ListRead",
    "list_read",
    "read_all",
    "locate_records",
    "record_body",
    "is_comment_only",
    "record_fields",
]

#: FVCOM's default 32-bit INTEGER range (the same values as :mod:`xfvcom.grid.gtsz`).
FORTRAN_INT_MIN, FORTRAN_INT_MAX = -(2**31), 2**31 - 1
#: SCAN_FILE tests NLINE > 200 BEFORE reading, so line 201 is the last line it reads.
SCAN_FILE_LINES = 201
#: SCAN_FILE's ``CHARACTER(LEN=400) :: INPLINE`` line buffer (bytes).
INPLINE_BYTES = 400
#: GET_VALUE's ``CHARACTER(LEN=40) :: VARNAME`` key buffer (bytes).
VARNAME_BYTES = 40
#: GET_VALUE's REALVAL/INTVAL/STRINGVAL token buffers (a longer list is a FATAL_ERROR).
MAX_VALUES = 150
#: GET_VALUE's NUMCHARS: a value holding any other character is typed 'string'.
NUMCHARS = frozenset("0123456789+-Ee. ")
#: SCAN_FILE's continuation suffix (two backslashes in every build's test).
_CONT_SUFFIX = "\\\\"
#: A count value: ONE integer token with an optional sign (FVCOM's READ(TEMP,*) INTVAL).
_COUNT_VALUE = re.compile(r"[+-]?[0-9]+")


# ---------------------------------------------------------------------------
# GET_VALUE
# ---------------------------------------------------------------------------
def fortran_bytes(raw: str) -> str:
    """``raw`` as the bytes a Fortran ``READ '(a)'`` stores: its UTF-8 encoding, one ``str``
    character per byte (decoded as latin-1), so lengths and cuts are byte counts."""
    return raw.encode("utf-8").decode("latin-1")


def get_value(text: str) -> tuple[str, str] | None:
    """``(VARNAME, VARVAL)`` of a line as ``GET_VALUE`` splits it, or ``None`` ("no data").

    ``text`` is the line in :func:`fortran_bytes` form.  ADJUSTL (spaces only), blank or
    comment line -> no data; commas -> blanks; the ``!`` comment removed; no ``=`` -> no
    data; the key cut to 40 bytes and right-trimmed of spaces; the value trimmed of spaces
    (not tabs); an empty value -> no data.
    """
    t = text.lstrip(" ")
    if not t.rstrip(" ") or t.startswith("!"):
        return None
    t = t.replace(",", " ").split("!", 1)[0]
    key, eq, val = t.partition("=")
    if not eq:
        return None
    val = val.strip(" ")
    if not val:
        return None
    return key[:VARNAME_BYTES].rstrip(" "), val


def fvcom_type(val: str) -> str:
    """The type ``GET_VALUE`` assigns to a normalised value (commas blanked, comment removed,
    space-trimmed): ``logical``, ``string``, ``float`` or ``integer``."""
    if val in ("T", "F"):
        return "logical"
    if any(c not in NUMCHARS for c in val):
        return "string"
    return "float" if "." in val else "integer"


def bad(path: Path, key: str, val: str, need: str) -> ValueError:
    return ValueError(f"{path}: '{key} = {val}': {need}")


def with_point(tok: str) -> str:
    """``tok`` with a decimal point in its mantissa ('8' -> '8.0', '8e-4' ->
    '8.0e-4', '1-3' -> '1.0-3'), the literal FVCOM types as 'float' with the
    same value.  A token that is not a real literal is returned with '.0'
    appended."""
    m = _FORTRAN_REAL.fullmatch(tok)
    i = m.end("mant") if m is not None else len(tok)
    return f"{tok[:i]}.0{tok[i:]}"


def check_numeric_line(path: Path, lineno: int, key: str, val: str) -> None:
    """Reject a numeric key/value line FVCOM's ``GET_VALUE`` cannot read.

    ``SCAN_FILE`` (mod_utils.F) calls ``GET_VALUE`` on every line it passes
    BEFORE comparing the key, and ``GET_VALUE`` converts every token of a
    numeric value with an internal list-directed ``READ`` without ``IOSTAT``:
    into the 32-bit ``INTEGER`` buffer when the value has no decimal point, into
    the ``REAL(DP)`` buffer otherwise.  A token that does not convert therefore
    stops FVCOM's reader even when its key is never used (an inactive
    ``GTSZ ZLEV`` / ``GTSZ LOWFRAC``, an unknown key such as ``mesh_scale = 1e3``).
    """
    typ = fvcom_type(val)
    if typ not in ("integer", "float"):
        return
    why = (
        "; FVCOM's SCAN_FILE converts every numeric line it passes before "
        "matching the key, so it stops here even when the key is unused"
    )
    for tok in val.split():
        if typ == "integer":
            try:
                iv = int(tok)
            except ValueError:
                hint = ""
                try:
                    fortran_real(tok)
                    hint = (
                        f" -- for a floating value write it with a decimal point, "
                        f"e.g. '{with_point(tok)}'"
                    )
                except ValueError:
                    pass
                raise bad(
                    path,
                    key,
                    val,
                    f"line {lineno}: '{tok}' is not a valid integer (a value without "
                    f"a decimal point is read as an INTEGER){why}{hint}",
                ) from None
            if not FORTRAN_INT_MIN <= iv <= FORTRAN_INT_MAX:
                raise bad(
                    path,
                    key,
                    val,
                    f"line {lineno}: '{tok}' is outside FVCOM's 32-bit INTEGER "
                    f"range [{FORTRAN_INT_MIN}, {FORTRAN_INT_MAX}] (a value without "
                    f"a decimal point is read as an INTEGER){why}",
                )
        else:
            if _FORTRAN_REAL.fullmatch(tok) is None:
                raise bad(
                    path,
                    key,
                    val,
                    f"line {lineno}: '{tok}' is not a valid floating value (not a "
                    f"Fortran real literal){why}",
                )
            try:
                fortran_real(tok)
            except ValueError:
                raise bad(
                    path,
                    key,
                    val,
                    f"line {lineno}: '{tok}' is not finite (overflows REAL(DP): "
                    "Intel Fortran stops on it, gfortran reads Infinity); FVCOM's "
                    "SCAN_FILE converts every numeric line it passes before matching "
                    "the key, even when the key is unused",
                ) from None


# ---------------------------------------------------------------------------
# SCAN_FILE count headers
# ---------------------------------------------------------------------------
def _views(raw: str) -> tuple[tuple[str, str] | None, tuple[str, str] | None]:
    """GET_VALUE of the complete line and of FVCOM's 400-byte INPLINE buffer."""
    b = fortran_bytes(raw)
    return get_value(b), get_value(b[:INPLINE_BYTES])


def _count_value(path: Path, lineno: int, raw: str, val: str) -> int:
    """The positive count of a count header's normalised value, or ValueError."""
    if any(c.isspace() and c != " " for c in val):
        raise ValueError(
            f"{path}: line {lineno}: count header {raw.strip()!r} holds a tab or other "
            "non-space whitespace in its value; FVCOM's GET_VALUE trims only spaces and "
            "types such a value as a string, which SCAN_FILE(ISCAL=...) refuses -- use spaces"
        )
    if _COUNT_VALUE.fullmatch(val) is None or not 1 <= int(val) <= FORTRAN_INT_MAX:
        raise ValueError(
            f"{path}: line {lineno}: count header {raw.strip()!r} must be "
            "'<name> = N' with N one positive integer"
        )
    return int(val)


def scan_count_headers(
    path: Path,
    lines: Sequence[tuple[int, str]],
    keys: Mapping[str, str],
    data_start: int,
    block: str,
) -> dict[str, int]:
    """Look up FVCOM count headers the way ``SCAN_FILE(..., ISCAL=...)`` does.

    ``lines`` are the file's non-blank lines as ``(physical line number, text)``;
    ``keys`` maps each exact, case-sensitive FVCOM key (``"Node Number"``) to the name it
    is returned under; ``data_start`` is the index in ``lines`` of the first data record
    and ``block`` names that block in messages.  Returns the counts found (a missing key is
    left out: the caller names it).  Raises :class:`ValueError` for a count header that
    the 400-byte line buffer would read differently, lies in or after the data block or
    beyond line 201, appears twice, or whose value is not one positive integer padded with
    spaces only; and for any line FVCOM's lookups traverse (every line up to the last
    count header found) that GET_VALUE could not convert or that continues onto the next
    line.
    """
    counts: dict[str, int] = {}
    last = 0
    for k, (lineno, raw) in enumerate(lines):
        full, seen = _views(raw)
        kf = keys.get(full[0]) if full else None
        ks = keys.get(seen[0]) if seen else None
        if (kf or ks) and full != seen:
            raise ValueError(
                f"{path}: line {lineno}: count header is {len(fortran_bytes(raw))} bytes "
                f"long and FVCOM's SCAN_FILE reads only the first {INPLINE_BYTES} "
                "characters (bytes, UTF-8) of a header line, which changes what it reads "
                "-- shorten the line before the value and its '!' comment"
            )
        if kf is None:
            continue
        assert full is not None
        if k >= data_start:
            raise ValueError(
                f"{path}: line {lineno}: count header {raw.strip()!r} follows the start "
                f"of the {block} (line {lines[data_start][0]})"
            )
        if lineno > SCAN_FILE_LINES:
            raise ValueError(
                f"{path}: line {lineno}: count header {raw.strip()!r} lies beyond line "
                f"{SCAN_FILE_LINES}, the last line that FVCOM's SCAN_FILE reads"
            )
        if kf in counts:
            raise ValueError(
                f"{path}: line {lineno}: duplicate count header {raw.strip()!r}"
            )
        counts[kf] = _count_value(path, lineno, raw, full[1])
        last = lineno
    # every line the lookups traverse goes through GET_VALUE's unconditional conversion
    for lineno, raw in lines:
        if lineno > last:
            break
        buf = fortran_bytes(raw)[:INPLINE_BYTES]
        t = buf.rstrip(" ")
        if len(t) > 1 and t.endswith(_CONT_SUFFIX):
            raise ValueError(
                f"{path}: line {lineno} ends in a line-continuation suffix '\\\\'; FVCOM's "
                "SCAN_FILE joins the next line onto it, which can hide a header -- remove it"
            )
        kv = get_value(buf)
        if kv is None:
            continue
        key, val = kv
        ntok = len(val.split())
        if ntok > MAX_VALUES or (ntok >= MAX_VALUES and fvcom_type(val) == "string"):
            raise ValueError(
                f"{path}: line {lineno} ('{key}') carries {ntok} values; FVCOM's "
                f"GET_VALUE holds at most {MAX_VALUES} and stops with a FATAL_ERROR"
            )
        check_numeric_line(path, lineno, key, val)
    return counts


# ---------------------------------------------------------------------------
# list-directed records
# ---------------------------------------------------------------------------
def record_body(raw: str) -> str:
    """The record part of a line: everything before the first ``!`` comment marker."""
    return raw.split("!", 1)[0]


def is_comment_only(raw: str) -> bool:
    """True for a line with nothing but blanks before ``!`` (or a blank line)."""
    return not record_body(raw).strip()


def record_fields(raw: str) -> list[str]:
    """The list-directed fields of a record (comment removed, ``r*value`` expanded)."""
    return list_fields(record_body(raw))


# ---> jsasaki 2026-11-05: review round 19 #1 #3 #4 #5: FVCOM's list-directed READs modelled as
# reads (not physical lines): the header search for the first READ that succeeds, the NHEADER rewind,
# '/' terminators and records continued onto further lines
#: The item types of a list-directed READ list.
INT, REAL = "integer", "real"
_INT_LITERAL = re.compile(r"[+-]?[0-9]+")


@dataclass(frozen=True)
class ListRead:
    """One list-directed ``READ(unit,*,IOSTAT=IOS) v1, v2, ...`` over the physical lines.

    ``status`` is ``"ok"`` (IOS = 0, every item set), ``"unset"`` (IOS = 0 but a null value or a
    ``/`` left an item unchanged), ``"error"`` (IOS > 0, a conversion error; FVCOM's data loops
    count such a READ as a row) or ``"eof"`` (IOS < 0).  ``start`` / ``end`` are the indices of
    the first and last line consumed (``end`` inclusive; the next READ starts at ``end + 1``), and
    ``first_line`` / ``last_line`` their physical line numbers.
    """

    status: str
    start: int
    end: int
    first_line: int
    last_line: int
    values: tuple[int | float | None, ...]
    message: str = ""


def _convert(tok: str, typ: str) -> int | float:
    if typ == INT:
        if _INT_LITERAL.fullmatch(tok) is None:
            raise ValueError(f"'{tok}' is not a valid integer")
        iv = int(tok)
        if not FORTRAN_INT_MIN <= iv <= FORTRAN_INT_MAX:
            raise ValueError(f"'{tok}' is outside the 32-bit INTEGER range")
        return iv
    return fortran_real(tok)


def list_read(
    lines: Sequence[tuple[int, str]], pos: int, types: Sequence[str]
) -> ListRead:
    """Model one list-directed READ of ``types`` starting at ``lines[pos]``.

    ``lines`` are ALL physical lines as ``(line number, text)``, blank ones included.  As in the
    Fortran runtime (checked with gfortran 8.5 and ifx 2025.1, 2026-11-05): a READ that still needs
    values continues onto the next line (blank lines included); once the list is satisfied the rest
    of the line is ignored; a conversion error ends the READ (IOS > 0) and the next READ starts on
    the following line; ``/`` (attached or separated) ends the READ, leaving the remaining items
    unchanged; a null value leaves its item unchanged; end of file before the list is satisfied is
    IOS < 0.  A ``!`` comment is not list-directed syntax: once the list is satisfied it is ignored
    with the rest of the line (as these readers always did), but a ``!`` reached while values are
    still needed is read as a value and fails, as in Fortran.  A repetition ``r*v`` above
    :data:`~xfvcom.fortran_numbers.MAX_REPEAT` is a conversion error (gfortran refuses it).
    """
    vals: list[int | float | None] = []
    need = len(types)
    first = lines[pos][0] if pos < len(lines) else 0

    def done(status: str, k: int, msg: str = "") -> ListRead:
        return ListRead(status, pos, k, first, lines[k][0], tuple(vals), msg)

    for k in range(pos, len(lines)):
        raw = lines[k][1]
        body = record_body(raw)
        has_comment = len(body) < len(raw)
        pre, slash, _rest = body.partition("/")
        for r, tok in list_items(pre):
            if r > MAX_REPEAT:
                return done(
                    "error",
                    k,
                    f"repetition count {r} exceeds {MAX_REPEAT}, the largest the Fortran "
                    "runtime (gfortran) accepts; it stops on this record",
                )
            for _ in range(min(r, need - len(vals))):
                if tok == "":
                    vals.append(None)
                    continue
                try:
                    vals.append(_convert(tok, types[len(vals)]))
                except ValueError as exc:
                    return done(
                        "error",
                        k,
                        f"{exc} (item {len(vals) + 1} is {types[len(vals)]})",
                    )
            if len(vals) == need:
                if any(v is None for v in vals):
                    return done("unset", k, "a null value leaves an item unset")
                return done("ok", k)
        if slash:
            return done(
                "unset",
                k,
                f"'/' ends the READ after {len(vals)} of the {need} values it needs",
            )
        if has_comment:
            return done(
                "error",
                k,
                f"the '!' comment is read as a value after only {len(vals)} of the {need} "
                "values (a list-directed READ has no comments)",
            )
    end = len(lines) - 1
    return ListRead("eof", pos, end, first, lines[end][0] if lines else 0, tuple(vals))


def read_all(
    lines: Sequence[tuple[int, str]], start: int, types: Sequence[str]
) -> tuple[list[ListRead], ListRead]:
    """The READs of FVCOM's data loop from ``start`` to end of file: every READ with IOS >= 0
    (``ok``, ``unset`` and ``error`` alike, as READ_COLDSTART_OBC_GRID / _DEPTH count them) and
    the final end-of-file READ (whose ``values`` are non-empty for an incomplete last record).
    """
    reads: list[ListRead] = []
    pos = start
    while True:
        r = list_read(lines, pos, types)
        if r.status == "eof":
            return reads, r
        reads.append(r)
        pos = r.end + 1


def locate_records(
    path: Path,
    lines: Sequence[tuple[int, str]],
    types: Sequence[str],
    block: str,
    failed_out: list[ListRead] | None = None,
) -> int:
    """Index in ``lines`` (all physical lines) where FVCOM's data loop starts reading ``block``.

    FVCOM's header search (mod_input.F READ_COLDSTART_GRID / _OBC_GRID / _DEPTH, the
    ``lwang intel23@20230710`` loops) repeats ``READ(unit,*,IOSTAT=IOS)`` of the record's items
    until one succeeds (IOS = 0; a title, a ``key = value`` line or a comment simply fails and is
    passed), counting the failed READs in NHEADER; it then REWINDs and skips NHEADER lines with
    ``READ(unit,*)``.  A failed READ that consumed more than one line (a blank line, or a line
    that supplied some of the values, before the line it failed on) makes NHEADER smaller than the
    number of header lines, so the data loop would start inside the header and read a header line
    as a data row: such input is refused here, naming the lines.  Returns ``len(lines)`` when no
    READ succeeds.  The failed header READs are appended to ``failed_out`` when it is given (a
    malformed FIRST record is one of them: FVCOM's search passes it as a header line).
    """
    pos = 0
    failed: list[ListRead] = [] if failed_out is None else failed_out
    while True:
        r = list_read(lines, pos, types)
        if r.status == "eof":
            return len(lines)
        if r.status in ("ok", "unset"):
            break
        failed.append(r)
        pos = r.end + 1
    start, nheader = r.start, len(failed)
    stray = [n for n, raw in lines[nheader:start] if raw.strip()]
    if stray:
        why = []
        for f in failed:
            if f.end > f.start:
                kinds = ", ".join(
                    f"{'blank line' if not raw.strip() else 'line'} {n}"
                    for n, raw in lines[f.start : f.end]
                )
                why.append(
                    f"{kinds} (absorbed into the failed READ of line {f.last_line})"
                )
        raise ValueError(
            f"{path}: {'; '.join(why)}: FVCOM's header search reads such lines together with "
            f"the next one as ONE failed READ and counts one header line (NHEADER = {nheader}) "
            f"for them, so after the rewind it starts the {block} at line "
            f"{lines[nheader][0]} and reads header line {stray[0]} as a data row -- remove the "
            "blank line(s) before the header lines"
        )
    return start


# <--- jsasaki 2026-11-05
