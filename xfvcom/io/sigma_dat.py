"""Reader/writer for the FVCOM ``<casename>_sigma.dat`` vertical-coordinate file.

Focus is the ``SIGMA COORDINATE TYPE = SIGMAZ`` block (the generalized
terrain-following sigma-z / GTSZ coordinate), parsed to / emitted from a
:class:`~xfvcom.grid.gtsz.GtszSpec`. The base types (``UNIFORM`` / ``GEOMETRIC``
/ ``TANH`` / ``GENERALIZED`` / ``S_COORDINATE``) are read enough to recover the
level count + type (useful when comparing against a production baseline) but are
not generated here.

The on-disk format is the FVCOM ``SCAN_FILE`` key/value convention -- one
``KEY = VALUE`` per line, ``!`` starts a comment, booleans are ``T`` / ``F``,
vectors (``GTSZ ZLEV``) are space-separated. Authoritative parser:
``FVCOM/src/mod_input.F::READ_COLDSTART_SIGMA`` (``case(STYPE_SIGMAZ)``).

Example (a slope-adaptive Tokyo-Bay header)::

    NUMBER OF SIGMA LEVELS = 31
    SIGMA COORDINATE TYPE = SIGMAZ
    GTSZ BASE = 2
    GTSZ K1 = 10
    GTSZ K2 = 31
    GTSZ P1 = 2.0
    GTSZ L1 = 1.0
    GTSZ L2 = 1.0
    GTSZ NZ = 20
    GTSZ ZLEV = -2.0 -4.0 ... -40.0
    GTSZ MASK = T
    GTSZ SADAPT = T
    GTSZ SMAX = 0.000800
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# jsasaki 2026-11-05: review round 11 #2: FVCOM's 32-bit INTEGER range
from ..grid.gtsz import FORTRAN_INT_MAX, FORTRAN_INT_MIN, GtszSpec

__all__ = ["SigmaFile", "read_sigma_dat", "write_sigma_dat", "gtsz_to_lines"]


@dataclass
class SigmaFile:
    """A parsed ``*_sigma.dat`` header."""

    kb: int  #: NUMBER OF SIGMA LEVELS
    # jsasaki 2026-11-05: review round 8 #4: exact FVCOM spelling (no upper-casing)
    stype: str  #: SIGMA COORDINATE TYPE (one of FVCOM's case-sensitive names)
    gtsz: GtszSpec | None = None  #: the GTSZ spec when ``stype == 'SIGMAZ'``
    raw: dict[str, str] | None = None  #: all parsed KEY -> raw-value strings


# jsasaki 2026-11-05: review round 7 #2: the reader below mirrors FVCOM's
# SCAN_FILE / GET_VALUE (mod_utils.F) value typing instead of Python's lenient
# float()/int(): FVCOM types a value as 'logical' (exactly "T"/"F"), 'string'
# (any character outside "0123456789+-Ee. "), 'float' (a '.' anywhere in the
# value) or 'integer', and a request of another type returns -3, which
# READ_COLDSTART_SIGMA (mod_input.F) turns into a silent default for most GTSZ
# keys (e.g. "GTSZ P1 = 4" runs with P1 = 2.0).  Every form that FVCOM would
# read differently from its face value is REJECTED here with the fix to apply.
_NUMCHARS = frozenset("0123456789+-Ee. ")

#: SIGMAZ keys and the SCAN_FILE request type FVCOM uses for each
#: (mod_input.F READ_COLDSTART_SIGMA, case(STYPE_SIGMAZ)).
_INT_KEYS = ("GTSZ BASE", "GTSZ NZ", "GTSZ K1", "GTSZ K2", "BPG REF NZ")
_FLOAT_KEYS = (
    "GTSZ P1",
    "GTSZ L1",
    "GTSZ L2",
    "GTSZ SMOOTH",
    "GTSZ SMAX",
    "GTSZ DREF",
)
_BOOL_KEYS = ("GTSZ MASK", "GTSZ SADAPT", "GTSZ LOWSIG", "GTSZ DYE_NOWALL")
_FVEC_KEYS = ("GTSZ ZLEV", "GTSZ LOWFRAC", "BPG REF ZLEV")
_KNOWN_KEYS = (
    ("NUMBER OF SIGMA LEVELS", "SIGMA COORDINATE TYPE")
    + _INT_KEYS
    + _FLOAT_KEYS
    + _BOOL_KEYS
    + _FVEC_KEYS
)

# ---> jsasaki 2026-11-05: review round 8 #3 #4 #5: FVCOM scanner limits and
# the coordinate types it accepts (mod_utils.F SCAN_FILE / GET_VALUE,
# mod_input.F READ_COLDSTART_SIGMA, mod_main.F STYPE_*)
#: physical line buffer of SCAN_FILE (CHARACTER(LEN=400) INPLINE): READ '(a)'
#: silently truncates a longer line
_MAX_LINE = 400
#: SCAN_FILE gives up ("Read 200 lines of header...") once NLINE > 200, after
#: having read line 201: a key on line 202 or later is never found
_MAX_HEADER_LINE = 201
#: GET_VALUE's REALVAL/INTVAL token buffers (a longer list is a FATAL_ERROR)
_MAX_VALUES = 150
#: GET_VALUE's VARNAME is CHARACTER(LEN=40): longer key text is truncated
_MAX_KEY = 40
#: SIGMA COORDINATE TYPE values FVCOM selects (case-sensitive SELECT CASE)
_STYPES = ("UNIFORM", "GEOMETRIC", "TANH", "GENERALIZED", "S_COORDINATE", "SIGMAZ")
# <--- jsasaki 2026-11-05
# ---> jsasaki 2026-11-05: review round 9 #4: SCAN_FILE's line-continuation
# suffix (mod_utils.F ENDLINE): two backslashes in the IFORT/COMPAQ/IRIX build,
# four otherwise.  A line whose space-trimmed text ends in two backslashes is
# continued by every build (the four-backslash suffix ends in two as well), and
# FVCOM joins the next line onto it BEFORE looking for comments or keys.
_CONT_SUFFIX = "\\\\"  # two characters: backslash backslash
# <--- jsasaki 2026-11-05


def _is_continued(line: str) -> bool:
    """True when FVCOM's SCAN_FILE would join the next line onto ``line``."""
    t = line.rstrip(" ")
    return len(t) > 1 and t.endswith(_CONT_SUFFIX)


def _fvcom_buffer(line: str) -> str:
    """``line`` as it lands in SCAN_FILE's ``CHARACTER(LEN=400) INPLINE`` after
    ``READ(UNIT,'(a)')``: the first 400 bytes (a longer line is truncated)."""
    return line.encode()[:_MAX_LINE].decode("utf-8", errors="replace")


def _is_continued_fvcom(line: str) -> bool:
    """True when FVCOM would continue ``line`` either as written or after its
    400-byte truncation (SCAN_FILE tests the suffix on the TRUNCATED buffer)."""
    # jsasaki 2026-11-05: review round 10 #1: the suffix test runs on the truncated line
    return _is_continued(line) or _is_continued(_fvcom_buffer(line))


def _fvcom_type(val: str) -> str:
    """The type FVCOM's ``GET_VALUE`` assigns to a value already normalised by
    :func:`read_sigma_dat` (commas blanked, comment removed, space-trimmed)."""
    # jsasaki 2026-11-05: review round 8 #2: the value arrives normalised ONCE
    # (commas -> blanks) and the same text is used for typing and conversion
    if val in ("T", "F"):
        return "logical"
    if any(c not in _NUMCHARS for c in val):
        return "string"
    return "float" if "." in val else "integer"


def _scan(pairs: dict[str, str], key: str) -> str | None:
    """FVCOM ``SCAN_FILE`` key lookup (exact key text, first occurrence)."""
    return pairs.get(key)


def _bad(path: Path, key: str, val: str, need: str) -> ValueError:
    return ValueError(f"{path}: '{key} = {val}': {need}")


def _get_int(path: Path, pairs: dict[str, str], key: str) -> int | None:
    val = _scan(pairs, key)
    if val is None:
        return None
    # jsasaki 2026-11-05: review round 8 #2: val is already comma-normalised
    if _fvcom_type(val) != "integer" or len(val.split()) != 1:
        raise _bad(
            path,
            key,
            val,
            "FVCOM reads this key as ONE integer without a decimal point "
            "(e.g. '31', not '31.0'); any other form is not read as written",
        )
    try:
        # jsasaki 2026-11-05: review round 11 #2: keep the value for the range check
        iv = int(val.split()[0])
    except ValueError:
        raise _bad(path, key, val, "not a valid integer") from None
    # jsasaki 2026-11-05: review round 11 #2: GET_VALUE reads into a 32-bit INTEGER
    _check_int32(path, key, val, iv)
    return iv


# jsasaki 2026-11-05: review round 11 #2: new helper
def _check_int32(path: Path, key: str, val: str, iv: int) -> None:
    """Reject an integer FVCOM's ``GET_VALUE`` cannot hold: it reads the value
    into a default 32-bit ``INTEGER`` with an internal ``READ`` without
    ``IOSTAT``, so an out-of-range value stops FVCOM's reader."""
    if not FORTRAN_INT_MIN <= iv <= FORTRAN_INT_MAX:
        raise _bad(
            path,
            key,
            val,
            f"outside FVCOM's 32-bit INTEGER range [{FORTRAN_INT_MIN}, "
            f"{FORTRAN_INT_MAX}]; FVCOM's reader stops on it",
        )


#: FVCOM's fallback for a floating key it cannot read (mod_input.F)
_FLOAT_DEFAULTS = {
    "GTSZ P1": 2.0,
    "GTSZ L1": 1.0,
    "GTSZ L2": 1.0,
    "GTSZ SMOOTH": 0.0,
    "GTSZ SMAX": 0.0,
    "GTSZ DREF": 0.0,
}


def _get_float(path: Path, pairs: dict[str, str], key: str) -> float | None:
    val = _scan(pairs, key)
    if val is None:
        return None
    # jsasaki 2026-11-05: review round 8 #2: val is already comma-normalised
    toks = val.split()
    typ = _fvcom_type(val)
    # FVCOM re-scans a decimal-less GTSZ DREF as an integer (mod_input.F), so
    # DREF accepts the integer form.  For the other keys FVCOM falls back to its
    # default; a decimal-less value EQUAL to that default therefore means the
    # same thing to both readers and is accepted (older files write
    # "GTSZ P1 = 2"); any other decimal-less value is rejected.
    ok = typ == "float" or (key == "GTSZ DREF" and typ == "integer")
    if not ok and typ == "integer" and len(toks) == 1:
        try:
            ok = int(toks[0]) == _FLOAT_DEFAULTS[key]
        except ValueError:
            ok = False
    if not ok or len(toks) != 1:
        lit = toks[0] if toks else val
        raise _bad(
            path,
            key,
            val,
            (
                "FVCOM reads this key as ONE floating value and only recognises it "
                "when it contains a decimal point; without one FVCOM silently uses "
                f"its default {_FLOAT_DEFAULTS[key]} instead -- write e.g. '{lit}.0'"
                if typ == "integer" and len(toks) == 1
                else "FVCOM reads this key as ONE floating value with a decimal point"
            ),
        )
    try:
        x = float(toks[0])
    except ValueError:
        raise _bad(path, key, val, "not a valid floating value") from None
    # jsasaki 2026-11-05: review round 11 #2: an integer-form DREF is read as an INTEGER
    if typ == "integer":
        try:
            ival = int(toks[0])
        except ValueError:
            raise _bad(
                path, key, val, "FVCOM reads this decimal-less form as an INTEGER"
            ) from None
        _check_int32(path, key, val, ival)
    # jsasaki 2026-11-05: review round 10 #4: an overflowing literal (1.0e999) converts to inf
    if not np.isfinite(x):
        raise _bad(path, key, val, "the value is not finite (overflow?)")
    return x


def _get_bool(path: Path, pairs: dict[str, str], key: str) -> bool | None:
    val = _scan(pairs, key)
    if val is None:
        return None
    if _fvcom_type(val) != "logical":
        raise _bad(
            path,
            key,
            val,
            "FVCOM recognises only a bare 'T' or 'F' here; any other form "
            "silently becomes F",
        )
    # jsasaki 2026-11-05: review round 8 #2: compare the normalised value ("T," is T)
    return val == "T"


def _get_fvec(path: Path, pairs: dict[str, str], key: str) -> tuple[float, ...] | None:
    val = _scan(pairs, key)
    if val is None:
        return None
    # jsasaki 2026-11-05: review round 8 #2: val is already comma-normalised
    toks = val.split()
    if _fvcom_type(val) != "float":
        raise _bad(
            path,
            key,
            val,
            "FVCOM reads this key as a list of floating values and needs at "
            "least one decimal point in it (e.g. '-2.0 -4.0')",
        )
    if len(toks) < 2:
        raise _bad(
            path,
            key,
            val,
            "FVCOM does not read a one-value list for this key (SCAN_FILE "
            "returns -4, i.e. the key counts as absent)",
        )
    try:
        xs = tuple(float(t) for t in toks)
    except ValueError:
        raise _bad(path, key, val, "not a list of valid floating values") from None
    # jsasaki 2026-11-05: review round 10 #4: an overflowing literal (1.0e999) converts to inf
    if not np.isfinite(xs).all():
        raise _bad(path, key, val, "a value is not finite (overflow?)")
    return xs


def read_sigma_dat(path: str | Path) -> SigmaFile:
    """Parse a ``*_sigma.dat`` file into a :class:`SigmaFile`.

    Values are typed as FVCOM's ``SCAN_FILE`` types them; a SIGMAZ value
    that FVCOM would not read as written (a decimal-less floating value, a
    decimal integer, a logical other than ``T``/``F``, a one-value list, a key
    differing from FVCOM's spelling only in case/spacing, a repeated key, a tab
    or carriage return in a key/value line, a coordinate type not spelled
    exactly as FVCOM's, a key beyond line 201, a line over 400 characters, a
    list over 150 values) raises :class:`ValueError` naming the key and the fix.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    known_norm = {k: k for k in _KNOWN_KEYS}
    pairs: dict[str, str] = {}
    # ---> jsasaki 2026-11-05: review round 8 #2 #3 #5: reproduce SCAN_FILE /
    # GET_VALUE line handling: physical lines as FVCOM reads them (no newline
    # translation, so a CR is seen), commas blanked ONCE for the whole line,
    # trimming of SPACES only (ADJUSTL / TRIM), the 400-character line buffer,
    # the 201-line header search and the 150-value token buffers.  Forms that
    # FVCOM reads differently from Python are rejected.
    with path.open(newline="") as fp:
        text_lines = fp.read().split("\n")
    if text_lines and text_lines[-1] == "":
        text_lines.pop()
    for lineno, raw_line in enumerate(text_lines, start=1):
        # ---> jsasaki 2026-11-05: review round 9 #4: continuations are joined
        # before comments are recognised, so a continued comment swallows the
        # next line (a key there is lost to FVCOM); not reproduced -- rejected
        # jsasaki 2026-11-05: review round 10 #1: also after the 400-byte truncation
        if _is_continued_fvcom(raw_line):
            raise ValueError(
                f"{path}: line {lineno} ends in a backslash pair, FVCOM's line-"
                "continuation suffix (as written, or after FVCOM truncates the "
                f"line to its {_MAX_LINE}-character buffer); FVCOM joins the next "
                "line onto it (onto a comment, too, which hides the next key) -- "
                "remove the trailing backslashes or shorten the line"
            )
        # <--- jsasaki 2026-11-05
        # jsasaki 2026-11-05: review round 9 #3: SCAN_FILE passes ADJUSTL(TLINE)
        lead = raw_line.lstrip(" ")
        if not lead.rstrip(" ") or lead.startswith("!"):
            continue  # GET_VALUE: blank or comment line
        # jsasaki 2026-11-05: review round 9 #3: comment/key/value split on the adjusted line
        bang = lead.find("!")
        body = lead if bang < 0 else lead[:bang]
        if "=" not in body:
            continue  # GET_VALUE: no '=' -> "no data"
        if len(raw_line.encode()) > _MAX_LINE:
            raise ValueError(
                f"{path}: line {lineno} is {len(raw_line.encode())} characters long; "
                f"FVCOM's SCAN_FILE reads only the first {_MAX_LINE} and would "
                "silently truncate it"
            )
        if any(c.isspace() and c != " " for c in body):
            raise ValueError(
                f"{path}: line {lineno} contains a tab, carriage return or other "
                "non-space whitespace; FVCOM trims only spaces and reads such a "
                "character as part of the key or value (a numeric value becomes a "
                "string and is silently replaced by the default) -- use spaces and "
                "Unix line endings"
            )
        # jsasaki 2026-11-05: review round 9 #3: FVCOM's order -- ADJUSTL first
        # (above), THEN commas -> blanks, and no second ADJUSTL of the key: a
        # leading comma leaves a leading blank in the key, which then matches no
        # FVCOM key (rejected below as a misspelling of a known key)
        body = body.replace(",", " ")
        key, _, val = body.partition("=")
        # VARNAME = TEXT_LINE(1:EQLOC-1) into CHARACTER(40), compared after TRIM
        key = key[:_MAX_KEY].rstrip(" ")
        val = val.strip(" ")
        if not key or not val:
            continue  # GET_VALUE: "no data"
        ntok = len(val.split())
        if lineno <= _MAX_HEADER_LINE and (
            ntok > _MAX_VALUES or (ntok >= _MAX_VALUES and _fvcom_type(val) == "string")
        ):
            raise ValueError(
                f"{path}: line {lineno} ('{key}') carries {ntok} values; FVCOM's "
                f"GET_VALUE holds at most {_MAX_VALUES} (fewer for strings) and "
                "stops with a FATAL_ERROR"
            )
        norm = " ".join(key.split()).upper()
        if norm in known_norm and key != known_norm[norm]:
            raise ValueError(
                f"{path}: key '{key}' is not spelled exactly '{known_norm[norm]}'; "
                "FVCOM matches keys case- and space-exactly and would ignore it"
            )
        if key in known_norm and lineno > _MAX_HEADER_LINE:
            raise ValueError(
                f"{path}: key '{key}' is on line {lineno}; FVCOM's SCAN_FILE stops "
                f"searching after line {_MAX_HEADER_LINE} and would not find it -- "
                "move the keys to the top of the file"
            )
        if key in pairs:
            raise ValueError(
                f"{path}: key '{key}' appears more than once "
                "(FVCOM uses only the first occurrence)"
            )
        pairs[key] = val
    # <--- jsasaki 2026-11-05

    if "NUMBER OF SIGMA LEVELS" not in pairs:
        raise ValueError(f"{path}: missing 'NUMBER OF SIGMA LEVELS'")
    if "SIGMA COORDINATE TYPE" not in pairs:
        raise ValueError(f"{path}: missing 'SIGMA COORDINATE TYPE'")
    kb = _get_int(path, pairs, "NUMBER OF SIGMA LEVELS")
    assert kb is not None
    # jsasaki 2026-11-05: review round 8 #7: mod_input.F CHECK_LAYER_COUNT
    if kb < 3:
        raise ValueError(
            f"{path}: NUMBER OF SIGMA LEVELS = {kb}; FVCOM needs at least 3 levels "
            "(two layers) for a 3-D run"
        )
    # ---> jsasaki 2026-11-05: review round 8 #4: FVCOM selects the type
    # case-sensitively and stops on any other spelling; do not upper-case it
    stype = pairs["SIGMA COORDINATE TYPE"]
    if stype not in _STYPES:
        raise ValueError(
            f"{path}: 'SIGMA COORDINATE TYPE = {stype}' is not one of FVCOM's "
            f"types {', '.join(_STYPES)} (case-sensitive); FVCOM would stop"
        )
    # <--- jsasaki 2026-11-05

    gtsz: GtszSpec | None = None
    if stype == "SIGMAZ":
        # same order and conditions as mod_input.F READ_COLDSTART_SIGMA
        base = _get_int(path, pairs, "GTSZ BASE")
        nz = _get_int(path, pairs, "GTSZ NZ")
        if nz is None:
            raise ValueError(f"{path}: SIGMAZ file lacks 'GTSZ NZ' (FVCOM stops)")
        k1 = _get_int(path, pairs, "GTSZ K1")
        k2 = _get_int(path, pairs, "GTSZ K2")
        if nz > 0 and (k1 is None or k2 is None):
            raise ValueError(
                f"{path}: 'GTSZ K1' and 'GTSZ K2' are required when GTSZ NZ > 0"
            )
        p1 = _get_float(path, pairs, "GTSZ P1")
        l1 = _get_float(path, pairs, "GTSZ L1")
        l2 = _get_float(path, pairs, "GTSZ L2")
        smooth = _get_float(path, pairs, "GTSZ SMOOTH")
        mask = _get_bool(path, pairs, "GTSZ MASK")
        sadapt = _get_bool(path, pairs, "GTSZ SADAPT")
        smax = _get_float(path, pairs, "GTSZ SMAX")
        lowsig = _get_bool(path, pairs, "GTSZ LOWSIG")
        dref = _get_float(path, pairs, "GTSZ DREF")
        k1v = 1 if k1 is None else k1
        lowfrac: tuple[float, ...] | None = None
        if lowsig and k1v < kb:
            lowfrac = _get_fvec(path, pairs, "GTSZ LOWFRAC")
        dye_nowall = _get_bool(path, pairs, "GTSZ DYE_NOWALL")
        zlev: tuple[float, ...] = ()
        if nz > 0:
            zl = _get_fvec(path, pairs, "GTSZ ZLEV")
            if zl is None:
                raise ValueError(f"{path}: missing 'GTSZ ZLEV' (GTSZ NZ > 0)")
            zlev = zl
        bpg_nz = _get_int(path, pairs, "BPG REF NZ") or 0
        # jsasaki 2026-11-05: review round 8 #6: mod_input.F range check
        if not 0 <= bpg_nz <= 500:
            raise ValueError(
                f"{path}: FVCOM requires 0 <= BPG REF NZ <= 500 (got {bpg_nz})"
            )
        bpg: tuple[float, ...] = ()
        if bpg_nz > 0:
            bz = _get_fvec(path, pairs, "BPG REF ZLEV")
            if bz is None or len(bz) != bpg_nz:
                raise ValueError(
                    f"{path}: 'BPG REF ZLEV' must carry exactly BPG REF NZ = "
                    f"{bpg_nz} values"
                )
            bpg = bz
        gtsz = GtszSpec(
            kb=kb,
            base=2 if base is None else base,
            k1=k1v,
            k2=kb if k2 is None else k2,
            nz=nz,
            zlev=zlev,
            p1=2.0 if p1 is None else p1,
            l1=1.0 if l1 is None else l1,
            l2=1.0 if l2 is None else l2,
            smooth=0.0 if smooth is None else smooth,
            mask=bool(mask),
            sadapt=bool(sadapt),
            lowsig=bool(lowsig),
            smax=0.0 if smax is None else smax,
            dref=0.0 if dref is None else dref,
            lowfrac=lowfrac or (),
            dye_nowall=bool(dye_nowall),
            bpg_ref_zlev=bpg,
        )
    return SigmaFile(kb=kb, stype=stype, gtsz=gtsz, raw=pairs)


# jsasaki 2026-11-05: review round 7 #2: floating values are always written with
# a decimal point -- FVCOM ignores a decimal-less floating value (see above)
def _ffmt(x: float) -> str:
    """Format a floating value so FVCOM's SCAN_FILE types it as 'float'."""
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"cannot write the non-finite value {x} to a sigma.dat")
    # jsasaki 2026-11-05: review round 9 #5: the shortest EXACT spelling (repr), so
    # that reading the file back gives the same float64 (no rounding to zero)
    s = repr(x)
    if "." not in s:
        if "e" in s:
            mant, _, exp = s.partition("e")
            s = f"{mant}.0e{exp}"
        else:
            s += ".0"
    return s


def _compact_token(tok: str) -> str:
    """The shortest spelling of a decimal token with the same value
    ('-2.0' -> '-2', '0.500000' -> '0.5', '1.5e-05' unchanged)."""
    mant, e, exp = tok.partition("e")
    if "." in mant:
        mant = mant.rstrip("0").rstrip(".")
        if mant in ("", "-", "+"):
            mant += "0"
    return mant + e + exp


def _list_line(key: str, toks: list[str]) -> str:
    """``KEY = v1 v2 ...`` for a floating list, within FVCOM's scanner limits.

    FVCOM's GET_VALUE types a whole list as floating when the value holds a
    decimal point ANYWHERE, and then reads every token list-directed into a
    real, so '-2' inside such a list is read as -2.0.  When the list with a
    decimal point on every token is longer than SCAN_FILE's 400-character line
    buffer, the tokens are written in their shortest exact spelling with one
    decimal point kept.  A list that still does not fit, or that has more than
    150 values, raises :class:`ValueError`: SCAN_FILE's backslash line
    continuation is compiler-dependent (ENDLINE differs between ifort and other
    compilers in mod_utils.F), so it is not emitted.
    """
    # ---> jsasaki 2026-11-05: review round 8 #5: never emit a line FVCOM
    # truncates (400 characters) or a list it cannot hold (150 values)
    if len(toks) > _MAX_VALUES:
        raise ValueError(
            f"{key}: {len(toks)} values; FVCOM's GET_VALUE reads at most "
            f"{_MAX_VALUES} values per key"
        )
    line = f"{key} = {' '.join(toks)}"
    if len(line) <= _MAX_LINE:
        return line
    short = [_compact_token(t) for t in toks]
    if not any("." in t for t in short):
        mant, e, exp = short[0].partition("e")
        short[0] = f"{mant}.0{e}{exp}"
    line = f"{key} = {' '.join(short)}"
    if len(line) > _MAX_LINE:
        # jsasaki 2026-11-05: review round 9 #5: values are exact now; say how to shorten them
        raise ValueError(
            f"{key}: the value list needs {len(line)} characters even in its "
            f"shortest exact form; FVCOM's SCAN_FILE reads only {_MAX_LINE} "
            "characters per line -- use fewer levels or shorter values "
            "(e.g. round the levels to a few decimals before building the spec)"
        )
    return line
    # <--- jsasaki 2026-11-05


def gtsz_to_lines(spec: GtszSpec, *, header_comment: str | None = None) -> list[str]:
    """Render a :class:`GtszSpec` as the lines of a SIGMAZ ``*_sigma.dat``."""
    spec.validate()
    lines: list[str] = []
    if header_comment:
        for cl in header_comment.splitlines():
            # ---> jsasaki 2026-11-05: review round 10 #1: FVCOM truncates a line to
            # its 400-character buffer, which can expose a continuation suffix
            if len(f"! {cl}".encode()) > _MAX_LINE:
                raise ValueError(
                    f"header comment line {cl[:40]!r}... is "
                    f"{len(f'! {cl}'.encode())} characters long as written; FVCOM's "
                    f"SCAN_FILE reads only the first {_MAX_LINE} (a truncated line "
                    "can end in its continuation suffix and swallow the next key) "
                    "-- split it into shorter lines"
                )
            # <--- jsasaki 2026-11-05
            # jsasaki 2026-11-05: review round 9 #4: a continued comment swallows the next line
            if _is_continued(f"! {cl}"):
                raise ValueError(
                    f"header comment line {cl!r} ends in a backslash pair, FVCOM's "
                    "line-continuation suffix: FVCOM would join the next line (a "
                    "key) onto the comment and lose it -- remove the trailing "
                    "backslashes"
                )
            lines.append(f"! {cl}")
    lines.append(f"NUMBER OF SIGMA LEVELS = {spec.kb}")
    lines.append("SIGMA COORDINATE TYPE = SIGMAZ")
    lines.append(f"GTSZ BASE = {spec.base}")
    lines.append(f"GTSZ K1 = {spec.k1}")
    lines.append(f"GTSZ K2 = {spec.k2}")
    # jsasaki 2026-11-05: review round 7 #2: always a decimal point
    lines.append(f"GTSZ P1 = {_ffmt(spec.p1)}")
    # jsasaki 2026-11-05: review round 7 #2: always a decimal point
    lines.append(f"GTSZ L1 = {_ffmt(spec.l1)}")
    # jsasaki 2026-11-05: review round 7 #2: always a decimal point
    lines.append(f"GTSZ L2 = {_ffmt(spec.l2)}")
    lines.append(f"GTSZ NZ = {spec.nz}")
    # ---> jsasaki 2026-11-05: review round 9 #6: FVCOM's SCAN_FILE (and
    # read_sigma_dat) cannot read a one-value list for a mandatory vector
    if spec.nz > 0 and len(spec.zlev) < 2:
        raise ValueError(
            f"GTSZ ZLEV has {len(spec.zlev)} value(s); FVCOM's SCAN_FILE does not "
            "read a one-value list (it returns -4 and FVCOM stops), so a file with "
            "GTSZ NZ = 1 cannot be written -- use at least two z-levels"
        )
    if len(spec.bpg_ref_zlev) == 1:
        raise ValueError(
            "BPG REF ZLEV has one value; FVCOM's SCAN_FILE does not read a "
            "one-value list (it returns -4 and FVCOM stops) -- use at least two "
            "reference levels or none"
        )
    # <--- jsasaki 2026-11-05
    if spec.nz > 0:
        # jsasaki 2026-11-05: review round 8 #5: within the scanner limits
        lines.append(_list_line("GTSZ ZLEV", [_ffmt(z) for z in spec.zlev]))
    if spec.smooth != 0.0:
        # jsasaki 2026-11-05: review round 7 #2: always a decimal point
        lines.append(f"GTSZ SMOOTH = {_ffmt(spec.smooth)}")
    lines.append(f"GTSZ MASK = {'T' if spec.mask else 'F'}")
    lines.append(f"GTSZ SADAPT = {'T' if spec.sadapt else 'F'}")
    # jsasaki 2026-11-05: review round 9 #5: exact, decimal-bearing (was .6f: 4e-7 -> 0)
    lines.append(f"GTSZ SMAX = {_ffmt(spec.smax)}")
    if spec.lowsig:
        lines.append("GTSZ LOWSIG = T")
    if spec.dref > 0.0:
        # a decimal point always, NOT :g -- FVCOM's SCAN_FILE type-detects "30" as integer and
        # returns -3 against an FSCAL request, silently zeroing the cap
        # jsasaki 2026-11-05: review round 9 #5: exact, decimal-bearing (_ffmt)
        lines.append(f"GTSZ DREF = {_ffmt(spec.dref)}")
    # jsasaki 2026-11-05: review round 7 #2: FVCOM ignores a one-value list
    # (a single lower-band layer has fraction 1 either way)
    if len(spec.lowfrac) >= 2:
        # jsasaki 2026-11-05: review round 8 #5: within the scanner limits
        # jsasaki 2026-11-05: review round 9 #5: exact, decimal-bearing (1e-7 stayed 0.000000)
        lines.append(_list_line("GTSZ LOWFRAC", [_ffmt(f) for f in spec.lowfrac]))
    if spec.dye_nowall:
        lines.append("GTSZ DYE_NOWALL = T")
    if spec.bpg_ref_zlev:
        lines.append(f"BPG REF NZ = {len(spec.bpg_ref_zlev)}")
        # jsasaki 2026-11-05: review round 8 #5: within the scanner limits
        lines.append(_list_line("BPG REF ZLEV", [_ffmt(z) for z in spec.bpg_ref_zlev]))
    # ---> jsasaki 2026-11-05: review round 8 #5: every key must sit within
    # SCAN_FILE's 201-line header search (a long header comment pushes it out)
    for n, ln in enumerate(lines, start=1):
        if not ln.startswith("!") and n > _MAX_HEADER_LINE:
            raise ValueError(
                f"'{ln.partition('=')[0].strip()}' would be on line {n}; FVCOM's "
                f"SCAN_FILE searches only the first {_MAX_HEADER_LINE} lines -- "
                "shorten the header comment"
            )
    # <--- jsasaki 2026-11-05
    return lines


def write_sigma_dat(
    path: str | Path, spec: GtszSpec, *, header_comment: str | None = None
) -> None:
    """Write a SIGMAZ ``*_sigma.dat`` for ``spec``."""
    lines = gtsz_to_lines(spec, header_comment=header_comment)
    Path(path).write_text("\n".join(lines) + "\n")
