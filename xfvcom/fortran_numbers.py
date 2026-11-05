"""Fortran list-directed real-literal conversion shared by the FVCOM input readers.

Moved here from :mod:`xfvcom.io.sigma_dat` (review round 14 #2) so that the
mesh reader (:mod:`xfvcom.grid.dat_reader`), the depth readers and the sigma
reader share ONE conversion without importing across the ``grid`` / ``io``
packages (``xfvcom.io`` imports ``xfvcom.grid``).  The module depends only on
``re`` and ``numpy``; :mod:`xfvcom.io.sigma_dat` re-exports ``fortran_real``.
"""

from __future__ import annotations

import re

# jsasaki 2026-11-05: review round 20 #5: lazy list items
from collections.abc import Iterator

import numpy as np

# jsasaki 2026-11-05: review round 20 #2 #5: repeat_count / line_state
__all__ = [
    "fortran_real",
    "list_fields",
    "list_items",
    "line_state",
    "repeat_count",
    "MAX_REPEAT",
    # jsasaki 2026-11-05: review round 21 #1: the portable list-directed blanks
    "LIST_BLANKS",
]

#: A real literal as Fortran's list-directed READ accepts it (checked against
#: gfortran and Intel ifx): optional sign, a mantissa with at least one digit
#: and at most one decimal point, and an optional exponent written either with
#: a letter (E/D/Q, any case, optional sign) or as a bare SIGNED integer
#: ('1.0-3' = 0.001, '1.0+3' = 1000).  'E'/'D'/'Q' without digits ('1e') is
#: refused: gfortran stops on it (ifx reads 1.0), so it is not portable.
_FORTRAN_REAL = re.compile(
    r"(?P<mant>[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+))"
    r"(?:(?P<letter>[EeDdQq])(?P<lexp>[+-]?[0-9]+)|(?P<sexp>[+-][0-9]+))?"
)


def fortran_real(tok: str) -> float:
    """Convert one token the way Fortran's list-directed ``READ`` into a
    ``REAL(DP)`` does, or raise :class:`ValueError`.

    Accepts implicit-exponent forms (``1.0-3``, ``1.0+3``, ``.5-1``), ``D``/``Q``
    exponents (``1.0d-3``) and signs; refuses what the Fortran runtime refuses
    (``1.0.0``, ``abc``, ``1e``, ``1.0e+``, ``.``) and a value that overflows
    ``REAL(DP)`` (Intel Fortran stops on it, gfortran reads Infinity).  An
    underflow reads as zero, as in Fortran.
    """
    m = _FORTRAN_REAL.fullmatch(tok)
    if m is None:
        raise ValueError(f"'{tok}' is not a Fortran real literal")
    exp = m["lexp"] if m["letter"] else m["sexp"]
    x = float(m["mant"] + ("" if exp is None else "e" + exp))
    if not np.isfinite(x):
        raise ValueError(f"'{tok}' is not finite (overflows REAL(DP))")
    return x


# ---> jsasaki 2026-11-05: review round 21 #1: only the blanks list-directed input accepts (space
# and tab: gfortran 8.5 and ifx 2025.1) separate values, never Python's Unicode-aware whitespace
#: The characters that separate list-directed values like a blank on every compiler checked
#: (gfortran 8.5, ifx 2025.1): space and tab.  Anything else -- U+00A0 (bytes C2 A0), a form feed,
#: a vertical tab, a carriage return inside a line (gfortran: blank, ifx: error), ';' (gfortran:
#: separator, ifx: error) -- stays inside the token, so its conversion fails as Fortran's does.
LIST_BLANKS = " \t"
#: A list-directed value separator: a comma with optional blanks around it, or blanks alone.
_LIST_SEPARATOR = re.compile(r"[ \t]*,[ \t]*|[ \t]+")
# <--- jsasaki 2026-11-05
# ---> jsasaki 2026-11-05: review round 18 #7: list-directed repetition 'r*value' / 'r*'
#: A repeated value: a positive repeat count, '*', and the value (empty = r null values).
_REPEAT = re.compile(r"(?P<r>[0-9]+)\*(?P<val>.*)")
#: Upper bound on the fields one record expands to (callers use only the first few).
MAX_LIST_FIELDS = 4096
# <--- jsasaki 2026-11-05
# jsasaki 2026-11-05: review round 19 #2: the largest repeat count the gfortran runtime accepts
# (libgfortran list_read.c MAX_REPEAT; 'Repeat count overflow' above it, checked 2026-11-05 with
# gfortran 8.5: 200000000*1 reads, 200000001*1 fails).  ifx accepts larger counts, so this is the
# portable limit.
MAX_REPEAT = 200_000_000


# ---> jsasaki 2026-11-05: review round 20 #2 #5: lazy items (repeat counts converted only when a READ
# consumes them, compared textually against MAX_REPEAT) and the separator state carried across lines
def repeat_count(rtext: str) -> int:
    """The repeat count ``rtext`` (decimal digits, leading zeros allowed) as an ``int``, or
    :class:`ValueError` when it exceeds :data:`MAX_REPEAT`.  The digits are compared TEXTUALLY
    first, so a count of any length is refused without Python's integer conversion (whose
    4300-digit limit would otherwise raise its own error)."""
    digits = rtext.lstrip("0") or "0"
    limit = str(MAX_REPEAT)
    if len(digits) > len(limit) or (len(digits) == len(limit) and digits > limit):
        shown = (
            digits if len(digits) <= 20 else f"{digits[:8]}...({len(digits)} digits)"
        )
        raise ValueError(
            f"repetition count {shown} exceeds {MAX_REPEAT}, the largest the Fortran runtime "
            "(gfortran) accepts"
        )
    return int(digits)


def list_items(body: str, after_value: bool = False) -> Iterator[tuple[str, str]]:
    """The list-directed items of one physical line of a READ, LAZILY, as ``(repeat count text,
    value)`` pairs, NOT expanded: ``3 2*7`` -> ``('1', '3'), ('2', '7')``.

    ``body`` is the line's text before any ``/`` (a ``!`` is NOT removed: list-directed input has
    no comments, so ``20.0!note`` is one value that no conversion accepts).  A null value is
    ``('1', '')``; ``r*`` is ``(r, '')``.  A malformed repetition ('0*5', '*5') is kept as one
    literal value, which no conversion accepts.  The count is returned as text: a reader converts
    it with :func:`repeat_count` only when its READ consumes the item (Fortran never looks at the
    items after the ones its list needs).

    ``after_value`` is the separator state left by the READ's previous line: True when that line
    ended with a value (the end of a record acts as a blank), so a comma that begins this line
    completes that separator instead of starting a null value (``1 3150`` + ``,1`` reads 1, 3150,
    1; ``1,3150,`` + ``,1`` leaves the third item null -- gfortran 8.5 and ifx 2025.1).
    """
    # jsasaki 2026-11-05: review round 21 #1: trim list-directed blanks only (not Unicode whitespace)
    s = body.strip(LIST_BLANKS)
    if not s:
        return
    fields = _LIST_SEPARATOR.split(s)
    if fields and fields[-1] == "" and s.endswith(","):
        fields.pop()
    if after_value and s.startswith(",") and fields and fields[0] == "":
        fields.pop(0)
    for f in fields:
        m = _REPEAT.fullmatch(f)
        if m is not None and m["r"].lstrip("0"):
            yield m["r"], m["val"]
        else:
            yield "1", f


def line_state(body: str, after_value: bool) -> bool:
    """The separator state (see :func:`list_items`) after a READ has passed line ``body``: a
    blank line leaves it unchanged, a line ending in a comma clears it, any other sets it.
    """
    # jsasaki 2026-11-05: review round 21 #1: trim list-directed blanks only (not Unicode whitespace)
    s = body.strip(LIST_BLANKS)
    if not s:
        return after_value
    return not s.endswith(",")


# <--- jsasaki 2026-11-05


def list_fields(body: str) -> list[str]:
    """Split a record body (comment already removed) into list-directed fields.

    Commas separate values like blanks, as in Fortran's list-directed ``READ``
    (``400000.0, 3900000.0, 10.0``).  A NULL value -- nothing between two
    commas, or a leading comma -- is returned as an empty string, which no
    numeric conversion accepts, so a record relying on Fortran's
    leave-unchanged semantics for null values is refused by the caller.  A
    trailing comma ends the record and adds no field.  A repeated value
    ``r*value`` becomes ``r`` fields (``1*3150`` -> ``3150``, ``2*`` -> two null
    values); a count above :data:`MAX_REPEAT` or an expansion beyond ``MAX_LIST_FIELDS``
    raises :class:`ValueError` (review round 19 #2).
    """
    # jsasaki 2026-11-05: review round 19 #2: a count the Fortran runtime refuses, or one that would
    # expand beyond MAX_LIST_FIELDS, is an explicit error (no silent substitution of a smaller count)
    out: list[str] = []
    for rtext, val in list_items(body):
        # jsasaki 2026-11-05: review round 20 #5: the count compared textually before conversion
        r = repeat_count(rtext)
        if len(out) + r > MAX_LIST_FIELDS:
            raise ValueError(
                f"repetition count {r} is not supported (a record may expand to at most "
                f"{MAX_LIST_FIELDS} fields here)"
            )
        out.extend([val] * r)
    return out
