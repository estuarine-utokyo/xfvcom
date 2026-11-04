"""Fortran list-directed real-literal conversion shared by the FVCOM input readers.

Moved here from :mod:`xfvcom.io.sigma_dat` (review round 14 #2) so that the
mesh reader (:mod:`xfvcom.grid.dat_reader`), the depth readers and the sigma
reader share ONE conversion without importing across the ``grid`` / ``io``
packages (``xfvcom.io`` imports ``xfvcom.grid``).  The module depends only on
``re`` and ``numpy``; :mod:`xfvcom.io.sigma_dat` re-exports ``fortran_real``.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = ["fortran_real", "list_fields", "list_items", "MAX_REPEAT"]

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


#: A list-directed value separator: a comma with optional blanks around it, or blanks alone.
_LIST_SEPARATOR = re.compile(r"\s*,\s*|\s+")
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


def list_items(body: str) -> list[tuple[int, str]]:
    """The list-directed items of a record body (comment and any '/' already removed) as
    ``(repeat count, value)`` pairs, NOT expanded: ``3 2*7`` -> ``[(1, '3'), (2, '7')]``.

    A null value is ``(1, '')``; ``r*`` is ``(r, '')``.  A malformed repetition ('0*5', '*5')
    is kept as one literal value, which no conversion accepts.  Counts are not checked here:
    a reader checks a count against :data:`MAX_REPEAT` only when it consumes the item (Fortran
    never looks at the items after the ones its READ list needs).
    """
    s = body.strip()
    if not s:
        return []
    fields = _LIST_SEPARATOR.split(s)
    if fields and fields[-1] == "" and s.endswith(","):
        fields.pop()
    out: list[tuple[int, str]] = []
    for f in fields:
        m = _REPEAT.fullmatch(f)
        rs = m["r"].lstrip("0") if m is not None else ""
        if m is not None and rs:
            out.append((int(rs), m["val"]))
        else:
            out.append((1, f))
    return out


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
    for r, val in list_items(body):
        if r > MAX_REPEAT:
            raise ValueError(
                f"repetition count {r} exceeds {MAX_REPEAT}, the largest the Fortran runtime "
                "(gfortran) accepts"
            )
        if len(out) + r > MAX_LIST_FIELDS:
            raise ValueError(
                f"repetition count {r} is not supported (a record may expand to at most "
                f"{MAX_LIST_FIELDS} fields here)"
            )
        out.extend([val] * r)
    return out
