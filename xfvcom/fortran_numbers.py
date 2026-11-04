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

__all__ = ["fortran_real", "list_fields"]

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


def list_fields(body: str) -> list[str]:
    """Split a record body (comment already removed) into list-directed fields.

    Commas separate values like blanks, as in Fortran's list-directed ``READ``
    (``400000.0, 3900000.0, 10.0``).  A NULL value -- nothing between two
    commas, or a leading comma -- is returned as an empty string, which no
    numeric conversion accepts, so a record relying on Fortran's
    leave-unchanged semantics for null values is refused by the caller.  A
    trailing comma ends the record and adds no field.
    """
    s = body.strip()
    if not s:
        return []
    fields = _LIST_SEPARATOR.split(s)
    if fields and fields[-1] == "" and s.endswith(","):
        fields.pop()
    return fields
