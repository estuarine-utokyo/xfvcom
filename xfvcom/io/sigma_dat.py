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

from ..grid.gtsz import GtszSpec

__all__ = ["SigmaFile", "read_sigma_dat", "write_sigma_dat", "gtsz_to_lines"]


@dataclass
class SigmaFile:
    """A parsed ``*_sigma.dat`` header."""

    kb: int  #: NUMBER OF SIGMA LEVELS
    stype: str  #: SIGMA COORDINATE TYPE (upper-case)
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


def _fvcom_type(val: str) -> str:
    """The type FVCOM's ``GET_VALUE`` assigns to a (comment-stripped) value."""
    val = val.replace(",", " ").strip()
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
    if _fvcom_type(val) != "integer" or len(val.replace(",", " ").split()) != 1:
        raise _bad(
            path,
            key,
            val,
            "FVCOM reads this key as ONE integer without a decimal point "
            "(e.g. '31', not '31.0'); any other form is not read as written",
        )
    try:
        return int(val.split()[0])
    except ValueError:
        raise _bad(path, key, val, "not a valid integer") from None


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
    toks = val.replace(",", " ").split()
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
        return float(toks[0])
    except ValueError:
        raise _bad(path, key, val, "not a valid floating value") from None


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
    return val.strip() == "T"


def _get_fvec(path: Path, pairs: dict[str, str], key: str) -> tuple[float, ...] | None:
    val = _scan(pairs, key)
    if val is None:
        return None
    toks = val.replace(",", " ").split()
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
        return tuple(float(t) for t in toks)
    except ValueError:
        raise _bad(path, key, val, "not a list of valid floating values") from None


def read_sigma_dat(path: str | Path) -> SigmaFile:
    """Parse a ``*_sigma.dat`` file into a :class:`SigmaFile`.

    Values are typed as FVCOM's ``SCAN_FILE`` types them; a SIGMAZ value
    that FVCOM would not read as written (a decimal-less floating value, a
    decimal integer, a logical other than ``T``/``F``, a one-value list, a key
    differing from FVCOM's spelling only in case/spacing, a repeated key) raises
    :class:`ValueError` naming the key and the fix.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    known_norm = {k: k for k in _KNOWN_KEYS}
    pairs: dict[str, str] = {}
    with path.open() as fp:
        for line in fp:
            # strip comments (everything from the first '!')
            bang = line.find("!")
            if bang >= 0:
                line = line[:bang]
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            # FVCOM matches the key text exactly (leading/trailing blanks aside)
            key = key.strip()
            val = val.strip()
            if not key or not val:
                continue  # GET_VALUE: "no data"
            norm = " ".join(key.split()).upper()
            if norm in known_norm and key != known_norm[norm]:
                raise ValueError(
                    f"{path}: key '{key}' is not spelled exactly '{known_norm[norm]}'; "
                    "FVCOM matches keys case- and space-exactly and would ignore it"
                )
            if key in pairs:
                raise ValueError(
                    f"{path}: key '{key}' appears more than once "
                    "(FVCOM uses only the first occurrence)"
                )
            pairs[key] = val

    if "NUMBER OF SIGMA LEVELS" not in pairs:
        raise ValueError(f"{path}: missing 'NUMBER OF SIGMA LEVELS'")
    if "SIGMA COORDINATE TYPE" not in pairs:
        raise ValueError(f"{path}: missing 'SIGMA COORDINATE TYPE'")
    kb = _get_int(path, pairs, "NUMBER OF SIGMA LEVELS")
    assert kb is not None
    stype = pairs["SIGMA COORDINATE TYPE"].strip().upper()

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
    s = f"{x:.10g}"
    if "." not in s:
        if "e" in s:
            mant, _, exp = s.partition("e")
            s = f"{mant}.0e{exp}"
        else:
            s += ".0"
    return s


def _fmt_zlev(zlev: tuple[float, ...]) -> str:
    # jsasaki 2026-11-05: review round 7 #2: decimal literals ('-2.0', not '-2')
    return " ".join(_ffmt(z) for z in zlev)


def gtsz_to_lines(spec: GtszSpec, *, header_comment: str | None = None) -> list[str]:
    """Render a :class:`GtszSpec` as the lines of a SIGMAZ ``*_sigma.dat``."""
    spec.validate()
    lines: list[str] = []
    if header_comment:
        for cl in header_comment.splitlines():
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
    if spec.nz > 0:
        lines.append(f"GTSZ ZLEV = {_fmt_zlev(spec.zlev)}")
    if spec.smooth != 0.0:
        # jsasaki 2026-11-05: review round 7 #2: always a decimal point
        lines.append(f"GTSZ SMOOTH = {_ffmt(spec.smooth)}")
    lines.append(f"GTSZ MASK = {'T' if spec.mask else 'F'}")
    lines.append(f"GTSZ SADAPT = {'T' if spec.sadapt else 'F'}")
    lines.append(f"GTSZ SMAX = {spec.smax:.6f}")
    if spec.lowsig:
        lines.append("GTSZ LOWSIG = T")
    if spec.dref > 0.0:
        # .6f, NOT :g -- FVCOM's SCAN_FILE type-detects "30" as integer and
        # returns -3 against an FSCAL request, silently zeroing the cap
        lines.append(f"GTSZ DREF = {spec.dref:.6f}")
    # jsasaki 2026-11-05: review round 7 #2: FVCOM ignores a one-value list
    # (a single lower-band layer has fraction 1 either way)
    if len(spec.lowfrac) >= 2:
        lines.append(f"GTSZ LOWFRAC = {' '.join(f'{f:.6f}' for f in spec.lowfrac)}")
    if spec.dye_nowall:
        lines.append("GTSZ DYE_NOWALL = T")
    if spec.bpg_ref_zlev:
        lines.append(f"BPG REF NZ = {len(spec.bpg_ref_zlev)}")
        lines.append(f"BPG REF ZLEV = {_fmt_zlev(spec.bpg_ref_zlev)}")
    return lines


def write_sigma_dat(
    path: str | Path, spec: GtszSpec, *, header_comment: str | None = None
) -> None:
    """Write a SIGMAZ ``*_sigma.dat`` for ``spec``."""
    lines = gtsz_to_lines(spec, header_comment=header_comment)
    Path(path).write_text("\n".join(lines) + "\n")
