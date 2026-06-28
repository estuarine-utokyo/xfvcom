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

from ..grid.gtsz import GtszSpec

__all__ = ["SigmaFile", "read_sigma_dat", "write_sigma_dat", "gtsz_to_lines"]


@dataclass
class SigmaFile:
    """A parsed ``*_sigma.dat`` header."""

    kb: int  #: NUMBER OF SIGMA LEVELS
    stype: str  #: SIGMA COORDINATE TYPE (upper-case)
    gtsz: GtszSpec | None = None  #: the GTSZ spec when ``stype == 'SIGMAZ'``
    raw: dict[str, str] | None = None  #: all parsed KEY -> raw-value strings


def _scan(pairs: dict[str, str], key: str) -> str | None:
    """FVCOM ``SCAN_FILE`` semantics: case-insensitive key match."""
    return pairs.get(key.upper())


def _as_int(s: str) -> int:
    return int(float(s.split()[0]))


def _as_float(s: str) -> float:
    return float(s.split()[0])


def _as_bool(s: str) -> bool:
    t = s.strip().upper()
    return t.startswith("T")


def _as_floats(s: str) -> tuple[float, ...]:
    return tuple(float(tok) for tok in s.replace(",", " ").split())


def read_sigma_dat(path: str | Path) -> SigmaFile:
    """Parse a ``*_sigma.dat`` file into a :class:`SigmaFile`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

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
            key = " ".join(key.split()).upper()  # normalize internal whitespace
            val = val.strip()
            if key:
                pairs[key] = val

    if "NUMBER OF SIGMA LEVELS" not in pairs:
        raise ValueError(f"{path}: missing 'NUMBER OF SIGMA LEVELS'")
    if "SIGMA COORDINATE TYPE" not in pairs:
        raise ValueError(f"{path}: missing 'SIGMA COORDINATE TYPE'")
    kb = _as_int(pairs["NUMBER OF SIGMA LEVELS"])
    stype = pairs["SIGMA COORDINATE TYPE"].strip().upper()

    gtsz: GtszSpec | None = None
    if stype == "SIGMAZ":
        nz = _as_int(_scan(pairs, "GTSZ NZ") or "0")
        zlev_s = _scan(pairs, "GTSZ ZLEV")
        zlev = _as_floats(zlev_s) if zlev_s else ()
        bpg_s = _scan(pairs, "BPG REF ZLEV")
        bpg = _as_floats(bpg_s) if bpg_s else ()
        base_s = _scan(pairs, "GTSZ BASE")
        k1_s = _scan(pairs, "GTSZ K1")
        k2_s = _scan(pairs, "GTSZ K2")
        gtsz = GtszSpec(
            kb=kb,
            base=_as_int(base_s) if base_s else 2,
            k1=_as_int(k1_s) if k1_s else (1 if nz == 0 else 1),
            k2=_as_int(k2_s) if k2_s else kb,
            nz=nz,
            zlev=zlev,
            p1=_as_float(_scan(pairs, "GTSZ P1") or "2.0"),
            l1=_as_float(_scan(pairs, "GTSZ L1") or "1.0"),
            l2=_as_float(_scan(pairs, "GTSZ L2") or "1.0"),
            smooth=_as_float(_scan(pairs, "GTSZ SMOOTH") or "0.0"),
            mask=_as_bool(_scan(pairs, "GTSZ MASK") or "F"),
            sadapt=_as_bool(_scan(pairs, "GTSZ SADAPT") or "F"),
            smax=_as_float(_scan(pairs, "GTSZ SMAX") or "0.0"),
            dye_nowall=_as_bool(_scan(pairs, "GTSZ DYE_NOWALL") or "F"),
            bpg_ref_zlev=bpg,
        )
    return SigmaFile(kb=kb, stype=stype, gtsz=gtsz, raw=pairs)


def _fmt_zlev(zlev: tuple[float, ...]) -> str:
    return " ".join(f"{z:g}" for z in zlev)


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
    lines.append(f"GTSZ P1 = {spec.p1:g}")
    lines.append(f"GTSZ L1 = {spec.l1:g}")
    lines.append(f"GTSZ L2 = {spec.l2:g}")
    lines.append(f"GTSZ NZ = {spec.nz}")
    if spec.nz > 0:
        lines.append(f"GTSZ ZLEV = {_fmt_zlev(spec.zlev)}")
    if spec.smooth != 0.0:
        lines.append(f"GTSZ SMOOTH = {spec.smooth:g}")
    lines.append(f"GTSZ MASK = {'T' if spec.mask else 'F'}")
    lines.append(f"GTSZ SADAPT = {'T' if spec.sadapt else 'F'}")
    lines.append(f"GTSZ SMAX = {spec.smax:.6f}")
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
