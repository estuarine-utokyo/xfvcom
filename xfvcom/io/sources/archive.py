"""Logical-name resolver for the ``$DATA_DIR`` river / wastewater archive.

Part of the 2026-05-24 archive reorganization
(``river_dl/docs/campaigns/20260524_data_archive_reorg.md``). Consumers
should request *logical names* — ``(domain, entity, station,
variable_group, freq)`` — and let this adapter resolve them to a physical
path via a per-domain ``_catalog.csv``. That decouples consumers from the
archive's on-disk layout, so a future restructure (e.g. variable-first ->
entity-first) touches only the catalog, not the ~30 hard-coded paths
scattered across repos.

The actual NetCDF reading stays in
:class:`xfvcom.io.sources.river_dl.RiverDLNetCDFSource` (which takes a
path); this module only resolves the path. Catalog generation is a pure
filesystem scan (no NetCDF opens), so it is cheap and login-node safe.

Layout understood by the scanner (transitional, P1 era):

    river/discharge/<River>/<Station>/<group>_<freq>.nc
    river/concentration/<River>.nc            (per-river climatology)
    wastewater/<Plant>/<group>_<freq>.nc

``<group>`` ∈ {discharge, concentration, concentration_modern};
``<freq>`` ∈ {hourly, daily}. After the P2/P3 entity-first restructure
the on-disk paths change but the catalog keys (and this API) do not.

CLI::

    python -m xfvcom.io.sources.archive build            # both domains under $DATA_DIR
    python -m xfvcom.io.sources.archive build --root /path/to/river
    python -m xfvcom.io.sources.archive resolve river Arakawa Iwabuchi discharge hourly
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Final

CATALOG_NAME: Final[str] = "_catalog.csv"
CATALOG_FIELDS: Final[tuple[str, ...]] = (
    "domain", "entity", "station", "variable_group", "freq", "relpath",
)
_FREQS: Final[tuple[str, ...]] = ("hourly", "daily")
_SKIP_DIRS: Final[frozenset[str]] = frozenset({"figs", "raw", "validation"})


def data_dir() -> Path:
    dd = os.environ.get("DATA_DIR", "")
    if not dd or not Path(dd).is_dir():
        raise RuntimeError("$DATA_DIR is not set or does not exist")
    return Path(dd)


def archive_root(domain: str) -> Path:
    """Absolute root of a domain archive, e.g. ``$DATA_DIR/river``."""
    return data_dir() / domain


# --------------------------------------------------------------------------
# Filename -> (variable_group, freq)
# --------------------------------------------------------------------------
def _parse_variant(filename: str) -> tuple[str, str]:
    stem = filename[:-3] if filename.endswith(".nc") else filename
    for fq in _FREQS:
        if stem.endswith("_" + fq):
            return stem[: -(len(fq) + 1)], fq
    return stem, ""  # no freq suffix (e.g. a climatology file)


# --------------------------------------------------------------------------
# Scanners (pure filesystem; no NetCDF opens)
# --------------------------------------------------------------------------
def _scan_river(root: Path) -> list[dict]:
    rows: list[dict] = []
    disc = root / "discharge"
    if disc.is_dir():
        for river_dir in sorted(p for p in disc.iterdir() if p.is_dir()):
            if river_dir.name in _SKIP_DIRS:
                continue
            for station_dir in sorted(p for p in river_dir.iterdir() if p.is_dir()):
                for nc in sorted(station_dir.glob("*.nc")):
                    vg, fq = _parse_variant(nc.name)
                    rows.append({
                        "domain": "river", "entity": river_dir.name,
                        "station": station_dir.name, "variable_group": vg,
                        "freq": fq, "relpath": str(nc.relative_to(root)),
                    })
    # Per-river concentration climatology (station-less); skip raw/ & figs/.
    conc = root / "concentration"
    if conc.is_dir():
        for nc in sorted(conc.glob("*.nc")):
            rows.append({
                "domain": "river", "entity": nc.stem, "station": "",
                "variable_group": "concentration_climatology", "freq": "",
                "relpath": str(nc.relative_to(root)),
            })
    return rows


def _scan_wastewater(root: Path) -> list[dict]:
    rows: list[dict] = []
    for plant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if plant_dir.name in _SKIP_DIRS:
            continue
        for nc in sorted(plant_dir.glob("*.nc")):
            vg, fq = _parse_variant(nc.name)
            rows.append({
                "domain": "wastewater", "entity": plant_dir.name, "station": "",
                "variable_group": vg, "freq": fq,
                "relpath": str(nc.relative_to(root)),
            })
    return rows


def scan_domain(root: Path, domain: str | None = None) -> list[dict]:
    """Scan a domain archive root and return catalog rows."""
    root = Path(root)
    domain = domain or root.name
    if domain == "river":
        return _scan_river(root)
    if domain == "wastewater":
        return _scan_wastewater(root)
    raise ValueError(f"unknown domain {domain!r} (expected river|wastewater)")


def write_catalog(root: Path, domain: str | None = None) -> Path:
    """Scan *root* and write ``<root>/_catalog.csv``. Returns the path."""
    root = Path(root)
    rows = scan_domain(root, domain)
    out = root / CATALOG_NAME
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CATALOG_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return out


# --------------------------------------------------------------------------
# Resolver
# --------------------------------------------------------------------------
class ArchiveCatalog:
    """Resolve logical names to physical paths via a domain ``_catalog.csv``.

    Parameters
    ----------
    domain : str
        ``"river"`` or ``"wastewater"``.
    root : Path, optional
        Domain root; defaults to ``$DATA_DIR/<domain>``.
    """

    def __init__(self, domain: str, root: Path | None = None) -> None:
        self.domain = domain
        self.root = Path(root) if root is not None else archive_root(domain)
        cat = self.root / CATALOG_NAME
        if not cat.is_file():
            raise FileNotFoundError(
                f"catalog not found: {cat} (run "
                f"`python -m xfvcom.io.sources.archive build`)"
            )
        with cat.open(encoding="utf-8") as f:
            self._rows = list(csv.DictReader(f))

    def resolve(
        self,
        entity: str,
        station: str | None = None,
        variable_group: str = "discharge",
        freq: str = "hourly",
    ) -> Path:
        """Return the absolute path for a logical name.

        Raises ``KeyError`` if no row matches, or if the match is
        ambiguous (multiple rows) — both indicate a catalog/query error.
        """
        st = "" if station is None else station
        hits = [
            r for r in self._rows
            if r["entity"] == entity and r["station"] == st
            and r["variable_group"] == variable_group and r["freq"] == freq
        ]
        if not hits:
            raise KeyError(
                f"no archive entry: domain={self.domain} entity={entity!r} "
                f"station={st!r} group={variable_group!r} freq={freq!r}"
            )
        if len(hits) > 1:
            raise KeyError(
                f"ambiguous archive entry ({len(hits)} matches) for "
                f"{entity!r}/{st!r}/{variable_group!r}/{freq!r}"
            )
        return self.root / hits[0]["relpath"]

    def entities(self) -> list[str]:
        return sorted({r["entity"] for r in self._rows})


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _cmd_build(args: argparse.Namespace) -> int:
    roots = ([Path(args.root)] if args.root
             else [archive_root("river"), archive_root("wastewater")])
    for root in roots:
        if not root.is_dir():
            print(f"[skip] {root} not a directory", file=sys.stderr)
            continue
        out = write_catalog(root)
        n = sum(1 for _ in out.open(encoding="utf-8")) - 1
        print(f"[wrote] {out} ({n} entries)")
    return 0


def _cmd_resolve(args: argparse.Namespace) -> int:
    cat = ArchiveCatalog(args.domain)
    print(cat.resolve(args.entity, args.station or None,
                      args.variable_group, args.freq))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="river/wastewater archive catalog")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="(re)generate _catalog.csv from the filesystem")
    b.add_argument("--root", help="single domain root (default: both under $DATA_DIR)")
    b.set_defaults(func=_cmd_build)
    r = sub.add_parser("resolve", help="resolve a logical name to a path")
    r.add_argument("domain")
    r.add_argument("entity")
    r.add_argument("station", nargs="?", default="")
    r.add_argument("variable_group", nargs="?", default="discharge")
    r.add_argument("freq", nargs="?", default="hourly")
    r.set_defaults(func=_cmd_resolve)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["ArchiveCatalog", "scan_domain", "write_catalog", "archive_root"]
