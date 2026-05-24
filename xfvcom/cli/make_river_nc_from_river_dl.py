# -*- coding: utf-8 -*-
"""Generate FVCOM river/sewer forcing NetCDF from a river_dl archive.

Wraps :class:`xfvcom.io.river_nc_generator.RiverNetCDFGenerator` with a YAML
mapping that pins each FVCOM river name (as it appears in
``RIVERS_NAMELIST*.nml``) to a per-river ``discharge_hourly.nc`` produced by
``river_dl``. The same mechanism handles sewers (which FVCOM models as
zero-bathymetry rivers); just point them at
``$DATA_DIR/wastewater/<Plant>/discharge_hourly.nc``.

See ``TB-FVCOM/hydro/docs/bc_construction_protocol.md`` §5.2 for the spec.

YAML map schema (example)::

    defaults:
      temp: 15.0          # river temperature [degC] (sewers usually 20)
      salt: 0.0           # always 0 for both rivers and sewers
    rivers:
      - name: EastArakawa
        source: ${DATA_DIR}/river/discharge/Arakawa/Iwabuchi/discharge_hourly.nc
        scale: 1.0        # optional; default 1.0 (leave at 1 for runtime tuning)
      - name: Shibaura
        source: ${DATA_DIR}/wastewater/Shibaura/discharge_hourly.nc
        temp: 20.0
      # kind: constant — for sources with no upstream river_dl NetCDF.
      # The (flux, temp, salt) tuple is broadcast across the time axis.
      - name: Kisarazu
        kind: constant
        flux: 0.2902       # m^3/s, constant
        temp: 10.0
        salt: 0.0          # optional; default from defaults: block
        # 'source' is forbidden; 'scale' is forbidden (already in physical units).

Environment variables in the ``source`` field (``${DATA_DIR}`` etc.) are
expanded by ``os.path.expandvars``.

Example
-------
::

    xfvcom-make-river-nc-from-river-dl \\
        --nml ~/Github/TB-FVCOM/input/goto2023/river/RIVERS_NAMELIST.nml \\
        --river-map river_dl_map.yaml \\
        --start 2020-01-01 --end 2021-01-01 \\
        -o tb18_river_2020.nc
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from xfvcom.io.river_nc_generator import RiverNetCDFGenerator
from xfvcom.io.sources.archive import ArchiveCatalog


def _expand(path: str) -> str:
    """Expand environment variables and ``~`` in a path string."""
    return os.path.expanduser(os.path.expandvars(path))


_CATALOG_CACHE: dict[str, ArchiveCatalog] = {}


def _resolve_logical_source(name: str, spec: dict[str, Any]) -> Path:
    """Resolve a logical ``source`` mapping to a path via the archive catalog.

    Schema v4 (2026-05-24 archive reorg): a river-map entry's ``source``
    may be a mapping ``{domain, entity, station, group, freq}`` resolved
    against ``$DATA_DIR/<domain>/_catalog.csv`` instead of a hard-coded
    path. This decouples the map from the archive's on-disk layout — a
    future restructure updates the catalog, not this YAML. A plain string
    ``source`` (a path) remains supported for backward compatibility.
    """
    dom = spec.get("domain")
    if dom not in ("river", "wastewater"):
        raise ValueError(
            f"river-map entry {name!r}: source.domain={dom!r} must be "
            f"'river' or 'wastewater'"
        )
    if "entity" not in spec:
        raise ValueError(
            f"river-map entry {name!r}: logical source missing 'entity'"
        )
    cat = _CATALOG_CACHE.get(dom)
    if cat is None:
        cat = _CATALOG_CACHE[dom] = ArchiveCatalog(dom)
    return cat.resolve(
        str(spec["entity"]),
        station=(str(spec["station"]) if spec.get("station") not in (None, "") else None),
        variable_group=str(spec.get("group", "discharge")),
        freq=str(spec.get("freq", "hourly")),
    )


_VALID_TEMP_SOURCE_KINDS = {"air_regression", "monthly_climatology"}


def _validate_temp_source(name: str, spec: Any) -> dict[str, Any]:
    """Validate a ``temp_source`` dict (per-row or inherited default).

    Returns a normalised dict ready to hand to ``RiverNetCDFGenerator``.
    Raises ``ValueError`` with a single-line message naming the
    offending entry on any schema violation.
    """
    if not isinstance(spec, dict):
        raise ValueError(
            f"river-map entry {name!r}: 'temp_source' must be a mapping, "
            f"got {type(spec).__name__}"
        )
    kind = spec.get("kind")
    if kind not in _VALID_TEMP_SOURCE_KINDS:
        raise ValueError(
            f"river-map entry {name!r}: temp_source.kind={kind!r} not "
            f"supported (expected one of "
            f"{sorted(_VALID_TEMP_SOURCE_KINDS)})"
        )
    out: dict[str, Any] = {"kind": kind}
    if kind == "air_regression":
        for k in (
            "air_nc_template",
            "air_var",
            "air_lat",
            "air_lon",
            "slope",
            "intercept",
        ):
            if k not in spec:
                raise ValueError(
                    f"river-map entry {name!r}: temp_source.kind=air_regression "
                    f"missing required key {k!r}"
                )
        out["air_nc_template"] = _expand(str(spec["air_nc_template"]))
        out["air_var"] = str(spec["air_var"])
        out["air_lat"] = float(spec["air_lat"])
        out["air_lon"] = float(spec["air_lon"])
        out["slope"] = float(spec["slope"])
        out["intercept"] = float(spec["intercept"])
        if "min_temp" in spec:
            out["min_temp"] = float(spec["min_temp"])
        if "max_temp" in spec:
            out["max_temp"] = float(spec["max_temp"])
    elif kind == "monthly_climatology":
        means = spec.get("monthly_means")
        if not isinstance(means, list) or len(means) != 12:
            raise ValueError(
                f"river-map entry {name!r}: temp_source.kind=monthly_climatology "
                f"requires monthly_means as a 12-element list (Jan..Dec)"
            )
        out["monthly_means"] = [float(m) for m in means]
    return out


def _load_river_map(
    yaml_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """Parse the river_dl YAML map (schema v3).

    Returns ``(river_dl_map, defaults)``. ``river_dl_map`` is keyed by river
    name; each value carries at least ``source: Path`` (or constant fields)
    plus a required ``temp_source`` mapping that drives the river_temp
    column at NC build time.

    Schema v3 changes vs v2:
      * ``temp:`` (a scalar) is REJECTED.  The user's rule "周年計算で
        定数はあり得ない" forbids a constant water-T for annual runs.
      * ``temp_source:`` is REQUIRED (per-row or via the defaults block).
        Two supported ``kind`` values:
        - ``air_regression``: T_water = slope * T_air + intercept,
          where T_air comes from a metforce NC at a specified
          (lat, lon).  Required subfields:
          ``air_nc_template``, ``air_var``, ``air_lat``, ``air_lon``,
          ``slope``, ``intercept``.  Optional ``min_temp`` / ``max_temp``
          clip.
        - ``monthly_climatology``: T_water taken from a 12-element
          ``monthly_means`` list, broadcast on the time axis as a
          month-of-year step function.

    See ``~/Github/TB-FVCOM/hydro/docs/directions/20260517_xfvcom_river_temp_seasonal.md``
    for the full design.
    """
    with yaml_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    defaults_raw = raw.get("defaults", {}) or {}
    if "temp" in defaults_raw:
        raise ValueError(
            "river-map defaults: 'temp:' scalar field is not allowed in "
            "schema v3; declare a 'temp_source' instead (kind: "
            "air_regression or kind: monthly_climatology)"
        )
    defaults = {
        "flux": float(defaults_raw.get("flux", 0.0)),
        "salt": float(defaults_raw.get("salt", 0.0)),
    }
    default_temp_source: dict[str, Any] | None = None
    if "temp_source" in defaults_raw:
        default_temp_source = _validate_temp_source(
            "<defaults>", defaults_raw["temp_source"]
        )

    river_dl_map: dict[str, dict[str, Any]] = {}
    for rv in raw.get("rivers", []):
        name = rv.get("name")
        if not name:
            raise ValueError(f"river-map entry missing required key 'name': {rv!r}")
        if "temp" in rv:
            raise ValueError(
                f"river-map entry {name!r}: 'temp:' scalar field is not "
                f"allowed in schema v3; declare a 'temp_source' instead "
                f"(kind: air_regression or kind: monthly_climatology)"
            )
        kind = str(rv.get("kind", "river_dl"))
        entry: dict[str, Any] = {"kind": kind}

        if kind == "constant":
            # Constant source: no upstream NetCDF; broadcast a fixed
            # flux (and salt) on the time axis.  Temperature comes from
            # the row's temp_source like every other entry.
            if "source" in rv:
                raise ValueError(
                    f"river-map entry {name!r} declares kind: constant "
                    f"but also carries a 'source' field; remove one or "
                    f"the other"
                )
            if "scale" in rv:
                raise ValueError(
                    f"river-map entry {name!r} declares kind: constant; "
                    f"'scale' is meaningless because the value is "
                    f"already in physical units"
                )
            if "flux" not in rv:
                raise ValueError(
                    f"river-map entry {name!r} declares kind: constant "
                    f"but is missing required key 'flux'"
                )
            entry["flux"] = float(rv["flux"])
            entry["salt"] = float(rv.get("salt", defaults["salt"]))
        elif kind == "river_dl":
            src = rv.get("source")
            if not src:
                raise ValueError(
                    f"river-map entry {name!r} missing required key "
                    f"'source' (kind: river_dl): {rv!r}"
                )
            if isinstance(src, dict):
                # Schema v4: logical source resolved via the archive catalog.
                entry["source"] = _resolve_logical_source(name, src)
            else:
                # Backward-compatible: explicit path string.
                entry["source"] = Path(_expand(str(src)))
            if "scale" in rv:
                entry["scale"] = float(rv["scale"])
            entry["salt"] = float(rv.get("salt", defaults["salt"]))
        else:
            raise ValueError(
                f"river-map entry {name!r} has unsupported kind: "
                f"{kind!r} (expected 'river_dl' or 'constant')"
            )

        if "temp_source" in rv:
            entry["temp_source"] = _validate_temp_source(name, rv["temp_source"])
        elif default_temp_source is not None:
            entry["temp_source"] = dict(default_temp_source)
        else:
            raise ValueError(
                f"river-map entry {name!r}: 'temp_source' is required "
                f"in schema v3 (per-row or via the defaults block); "
                f"declare kind: air_regression or kind: monthly_climatology"
            )

        river_dl_map[name] = entry
    return river_dl_map, defaults


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Generate FVCOM river / sewer forcing NetCDF-4 from a river_dl "
            "archive (per-river discharge_hourly.nc) via an explicit "
            "name → file mapping."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:

  xfvcom-make-river-nc-from-river-dl \\
      --nml ~/Github/TB-FVCOM/input/goto2023/river/RIVERS_NAMELIST.nml \\
      --river-map river_dl_map.yaml \\
      --start 2020-01-01 --end 2021-01-01 \\
      -o tb18_river_2020.nc
""",
    )
    p.add_argument(
        "--nml",
        required=True,
        type=Path,
        help="RIVERS_NAMELIST*.nml file (defines the river name list)",
    )
    p.add_argument(
        "--river-map",
        required=True,
        type=Path,
        help="YAML mapping from river name → discharge_hourly.nc path",
    )
    p.add_argument("--start", required=True, help="ISO time, UTC")
    p.add_argument("--end", required=True, help="ISO time, UTC")
    p.add_argument(
        "--start-tz",
        default="UTC",
        help="Timezone for naive start/end (default: UTC)",
    )
    p.add_argument("--dt", type=int, default=3600, help="time step [s]")

    # Defaults applied to rivers NOT in the map (rare; usually all rivers
    # are listed). These override the YAML 'defaults:' block if given.
    p.add_argument("--flux", type=float, help="default flux (m3/s) override")
    p.add_argument("--temp", type=float, help="default temp (degC) override")
    p.add_argument("--salt", type=float, help="default salt (PSU) override")

    p.add_argument("-o", "--output", type=Path, help="Output NetCDF file")
    args = p.parse_args()

    river_dl_map, defaults = _load_river_map(args.river_map)

    # CLI overrides take precedence over YAML defaults.
    flux = args.flux if args.flux is not None else defaults["flux"]
    salt = args.salt if args.salt is not None else defaults["salt"]
    # NB: schema v3 makes per-row temp_source mandatory; ``--temp`` is a
    # CLI default only consulted when the generator's _ScalarConstantSource
    # fallback (priority 3 inside _choose_source) is reached -- which no
    # longer happens for any river_dl_map entry, because every entry has
    # a temp_source.  We keep ``--temp`` for backward compat callers that
    # bypass the river_dl_map path entirely, but warn if it's set.
    if args.temp is not None:
        print(
            f"[WARN] --temp={args.temp} is ignored for river_dl_map entries; "
            f"all temperature now flows through YAML temp_source.",
            file=sys.stderr,
        )
    temp = args.temp if args.temp is not None else 15.0

    # Validate every river_dl entry's source file exists; fail fast.
    # kind: constant entries have no source file and are skipped.
    missing = [
        name
        for name, spec in river_dl_map.items()
        if spec.get("kind", "river_dl") == "river_dl"
        and not Path(spec["source"]).exists()
    ]
    if missing:
        print(
            "[ERROR] river_dl source NetCDF missing for: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 2

    gen = RiverNetCDFGenerator(
        args.nml,
        args.start,
        args.end,
        args.dt,
        flux,
        temp,
        salt,
        start_tz=args.start_tz,
        river_dl_map=river_dl_map,
    )

    out_path = args.output if args.output else args.nml.with_suffix(".nc")
    gen.write(out_path)
    print(f"Written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
