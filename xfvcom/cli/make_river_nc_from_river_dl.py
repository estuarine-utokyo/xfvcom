# -*- coding: utf-8 -*-
"""Generate FVCOM river/sewer forcing NetCDF from a river_dl archive.

Wraps :class:`xfvcom.io.river_nc_generator.RiverNetCDFGenerator` with a YAML
mapping that pins each FVCOM river name (as it appears in
``RIVERS_NAMELIST*.nml``) to a per-river ``discharge_hourly.nc`` produced by
``river_dl``. The same mechanism handles sewers (which FVCOM models as
zero-bathymetry rivers); just point them at
``$DATA_DIR/river_dl/sewage/<Plant>/discharge_hourly.nc``.

See ``TB-FVCOM/hydro/docs/bc_construction_protocol.md`` §5.2 for the spec.

YAML map schema (example)::

    defaults:
      temp: 15.0          # river temperature [degC] (sewers usually 20)
      salt: 0.0           # always 0 for both rivers and sewers
    rivers:
      - name: EastArakawa
        source: ${DATA_DIR}/river_dl/discharge/Arakawa/Iwabuchi/discharge_hourly.nc
        scale: 1.0        # optional; default 1.0 (leave at 1 for runtime tuning)
      - name: Shibaura
        source: ${DATA_DIR}/river_dl/sewage/Shibaura/discharge_hourly.nc
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


def _expand(path: str) -> str:
    """Expand environment variables and ``~`` in a path string."""
    return os.path.expanduser(os.path.expandvars(path))


def _load_river_map(
    yaml_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """Parse the river_dl YAML map.

    Returns ``(river_dl_map, defaults)``. ``river_dl_map`` is keyed by river
    name; each value carries at least ``source: Path`` plus optional
    ``scale``, ``temp``, ``salt``. ``defaults`` is a flat mapping passed to
    the generator's CLI-level fallbacks.
    """
    with yaml_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    defaults = {
        "flux": float(raw.get("defaults", {}).get("flux", 0.0)),
        "temp": float(raw.get("defaults", {}).get("temp", 15.0)),
        "salt": float(raw.get("defaults", {}).get("salt", 0.0)),
    }

    river_dl_map: dict[str, dict[str, Any]] = {}
    for rv in raw.get("rivers", []):
        name = rv.get("name")
        if not name:
            raise ValueError(f"river-map entry missing required key 'name': {rv!r}")
        kind = str(rv.get("kind", "river_dl"))
        entry: dict[str, Any] = {"kind": kind}

        if kind == "constant":
            # Constant source: no upstream NetCDF; broadcast a fixed
            # (flux, temp, salt) tuple on the time axis.
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
            entry["temp"] = float(rv.get("temp", defaults["temp"]))
            entry["salt"] = float(rv.get("salt", defaults["salt"]))
        elif kind == "river_dl":
            src = rv.get("source")
            if not src:
                raise ValueError(
                    f"river-map entry {name!r} missing required key "
                    f"'source' (kind: river_dl): {rv!r}"
                )
            entry["source"] = Path(_expand(str(src)))
            if "scale" in rv:
                entry["scale"] = float(rv["scale"])
            entry["temp"] = float(rv.get("temp", defaults["temp"]))
            entry["salt"] = float(rv.get("salt", defaults["salt"]))
        else:
            raise ValueError(
                f"river-map entry {name!r} has unsupported kind: "
                f"{kind!r} (expected 'river_dl' or 'constant')"
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
    temp = args.temp if args.temp is not None else defaults["temp"]
    salt = args.salt if args.salt is not None else defaults["salt"]

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
