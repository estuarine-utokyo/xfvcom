# SPDX-License-Identifier: MIT
"""
CLI: xfvcom-make-river-nml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from xfvcom.io.river_generator import RiverNmlGenerator


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate &NML_RIVER blocks from CSV")
    ap.add_argument("csv", type=Path, help="CSV file describing rivers")
    ap.add_argument(
        "-o", "--output", type=Path, help="Output namelist file (default: stdout)"
    )
    ns = ap.parse_args()

    gen = RiverNmlGenerator(ns.csv)
    nml = gen.generate()

    if ns.output:
        ns.output.write_text(nml, encoding="utf-8")
    else:
        print(nml)


if __name__ == "__main__":
    main()
