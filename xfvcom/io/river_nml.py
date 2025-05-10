# ~/Github/xfvcom/xfvcom/io/river_nml.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

_NML_START = re.compile(r"^\s*&NML_RIVER", re.I)
_KV_PAIR = re.compile(r"^\s*([A-Za-z_]+)\s*=\s*('?[^,']+'?|[^,]+)\s*,?\s*$")


def parse_river_namelist(
    file_path: str | Path,
    *,
    to_zero_based: bool = True,
) -> pd.DataFrame:
    """
    Read a FVCOM river namelist and return its content as a pandas.DataFrame.

    Parameters
    ----------
    file_path : str | Path
        Path to the namelist file.
    to_zero_based : bool, default True
        If True, convert ``RIVER_GRID_LOCATION`` from 1-based (Fortran)
        to 0-based (Python/xfvcom) indexing.

    Returns
    -------
    pandas.DataFrame
        Columns: ``river_name``, ``river_file``, ``grid_location``,
        ``vertical_distribution``.
    """
    file_path = Path(file_path).expanduser()
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    with file_path.open(encoding="utf-8") as fp:
        for line in fp:
            if _NML_START.match(line):
                current = {}
                continue
            if line.strip().startswith("/"):  # end of one block
                if current:
                    blocks.append(current)
                current = None
                continue
            if current is None:  # outside a block
                continue
            if m := _KV_PAIR.match(line):
                key, val = m.groups()
                val = val.strip().strip("'")  # remove quotes if any
                key = key.lower()
                if key == "river_grid_location":
                    val = int(val) - 1 if to_zero_based else int(val)
                current[key] = val

    df = pd.DataFrame(blocks).rename(
        columns={
            "river_name": "name",
            "river_file": "file",
            "river_grid_location": "grid_location",
            "river_vertical_distribution": "vertical_distribution",
        }
    )
    return df
