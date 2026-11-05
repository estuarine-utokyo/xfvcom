"""Reader/writer for FVCOM ``*_dep.dat`` bathymetry files.

Format (the FVCOM ``DEPTH_FILE``)::

    Node Number = 3210
    388431.892140 3946011.482976 4.680936
    388710.063711 3945738.208287 4.637125
    ...

i.e. a ``Node Number = N`` header followed by ``N`` whitespace-separated
``X  Y  H`` records (UTM x, UTM y, depth in metres, positive down). Unlike the
``*_grd.dat`` node block there is no leading index column and no ``Cell Number``
line. The companion grid reader is :func:`xfvcom.grid.read_dat`.
"""

from __future__ import annotations

# jsasaki 2026-11-05: review round 18 #4: 're' no longer needed (header regex replaced)
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# jsasaki 2026-11-05: review round 14 #2: FVCOM reads depth records list-directed
# jsasaki 2026-11-05: review round 17 #4: list-directed fields (commas separate like blanks)
# jsasaki 2026-11-05: review round 18 #1-#7: list_fields reached through the shared scanner
# jsasaki 2026-11-05: review round 20 #6: unused fortran_real import removed (conversion is in list_read)
# jsasaki 2026-11-05: review round 19 #1 #3 #4 #5: READ-based header search and records
# jsasaki 2026-11-05: review round 21 #1: physical_lines splits records at '\n' only
from ..fvcom_input import (
    REAL,
    locate_records,
    physical_lines,
    read_all,
    scan_count_headers,
)

__all__ = ["DepData", "read_dep", "write_dep"]


@dataclass
class DepData:
    """Parsed ``*_dep.dat``: node coordinates + depth."""

    x: NDArray[np.float64]  #: (M,) UTM x
    y: NDArray[np.float64]  #: (M,) UTM y
    h: NDArray[np.float64]  #: (M,) depth [m], positive down

    @property
    def n_node(self) -> int:
        return self.h.shape[0]


def read_dep(path: str | Path) -> DepData:
    """Read an FVCOM ``*_dep.dat`` bathymetry file -> :class:`DepData`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    # jsasaki 2026-11-05: review round 18 #1-#7: the header through the shared FVCOM scanner (exact key
    # 'Node Number', one positive integer, the 400-byte buffer, GET_VALUE conversion of the traversed lines)
    # ---> jsasaki 2026-11-05: review round 19 #1 #3 #4 #5: all physical lines (blanks included);
    # the depth block starts where FVCOM's header search first READs X,Y,H successfully (a free-text
    # title is a failed READ, i.e. a header line), with its NHEADER rewind checked; records are
    # list-directed READs that may continue onto further lines or end with '/'.  READs that fail
    # (or leave an item unset) are passed over, as this reader always did.
    # jsasaki 2026-11-05: review round 21 #1: split at '\n' only, not with universal newlines
    lines = physical_lines(path)
    types = (REAL, REAL, REAL)
    start = locate_records(path, lines, types, block="depth records")
    counts = scan_count_headers(
        path, lines, {"Node Number": "node"}, start, block="depth records"
    )
    if "node" not in counts:
        raise ValueError(f"{path}: 'Node Number = N' header not found")
    node = counts["node"]

    x: NDArray[np.float64] = np.empty(node, dtype=np.float64)
    y: NDArray[np.float64] = np.empty(node, dtype=np.float64)
    h: NDArray[np.float64] = np.empty(node, dtype=np.float64)
    reads, _eof = read_all(lines, start, types)
    read = 0
    last = 0
    for rd in reads:
        if read == node:
            break
        last = rd.last_line
        if rd.status != "ok":
            continue
        x[read], y[read], h[read] = (float(v) for v in rd.values)  # type: ignore[arg-type]
        read += 1
    if read != node:
        tail = f" (last line read: {last})" if last else ""
        raise ValueError(f"{path}: depth block ended prematurely ({read}/{node}){tail}")
    # <--- jsasaki 2026-11-05

    return DepData(x=x, y=y, h=h)


def write_dep(
    path: str | Path,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    h: NDArray[np.float64],
    *,
    fmt: str = "%.6f",
) -> None:
    """Write an FVCOM ``*_dep.dat`` (``Node Number = N`` + ``X Y H`` records)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    if not (x.shape == y.shape == h.shape):
        raise ValueError("write_dep: x, y, h must have the same shape")
    path = Path(path)
    with path.open("w") as fp:
        fp.write(f"Node Number = {x.shape[0]}\n")
        for xi, yi, hi in zip(x, y, h):
            fp.write(f"{fmt % xi} {fmt % yi} {fmt % hi}\n")
