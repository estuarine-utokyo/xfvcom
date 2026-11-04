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
from ..fortran_numbers import fortran_real
from ..fvcom_input import first_record_index, record_fields, scan_count_headers

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

    # ---> jsasaki 2026-11-05: review round 18 #1-#7: the header through the shared FVCOM scanner
    # (exact key 'Node Number', one positive integer, the 400-byte buffer, GET_VALUE conversion of
    # the traversed lines); records as list-directed fields with 'r*value' expanded.  Comment-only
    # lines are skipped and malformed records passed over, as this reader always did.
    with path.open() as fp:
        lines = [(n, ln.rstrip("\n")) for n, ln in enumerate(fp, 1) if ln.strip()]
    start = first_record_index(lines)
    counts = scan_count_headers(
        path, lines, {"Node Number": "node"}, start, block="depth records"
    )
    if "node" not in counts:
        raise ValueError(f"{path}: 'Node Number = N' header not found")
    node = counts["node"]
    idx = start

    x: NDArray[np.float64] = np.empty(node, dtype=np.float64)
    y: NDArray[np.float64] = np.empty(node, dtype=np.float64)
    h: NDArray[np.float64] = np.empty(node, dtype=np.float64)
    read = 0
    while read < node and idx < len(lines):
        parts = record_fields(lines[idx][1])
        idx += 1
        if len(parts) < 3:
            continue
        try:
            # jsasaki 2026-11-05: review round 14 #2: Fortran real literals ('1.0D+1', '1.0-3')
            x[read], y[read], h[read] = (
                fortran_real(parts[0]),
                fortran_real(parts[1]),
                fortran_real(parts[2]),
            )
        except ValueError:
            continue
        read += 1
    linenos = [n for n, _ln in lines]
    # <--- jsasaki 2026-11-05
    if read != node:
        # jsasaki 2026-11-05: review round 16 #3: name the last line read
        last = f" (last line read: {linenos[idx - 1]})" if idx > 0 else ""
        raise ValueError(f"{path}: depth block ended prematurely ({read}/{node}){last}")

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
