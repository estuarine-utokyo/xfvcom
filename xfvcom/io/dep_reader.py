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

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

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

    with path.open() as fp:
        lines = [ln.strip() for ln in fp if ln.strip()]

    node: int | None = None
    idx = 0
    while idx < len(lines):
        low = lines[idx].lower()
        if low.startswith("node number"):
            digits = re.findall(r"\d+", lines[idx])
            if digits:
                node = int(digits[0])
            idx += 1
            break
        idx += 1
    if node is None:
        raise ValueError(f"{path}: 'Node Number = N' header not found")

    x: NDArray[np.float64] = np.empty(node, dtype=np.float64)
    y: NDArray[np.float64] = np.empty(node, dtype=np.float64)
    h: NDArray[np.float64] = np.empty(node, dtype=np.float64)
    read = 0
    while read < node and idx < len(lines):
        parts = lines[idx].split()
        idx += 1
        if len(parts) < 3:
            continue
        try:
            x[read], y[read], h[read] = (
                float(parts[0]),
                float(parts[1]),
                float(parts[2]),
            )
        except ValueError:
            continue
        read += 1
    if read != node:
        raise ValueError(f"{path}: depth block ended prematurely ({read}/{node})")

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
