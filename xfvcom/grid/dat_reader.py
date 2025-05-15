from __future__ import annotations

"""Robust reader for FVCOM `*_grd.dat` mesh files (no UTM info)."""

import re
from pathlib import Path
from typing import TypedDict

import numpy as np

__all__ = ["read_dat"]


class _GridData(TypedDict):
    x: np.ndarray
    y: np.ndarray
    nv: np.ndarray  # (3, nele)
    zone: int | None
    northern: bool


def _is_int(tok: str) -> bool:
    return tok.lstrip("+-").isdigit()


def read_dat(path: str | Path) -> _GridData:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    # -------------------------------------------------- read non-blank lines
    with path.open() as fp:
        lines = [ln.rstrip("\n") for ln in fp if ln.strip()]

    # ---------------------------------------- locate counts (node / nele)
    node = nele = None  # type: ignore[assignment]
    idx = 0

    while idx < len(lines):
        ln = lines[idx]
        low = ln.lower()

        if low.startswith("node number"):
            # e.g. "Node Number = 3210"
            digits = re.findall(r"\d+", ln)
            if digits:
                node = int(digits[0])
            idx += 1
            continue
        if low.startswith("cell number") or low.startswith("elem number"):
            digits = re.findall(r"\d+", ln)
            if digits:
                nele = int(digits[0])
            idx += 1
            break
        idx += 1

    if node is None or nele is None:
        raise ValueError("Node/Cell counts not found in DAT file")

    # ----------------------------------------------------- element block (comes first)
    nv = np.empty((3, nele), dtype=int)
    read_elem = 0
    while read_elem < nele and idx < len(lines):
        parts = lines[idx].split()
        idx += 1
        if len(parts) != 5 or not _is_int(parts[0]):
            continue
        nv[:, read_elem] = [int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1]
        read_elem += 1
    if read_elem != nele:
        raise ValueError(f"Element block ended prematurely ({read_elem}/{nele})")

    # -------------------------------------------------------- node block (after elem)
    x = np.empty(node, dtype=float)
    y = np.empty(node, dtype=float)
    read_node = 0
    while read_node < node and idx < len(lines):
        parts = lines[idx].split()
        idx += 1
        if len(parts) != 4 or not _is_int(parts[0]):
            continue
        try:
            x[read_node] = float(parts[1])
            y[read_node] = float(parts[2])
        except ValueError:
            continue
        read_node += 1
    if read_node != node:
        raise ValueError(f"Node block ended prematurely ({read_node}/{node})")

    return _GridData(x=x, y=y, nv=nv, zone=None, northern=True)
