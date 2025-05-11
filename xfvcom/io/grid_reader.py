# -*- coding: utf-8 -*-
"""Parse FVCOM ASCII grid (.dat) into DataFrame or xarray.Dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


class GridASCII:
    """Light-weight container for FVCOM grid read from *.dat*."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.nnode: int
        self.nele: int
        self.nv: np.ndarray  # shape (nele, 3) int32, 1-origin
        self.nodes: pd.DataFrame  # columns = x, y, depth

        self._parse()

    # ------------------------------------------------------------------
    def _parse(self) -> None:
        with self.path.open() as fp:
            self.nnode = int(fp.readline().split("=")[1])
            self.nele = int(fp.readline().split("=")[1])

            # cell connectivity (nv)
            nv = np.loadtxt(
                [fp.readline() for _ in range(self.nele)],
                usecols=(2, 3, 4),
                dtype="i4",
            )
            self.nv = nv  # shape (nele, 3)

            # node block
            cols = ("x", "y", "depth")
            node_arr = np.loadtxt(
                [fp.readline() for _ in range(self.nnode)],
                usecols=(1, 2, 3),
                dtype="f8",
            )
            self.nodes = pd.DataFrame(node_arr, columns=cols)  # index = node-1

    # ------------------------------------------------------------------
    def to_xarray(self) -> xr.Dataset:
        """Return basic mesh variables as xarray Dataset."""
        ds = xr.Dataset(
            data_vars=dict(
                x=("node", self.nodes["x"].values.astype("f8")),
                y=("node", self.nodes["y"].values.astype("f8")),
                nv=(("nele", "three"), self.nv.astype("i4")),
                depth=("node", self.nodes["depth"].values.astype("f8")),
            ),
            coords={
                "node": np.arange(1, self.nnode + 1, dtype="i4"),
                "nele": np.arange(1, self.nele + 1, dtype="i4"),
                "three": [1, 2, 3],
            },
        )
        return ds
