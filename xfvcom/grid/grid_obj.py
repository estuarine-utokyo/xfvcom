from __future__ import annotations

"""Core mesh (grid) object used throughout xfvcom.

This module defines :class:`FvcomGrid`, a lightweight dataclass that
encapsulates the horizontal (2‑D) unstructured mesh used by FVCOM.

Key features
------------
* **Multiple constructors**
  * ``from_dataset`` – create from an *output* NetCDF that already contains the
    grid variables.
  * ``from_dat`` – parse the ASCII ``*_grd.dat`` file (UTM) and automatically
    compute geographic lon/lat.
* **NumPy‑based attributes** for fast numerics, plus :py:meth:`to_xarray` for
  high‑level analysis/visualisation.
* **Zero‑based connectivity** is enforced inside the class.

All in‑code comments remain in **English** as requested.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .dat_reader import read_dat
from .geo_utils import utm_to_lonlat

# -----------------------------------------------------------------------------
# Helper protocol – minimal Dataset interface
# -----------------------------------------------------------------------------


@runtime_checkable
class _DatasetLike(Protocol):
    def __getitem__(self, key: str) -> Any: ...
    def __contains__(self, key: str) -> bool: ...


# -----------------------------------------------------------------------------
# Main dataclass
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class FvcomGrid:
    """FVCOM horizontal grid (unstructured triangular mesh)."""

    # core arrays ---------------------------------------------------------
    x: NDArray[np.float64]  # node x (UTM metres)
    y: NDArray[np.float64]  # node y (UTM metres)
    nv: NDArray[np.int_]  # connectivity (3, nele) – **zero-based**

    # projection meta -----------------------------------------------------
    zone: int | None = None  # UTM zone number (1‑60)
    northern: bool = True  # hemisphere flag

    # optional geographic -----------------------------------------------
    lon: NDArray[np.float64] | None = field(default=None, repr=False)
    lat: NDArray[np.float64] | None = field(default=None, repr=False)
    lonc: NDArray[np.float64] | None = field(default=None, repr=False)
    latc: NDArray[np.float64] | None = field(default=None, repr=False)
    xc: NDArray[np.float64] | None = field(default=None, repr=False)
    yc: NDArray[np.float64] | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_dataset(cls, ds: _DatasetLike, *, validate: bool = True) -> "FvcomGrid":
        """Build from an xarray.Dataset that already contains grid variables."""
        req = ("x", "y", "nv")
        miss = [n for n in req if n not in ds]
        if miss and validate:
            raise KeyError("Dataset missing grid vars: " + ", ".join(miss))

        # ---- fallback for loose validation --------------------------------
        if not validate:
            # nv_zero → nv
            if "nv" not in ds and "nv_zero" in ds:
                ds = ds.assign(nv=ds["nv_zero"])
            # if planar x/y missing but lon/lat present, copy for plotting
            if "x" not in ds and "lon" in ds:
                ds = ds.assign(x=ds["lon"])
            if "y" not in ds and "lat" in ds:
                ds = ds.assign(y=ds["lat"])
            miss = [n for n in req if n not in ds]
            if miss:
                raise KeyError("Dataset still missing vars: " + ", ".join(miss))

        kw: dict[str, Any] = {
            "x": np.asarray(ds["x"].values, dtype=float),
            "y": np.asarray(ds["y"].values, dtype=float),
            "nv": np.asarray(ds["nv"].values, dtype=int),
        }
        for name in ("lon", "lat", "lonc", "latc"):
            if name in ds:
                kw[name] = np.asarray(ds[name].values, dtype=float)
        # compute element centres if not in dataset
        if "nv" in ds and "x" in ds and "y" in ds:
            nv = kw["nv"]
            kw["xc"] = kw["x"][nv].mean(axis=0)
            kw["yc"] = kw["y"][nv].mean(axis=0)

        return cls(**kw)  # type: ignore[arg-type]

    @classmethod
    def from_dat(
        cls,
        path: str | Path,
        *,
        utm_zone: int | None = None,
        northern: bool | None = None,
    ) -> "FvcomGrid":
        """Parse ``*_grd.dat`` and return a fully populated grid object."""
        data = read_dat(path)

        # UTM zone & hemisphere – file hint < user override ----------------
        zone = utm_zone or data["zone"]
        if zone is None:
            raise ValueError(
                "UTM zone could not be determined – pass utm_zone explicitly."
            )
        hemi = data["northern"] if northern is None else northern

        # geographic conversion ------------------------------------------
        lon, lat = utm_to_lonlat(data["x"], data["y"], zone=zone, northern=hemi)

        # sanity-check nv shape vs node count
        if data["nv"].max() >= data["x"].size:
            raise ValueError(
                "Connectivity (nv) references node index beyond range — "
                "DAT file may have been mis-parsed. "
                "Check nNode/nElem detection."
            )

        xc = data["x"][data["nv"]].mean(axis=0)
        yc = data["y"][data["nv"]].mean(axis=0)
        lonc, latc = utm_to_lonlat(xc, yc, zone=zone, northern=hemi)

        return cls(
            x=data["x"],
            y=data["y"],
            nv=data["nv"],
            zone=zone,
            northern=hemi,
            lon=lon,
            lat=lat,
            lonc=lonc,
            latc=latc,
            xc=xc,
            yc=yc,
        )

    # ------------------------------------------------------------------
    # Quick properties
    # ------------------------------------------------------------------
    @property
    def nele(self) -> int:  # number of elements
        return self.nv.shape[1]

    @property
    def node(self) -> int:  # number of nodes
        return self.x.size

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def to_xarray(self) -> xr.Dataset:
        """Return a minimal Dataset containing the grid variables."""
        ds = xr.Dataset(
            {
                "x": ("node", self.x),
                "y": ("node", self.y),
                "nv": (("three", "nele"), self.nv),
            },
            coords={
                "node": ("node", np.arange(self.node)),
                "nele": ("nele", np.arange(self.nele)),
                "three": ("three", np.arange(3)),
            },
            attrs={"cf_role": "mesh_topology"},
        )
        if self.lon is not None:
            ds["lon"] = ("node", self.lon)
        if self.lat is not None:
            ds["lat"] = ("node", self.lat)
        if self.lonc is not None:
            ds["lonc"] = ("nele", self.lonc)
        if self.latc is not None:
            ds["latc"] = ("nele", self.latc)
        if self.xc is not None:
            ds["xc"] = ("nele", self.xc)
        if self.yc is not None:
            ds["yc"] = ("nele", self.yc)

        return ds

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"FvcomGrid(nodes={self.node}, elements={self.nele}, "
            f"lonlat={'yes' if self.lon is not None else 'no'})"
        )


# -----------------------------------------------------------------------------
# Convenience wrapper – accept various inputs and always return FvcomGrid
# -----------------------------------------------------------------------------


def get_grid(obj: "FvcomGrid | xr.Dataset | str | Path") -> FvcomGrid:  # type: ignore[name-defined]
    if isinstance(obj, FvcomGrid):
        return obj
    if isinstance(obj, (str, Path)):
        return FvcomGrid.from_dat(obj)
    if isinstance(obj, xr.Dataset):
        # try strict first; if fails, fallback to loose
        try:
            return FvcomGrid.from_dataset(obj)
        except KeyError:
            return FvcomGrid.from_dataset(obj, validate=False)
    raise TypeError(
        f"Unsupported object type for grid extraction: {type(obj).__name__}"
    )
