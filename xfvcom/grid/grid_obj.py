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
        }
        # ------------------------------------------------------------------
        # nv: ensure (3, nele) shape & 0-based
        nv_raw = np.asarray(ds["nv"].values, dtype=int)
        if nv_raw.shape[0] != 3 and nv_raw.shape[1] == 3:
            nv_raw = nv_raw.T  # (nele,3) → (3,nele)
        # 1-based → 0-based
        if nv_raw.min() == 1:
            nv_raw = nv_raw - 1
        kw["nv"] = nv_raw

        for name in ("lon", "lat", "lonc", "latc"):
            if name in ds:
                kw[name] = np.asarray(ds[name].values, dtype=float)
        # compute element centres
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
    # Area calculation methods
    # ------------------------------------------------------------------
    def calculate_node_area(
        self,
        node_indices: list[int] | NDArray[np.int_] | None = None,
        index_base: int = 1,
    ) -> float:
        """Calculate total area of triangular elements containing specified nodes.

        Parameters
        ----------
        node_indices : list[int] | NDArray[np.int_] | None
            List of node indices. If None, calculates area for all nodes.
        index_base : int
            0 for zero-based indexing, 1 for one-based indexing (FVCOM default)

        Returns
        -------
        float
            Total area in square meters (assuming x, y are in meters/UTM)
        """
        import numpy as np

        # Handle None case - all nodes
        if node_indices is None:
            node_indices = np.arange(
                1 if index_base == 1 else 0, self.node + (1 if index_base == 1 else 0)
            )

        # Convert to numpy array
        node_indices_arr: NDArray[np.int_] = np.asarray(node_indices, dtype=int)

        # Convert to zero-based if needed
        if index_base == 1:
            node_indices_0 = node_indices_arr - 1
        else:
            node_indices_0 = node_indices_arr

        # Validate indices
        if np.any(node_indices_0 < 0) or np.any(node_indices_0 >= self.node):
            invalid = node_indices_arr[
                (node_indices_0 < 0) | (node_indices_0 >= self.node)
            ]
            raise ValueError(
                f"Invalid node indices (base-{index_base}): {invalid.tolist()}. "
                f"Valid range: {1 if index_base == 1 else 0} to "
                f"{self.node if index_base == 1 else self.node - 1}"
            )

        # Find all elements containing any of the selected nodes
        # nv is already 0-based internally
        node_set = set(node_indices_0.tolist())  # Convert to list for set creation
        element_mask: NDArray[np.bool_] = np.zeros(self.nele, dtype=bool)

        for i in range(3):  # Check each vertex of triangles
            element_mask |= np.isin(self.nv[i, :], list(node_set))

        # Get unique elements containing selected nodes
        selected_elements = np.where(element_mask)[0]

        if len(selected_elements) == 0:
            return 0.0

        # Calculate area of each selected triangle using shoelace formula
        total_area = 0.0

        for elem_idx in selected_elements:
            # Get the three nodes of this triangle (0-based)
            n1, n2, n3 = self.nv[:, elem_idx]

            # Get coordinates
            x1, y1 = self.x[n1], self.y[n1]
            x2, y2 = self.x[n2], self.y[n2]
            x3, y3 = self.x[n3], self.y[n3]

            # Shoelace formula for triangle area
            area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            total_area += area

        return total_area

    def get_node_element_boundaries(
        self,
        node_indices: list[int] | NDArray[np.int_] | None = None,
        index_base: int = 1,
        return_as: str = "lines",  # "lines" or "polygons"
    ) -> list[list[tuple[float, float]]]:
        """Get boundaries of triangular elements containing specified nodes.

        Parameters
        ----------
        node_indices : list[int] | NDArray[np.int_] | None
            List of node indices. If None, gets boundaries for all elements.
        index_base : int
            0 for zero-based indexing, 1 for one-based indexing (FVCOM default)
        return_as : str
            "lines" returns list of line segments (edges)
            "polygons" returns list of closed triangles

        Returns
        -------
        list[list[tuple[float, float]]]
            List of polylines/polygons as coordinate pairs (lon, lat)
            For "lines": each item is a line segment with 2 points
            For "polygons": each item is a closed triangle with 4 points (first=last)
        """
        import numpy as np

        # Handle None case - all nodes
        if node_indices is None:
            node_indices = np.arange(
                1 if index_base == 1 else 0, self.node + (1 if index_base == 1 else 0)
            )

        # Convert to numpy array
        node_indices_arr: NDArray[np.int_] = np.asarray(node_indices, dtype=int)

        # Convert to zero-based if needed
        if index_base == 1:
            node_indices_0 = node_indices_arr - 1
        else:
            node_indices_0 = node_indices_arr

        # Validate indices
        if np.any(node_indices_0 < 0) or np.any(node_indices_0 >= self.node):
            invalid = node_indices_arr[
                (node_indices_0 < 0) | (node_indices_0 >= self.node)
            ]
            raise ValueError(
                f"Invalid node indices (base-{index_base}): {invalid.tolist()}. "
                f"Valid range: {1 if index_base == 1 else 0} to "
                f"{self.node if index_base == 1 else self.node - 1}"
            )

        # Find all elements containing any of the selected nodes
        node_set = set(node_indices_0.tolist())
        element_mask: NDArray[np.bool_] = np.zeros(self.nele, dtype=bool)

        for i in range(3):  # Check each vertex of triangles
            element_mask |= np.isin(self.nv[i, :], list(node_set))

        # Get unique elements containing selected nodes
        selected_elements = np.where(element_mask)[0]

        if len(selected_elements) == 0:
            return []

        # Use geographic coordinates if available, otherwise use UTM
        if self.lon is not None and self.lat is not None:
            x_coords = self.lon
            y_coords = self.lat
        else:
            x_coords = self.x
            y_coords = self.y

        boundaries = []

        if return_as == "polygons":
            # Return closed triangles
            for elem_idx in selected_elements:
                # Get the three nodes of this triangle (0-based)
                n1, n2, n3 = self.nv[:, elem_idx]

                # Create closed polygon (first point repeated at end)
                polygon = [
                    (float(x_coords[n1]), float(y_coords[n1])),
                    (float(x_coords[n2]), float(y_coords[n2])),
                    (float(x_coords[n3]), float(y_coords[n3])),
                    (float(x_coords[n1]), float(y_coords[n1])),  # Close the polygon
                ]
                boundaries.append(polygon)

        elif return_as == "lines":
            # Return unique edges as line segments
            edges_set = set()

            for elem_idx in selected_elements:
                # Get the three nodes of this triangle (0-based)
                n1, n2, n3 = self.nv[:, elem_idx]

                # Add three edges (sorted to ensure uniqueness)
                edges = [
                    tuple(sorted([n1, n2])),
                    tuple(sorted([n2, n3])),
                    tuple(sorted([n3, n1])),
                ]
                edges_set.update(edges)

            # Convert edges to line segments
            for edge in edges_set:
                n1, n2 = edge
                line = [
                    (float(x_coords[n1]), float(y_coords[n1])),
                    (float(x_coords[n2]), float(y_coords[n2])),
                ]
                boundaries.append(line)
        else:
            raise ValueError(
                f"return_as must be 'lines' or 'polygons', got '{return_as}'"
            )

        return boundaries

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
