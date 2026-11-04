"""Connectivity (``nv``) normalisation shared by the grid object and the I/O layer.

Every consumer that turns a dataset's ``nv`` into array indices goes through
:func:`zero_based_connectivity`, so that orientation and indexing base are
decided in one place:

* **orientation** comes from the dimension names (``three`` / ``nele``) when
  they are present; only unnamed arrays fall back to the shape, and an unnamed
  ``(3, 3)`` array is refused as ambiguous;
* **indexing base** comes, in order, from an explicit ``start_index`` argument,
  from the ``nv.attrs["start_index"]`` attribute (0 or 1, written by
  :meth:`xfvcom.grid.FvcomGrid.to_xarray`), and only then from the caller's
  ``fallback`` rule. The fallback keeps each consumer's historical inference for
  datasets that do not state their base (e.g. FVCOM model output, which is
  one-based and carries no attribute).

Created 2026-11-05 for review round 6 #1 (zero-based connectivity was shifted
to negative indices by consumers that assumed one-based input).
"""

from __future__ import annotations

from typing import Any, Literal, Union

import numpy as np
from numpy.typing import NDArray

#: How to pick the indexing base when neither ``start_index`` nor the attribute
#: states it: a fixed base (0 or 1), ``"min_is_one"`` (one-based only if the
#: minimum is 1, else zero-based) or ``"min_is_zero"`` (zero-based only if the
#: minimum is 0, else one-based).
Fallback = Union[int, Literal["min_is_one", "min_is_zero"]]


def connectivity_three_by_nele(da: Any) -> NDArray[np.int_]:
    """Return connectivity as an integer ``(3, nele)`` array, validated first.

    Orientation comes from the dimension names (``three`` / ``nele``) when they
    are present; only unnamed inputs fall back to the shape, and a 3 x 3 array
    without names is refused as ambiguous. The values must be real, finite and
    integral before they are converted (a cast would truncate 0.9 to 0).
    """
    vals = np.asarray(getattr(da, "values", da))
    if vals.ndim != 2:
        raise ValueError(f"Connectivity (nv) must be 2-D (got shape {vals.shape})")
    if vals.dtype.kind not in "iuf":
        raise ValueError(f"Connectivity (nv) must be real numeric (got {vals.dtype})")
    if vals.dtype.kind == "f":
        if not np.isfinite(vals).all():
            raise ValueError("Connectivity (nv) contains non-finite values")
        if not np.array_equal(vals, np.round(vals)):
            raise ValueError("Connectivity (nv) contains non-integral values")
    dims = tuple(str(d) for d in getattr(da, "dims", ()))
    if "three" in dims or "nele" in dims:
        if "three" in dims:
            first = dims.index("three") == 0
        else:
            first = dims.index("nele") == 1
        if not first:
            vals = vals.T
    elif vals.shape[0] == 3 and vals.shape[1] != 3:
        pass
    elif vals.shape[1] == 3 and vals.shape[0] != 3:
        vals = vals.T  # (nele,3) -> (3,nele)
    elif vals.shape == (3, 3):
        raise ValueError(
            "Connectivity (nv) of shape (3, 3) without 'three'/'nele' dimension "
            "names is ambiguous"
        )
    if vals.shape[0] != 3:
        raise ValueError(
            f"Connectivity (nv) must have a dimension of length 3 (got {vals.shape})"
        )
    return np.asarray(vals, dtype=int)


def _check_base(base: Any, label: str) -> int:
    if isinstance(base, (bool, np.bool_)):
        raise ValueError(f"nv {label} must be 0 or 1 (got {base!r})")
    try:
        ok = base in (0, 1)
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(f"nv {label} must be 0 or 1 (got {base!r})")
    return int(base)


def connectivity_base(
    da: Any,
    vals: NDArray[np.int_],
    *,
    start_index: int | None = None,
    fallback: Fallback = "min_is_one",
) -> int:
    """Return the indexing base (0 or 1) of ``vals`` taken from ``da``.

    See the module docstring for the order of precedence.
    """
    if start_index is not None:
        return _check_base(start_index, "start_index")
    attr = getattr(da, "attrs", {}).get("start_index")
    if attr is not None:
        return _check_base(np.asarray(attr).reshape(-1)[0], "start_index attribute")
    if fallback == "min_is_one":
        return 1 if vals.size and vals.min() == 1 else 0
    if fallback == "min_is_zero":
        return 0 if vals.size and vals.min() == 0 else 1
    return _check_base(fallback, "fallback base")


def zero_based_connectivity(
    da: Any,
    *,
    start_index: int | None = None,
    fallback: Fallback = "min_is_one",
    n_nodes: int | None = None,
) -> NDArray[np.int_]:
    """Return ``nv`` as a validated, zero-based ``(3, nele)`` integer array.

    Parameters
    ----------
    da
        ``xarray.DataArray`` (dimension names and ``start_index`` attribute are
        honoured) or a plain array.
    start_index
        Explicit indexing base; overrides the attribute and the fallback.
    fallback
        Rule used only when neither ``start_index`` nor the attribute is given.
    n_nodes
        When given, every index must lie in ``[0, n_nodes)``; a negative index
        would otherwise silently wrap to the last node.
    """
    vals = connectivity_three_by_nele(da)
    base = connectivity_base(da, vals, start_index=start_index, fallback=fallback)
    out = vals - base
    if out.size and out.min() < 0:
        raise ValueError(
            f"Connectivity (nv) references a node outside the mesh "
            f"(an index below {base} in {base}-based connectivity)"
        )
    if n_nodes is not None and out.size and out.max() >= n_nodes:
        raise ValueError(
            f"Connectivity (nv) references a node outside the mesh ({n_nodes} nodes)"
        )
    return out
