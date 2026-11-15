"""Grid-dependent FVCOM inputs carried to a refined mesh by node renumbering.

Only the node ids change; every value, attribute and dimension is kept. The
functions refuse a source node that the map removed: a boundary or source that
is not frozen needs a decision by the caller, not a silent move.

* ``reindex_river_namelist`` -- ``RIVER_GRID_LOCATION`` of every ``&NML_RIVER``.
* ``reindex_obc_netcdf`` -- ``obc_nodes`` of an open-boundary forcing file
  (elevation, T/S, ...), every other variable copied unchanged.
"""

import os
import re

import numpy as np

from .mesh_pair import _integers, reindex_one_based
from .transfer import apply_2d

_LOCATION = re.compile(r"(?im)^([ \t]*RIVER_GRID_LOCATION[ \t]*=[ \t]*)([^,/\n!]*)")
_BLOCK = re.compile(r"(?im)^[ \t]*&NML_RIVER\b")


def _strip_comments(text):
    """The text with Fortran ``!`` comments blanked (outside quoted strings).

    Blanking keeps every offset, so matches in the result index the original.
    """
    out = []
    for line in text.splitlines(keepends=True):
        newline = "\n" if line.endswith("\n") else ""
        body = line[: len(line) - len(newline)]
        quote, cut = None, len(body)
        for k, ch in enumerate(body):
            if quote:
                if ch == quote:
                    quote = None
            elif ch in "'\"":
                quote = ch
            elif ch == "!":
                cut = k
                break
        out.append(body[:cut] + " " * (len(body) - cut) + newline)
    return "".join(out)


def reindex_river_namelist(text, node_map):
    """Return (new text, [(old id, new id), ...]) for an FVCOM rivers namelist.

    Each ``&NML_RIVER`` block must carry exactly one ``RIVER_GRID_LOCATION``
    written as a plain positive integer; anything else (an expression, a
    repeat count, a missing or duplicated key) is refused rather than guessed.
    Comments and every other character are kept.
    """
    if not isinstance(text, str):
        raise ValueError("rivers namelist: expected text")
    code = _strip_comments(text)
    if len(code) != len(text):
        raise ValueError("rivers namelist: comment stripping changed the length")
    blocks = [m.start() for m in _BLOCK.finditer(code)]
    locations = list(_LOCATION.finditer(code))
    if not blocks:
        raise ValueError("rivers namelist: no &NML_RIVER block")
    if len(locations) != len(blocks):
        raise ValueError(
            f"rivers namelist: {len(blocks)} &NML_RIVER blocks but "
            f"{len(locations)} RIVER_GRID_LOCATION entries"
        )
    edges = blocks[1:] + [len(code)]
    for start, end, match in zip(blocks, edges, locations):
        if not start < match.start() < end:
            raise ValueError(
                "rivers namelist: a block without its own RIVER_GRID_LOCATION"
            )
    old = []
    for match in locations:
        token = match.group(2).strip()
        if not re.fullmatch(r"\+?[0-9]+", token):
            raise ValueError(
                f"rivers namelist: RIVER_GRID_LOCATION {token!r} is not a plain integer"
            )
        old.append(int(token))
    new = reindex_one_based(np.array(old, dtype=np.int64), node_map)
    pieces, last = [], 0
    for match, value in zip(locations, new):
        start, end = match.span(2)
        token = match.group(2)
        width = len(token.rstrip())
        pieces += [text[last:start], str(int(value)), token[width:]]
        last = end
    pieces.append(text[last:])
    return "".join(pieces), [(int(a), int(b)) for a, b in zip(old, new)]


def _copy_variable(src, dst, name, data=None):
    var = src.variables[name]
    out = _create_like(src, dst, name)
    var.set_auto_maskandscale(False)
    out.set_auto_maskandscale(False)
    if data is not None:
        out[...] = data
    elif var.dimensions and src.dimensions[var.dimensions[0]].isunlimited():
        n = len(src.dimensions[var.dimensions[0]])
        for k in range(0, n, 4096):
            chunk = var[k : k + 4096]
            out[k : k + len(chunk)] = chunk
    else:
        out[...] = var[...]


def reindex_obc_netcdf(source, destination, node_map, *, variable="obc_nodes"):
    """Write ``destination``: ``source`` with its open-boundary node ids renumbered.

    Every dimension, attribute and other variable (dtype, chunking, filters,
    values) is copied unchanged and compared after writing. The destination
    must not exist; it is written beside itself under a temporary name and
    linked into place only after the comparison (never overwriting).
    """
    import netCDF4

    destination = os.fspath(destination)
    if os.path.lexists(destination):
        raise ValueError(f"{destination}: exists; refusing to overwrite")
    part = f"{destination}.part{os.getpid()}"
    with netCDF4.Dataset(source) as src:
        if variable not in src.variables:
            raise ValueError(f"{source}: no {variable} variable")
        ids = src.variables[variable]
        ids.set_auto_maskandscale(False)
        old = _integers(np.asarray(ids[...]), variable)
        if old.ndim != 1:
            raise ValueError(f"{source}: {variable} must be one-dimensional")
        new = reindex_one_based(old, node_map).astype(ids.dtype)
        if not np.array_equal(new.astype(np.int64), reindex_one_based(old, node_map)):
            raise ValueError(f"{variable}: renumbered ids do not fit {ids.dtype}")
        try:
            with netCDF4.Dataset(part, "w", format=src.data_model) as dst:
                dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})
                for name, dim in src.dimensions.items():
                    dst.createDimension(name, None if dim.isunlimited() else len(dim))
                for name in src.variables:
                    _copy_variable(src, dst, name, new if name == variable else None)
            _compare(source, part, variable, new)
            os.link(part, destination)
        finally:
            if os.path.lexists(part):
                os.unlink(part)
    return [(int(a), int(b)) for a, b in zip(old, new)]


def _compare(source, written, variable, ids):
    import netCDF4

    with netCDF4.Dataset(source) as a, netCDF4.Dataset(written) as b:
        if a.data_model != b.data_model:
            raise ValueError("copy: file format differs")
        if {k: a.getncattr(k) for k in a.ncattrs()}.keys() != set(b.ncattrs()):
            raise ValueError("copy: global attributes differ")
        for k in a.ncattrs():
            if not np.array_equal(
                np.asarray(a.getncattr(k)), np.asarray(b.getncattr(k))
            ):
                raise ValueError(f"copy: global attribute {k} differs")
        if [(n, len(d), d.isunlimited()) for n, d in a.dimensions.items()] != [
            (n, len(d), d.isunlimited()) for n, d in b.dimensions.items()
        ]:
            raise ValueError("copy: dimensions differ")
        if list(a.variables) != list(b.variables):
            raise ValueError("copy: variables differ")
        for name, va in a.variables.items():
            vb = b.variables[name]
            va.set_auto_maskandscale(False)
            vb.set_auto_maskandscale(False)
            if va.dtype != vb.dtype or va.dimensions != vb.dimensions:
                raise ValueError(f"copy: {name} type or dimensions differ")
            if va.ncattrs() != vb.ncattrs() or any(
                not np.array_equal(
                    np.asarray(va.getncattr(k)), np.asarray(vb.getncattr(k))
                )
                for k in va.ncattrs()
            ):
                raise ValueError(f"copy: {name} attributes differ")
            expected = ids if name == variable else None
            n = len(va) if va.dimensions else 0
            if expected is not None:
                if not np.array_equal(np.asarray(vb[...]), expected):
                    raise ValueError(f"copy: {name} not written as renumbered")
                continue
            if not va.dimensions or n <= 4096:
                same = np.asarray(va[...]).tobytes() == np.asarray(vb[...]).tobytes()
            else:
                same = all(
                    np.asarray(va[k : k + 4096]).tobytes()
                    == np.asarray(vb[k : k + 4096]).tobytes()
                    for k in range(0, n, 4096)
                )
            if not same:
                raise ValueError(f"copy: {name} values differ")


TIME_VARIABLES = ("time", "Itime", "Itime2", "Times")


def _window(time, start, end):
    """Record indices [i0, i1] bracketing [start, end] (both inclusive)."""
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or len(time) == 0 or not np.isfinite(time).all():
        raise ValueError("forcing: time must be a finite non-empty vector")
    if not (np.diff(time) > 0).all():
        raise ValueError("forcing: time must increase strictly")
    if not (np.isfinite(start) and np.isfinite(end)) or start > end:
        raise ValueError("forcing: window start must not exceed its end")
    if start < time[0] or end > time[-1]:
        raise ValueError(
            f"forcing: window [{start}, {end}] is not bracketed by the file "
            f"[{time[0]}, {time[-1]}]"
        )
    i0 = int(np.searchsorted(time, start, side="right") - 1)
    i1 = int(np.searchsorted(time, end, side="left"))
    return i0, i1


def transfer_forcing_netcdf(
    source, destination, op, coordinates, *, start, end, block=240
):
    """Write ``destination``: an FVCOM grid forcing file carried to the target mesh.

    ``op`` is a ``Transfer`` from ``build_transfer``. Every ``(time, node)`` and
    ``(time, nele)`` variable is transferred record by record with
    ``apply_2d`` (frozen nodes and matched elements copy their bytes); the time
    variables (``time``, ``Itime``, ``Itime2``, ``Times``) are cut to the
    records bracketing ``[start, end]`` (the ``time`` variable's units); every
    variable on ``node``, ``nele`` or ``three`` alone (coordinates, ``nv``)
    must be given in ``coordinates`` as the target's values. Any other
    variable or dimension layout is refused. Attributes, dtypes and the file
    format are kept; chunking follows the source with the target sizes.
    The destination must not exist and is linked into place only when complete.
    """
    import netCDF4

    destination = os.fspath(destination)
    if os.path.lexists(destination):
        raise ValueError(f"{destination}: exists; refusing to overwrite")
    sizes = {
        "node": op.node_matrix.shape[0],
        "nele": op.element_matrix.shape[0],
        "three": 3,
    }
    base_sizes = {
        "node": op.node_matrix.shape[1],
        "nele": op.element_matrix.shape[1],
        "three": 3,
    }
    part = f"{destination}.part{os.getpid()}"
    with netCDF4.Dataset(source) as src:
        for name, dim in src.dimensions.items():
            if name in base_sizes and len(dim) != base_sizes[name]:
                raise ValueError(
                    f"{source}: dimension {name} = {len(dim)}, the base mesh has "
                    f"{base_sizes[name]}"
                )
        if "time" not in src.variables or src.variables["time"].dimensions != ("time",):
            raise ValueError(f"{source}: no time(time) variable")
        i0, i1 = _window(src.variables["time"][:], start, end)
        plan = {}
        for name, var in src.variables.items():
            dims = var.dimensions
            if name in TIME_VARIABLES and dims and dims[0] == "time":
                plan[name] = "cut"
            elif dims in (("time", "node"), ("time", "nele")):
                plan[name] = "node" if dims[1] == "node" else "element"
            elif dims and set(dims) <= set(sizes):
                if name not in coordinates:
                    raise ValueError(f"{name}{dims}: target values not given")
                value = np.asarray(coordinates[name])
                if value.shape != tuple(sizes[d] for d in dims):
                    raise ValueError(f"{name}: target values have the wrong shape")
                plan[name] = "given"
            else:
                raise ValueError(f"{name}{dims}: no rule to carry it to the target")
        unused = set(coordinates) - {n for n, p in plan.items() if p == "given"}
        if unused:
            raise ValueError(f"coordinates not in the source: {sorted(unused)}")
        try:
            with netCDF4.Dataset(part, "w", format=src.data_model) as dst:
                dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})
                for name, dim in src.dimensions.items():
                    if dim.isunlimited():
                        dst.createDimension(name, None)
                    elif name == "time":
                        dst.createDimension(name, i1 - i0 + 1)
                    else:
                        dst.createDimension(name, sizes.get(name, len(dim)))
                for name, var in src.variables.items():
                    out = _create_like(src, dst, name)
                    var.set_auto_maskandscale(False)
                    out.set_auto_maskandscale(False)
                    if plan[name] == "given":
                        out[...] = np.asarray(coordinates[name]).astype(var.dtype)
                    elif plan[name] == "cut":
                        out[...] = var[i0 : i1 + 1]
                    else:
                        for k in range(i0, i1 + 1, block):
                            stop = min(k + block, i1 + 1)
                            values = np.asarray(var[k:stop]).T
                            out[k - i0 : stop - i0] = apply_2d(
                                op, values, location=plan[name]
                            ).T
            os.link(part, destination)
        finally:
            if os.path.lexists(part):
                os.unlink(part)
    return {"records": (i0, i1), "variables": plan}


def _create_like(src, dst, name):
    var = src.variables[name]
    kwargs = {}
    filters = var.filters() or {}
    if filters.get("zlib"):
        kwargs.update(zlib=True, complevel=filters.get("complevel", 4))
    if filters.get("shuffle"):
        kwargs["shuffle"] = True
    if filters.get("fletcher32"):
        kwargs["fletcher32"] = True
    chunking = var.chunking()
    if chunking == "contiguous":
        kwargs["contiguous"] = True
    elif chunking is not None:
        # the time chunk is kept; a spatial chunk spanning the whole base
        # dimension spans the target's, any other is capped by it
        kwargs["chunksizes"] = [
            (
                c
                if dst.dimensions[d].isunlimited() or d == "time"
                else (
                    len(dst.dimensions[d])
                    if c == len(src.dimensions[d])
                    else min(c, len(dst.dimensions[d]))
                )
            )
            for c, d in zip(chunking, var.dimensions)
        ]
    kwargs["fill_value"] = (
        var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else False
    )
    out = dst.createVariable(name, var.datatype, var.dimensions, **kwargs)
    out.setncatts({a: var.getncattr(a) for a in var.ncattrs() if a != "_FillValue"})
    return out
