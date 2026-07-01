#!/usr/bin/env python3
# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Drive :class:`xfvcom.io.JcopeObcGenerator` from the command line.

This is a thin runnable wrapper used to smoke-test the OBC generator
against real archive data and as a building block for project-specific
batch jobs. It loads:

* a FVCOM grid (``*_grd.dat``) for the geographic coordinates of every
  node and triangle,
* a FVCOM depth file (``*_dep.dat``) for the bathymetry at each node,
* a FVCOM OBC list (``*_obc.dat``) for the 1-based node IDs along the
  open boundary,
* the jcopetda archive (``basic.nc`` + ``jcopetda_region_YYYY.nc``),

and writes the TS-OBC and elevation NetCDFs alongside a short text
summary so the operator can sanity-check the result before driving FVCOM
with it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from xfvcom.grid.grid_obj import FvcomGrid
from xfvcom.io import JcopeGrid, JcopeObcGenerator, write_elevation_nc, write_tsobc_nc


def parse_year_spec(spec: str) -> list[int]:
    """Accept a single year, a comma list, or a 'YYYY-YYYY' inclusive range."""
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def format_year_label(years: list[int]) -> str:
    if len(years) == 1:
        return str(years[0])
    return f"{min(years)}-{max(years)}"


def parse_obc_dat(path: Path) -> np.ndarray:
    """Return the 1-based OBC node IDs from a FVCOM ``*_obc.dat`` file."""
    nodes: list[int] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("OBC Node Number"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                nodes.append(int(parts[1]))
            except ValueError:
                continue
    if not nodes:
        raise ValueError(f"no OBC nodes parsed from {path}")
    return np.asarray(nodes, dtype=np.int32)


def parse_dep_dat(path: Path) -> np.ndarray:
    """Return the bathymetry column from a FVCOM ``*_dep.dat`` file."""
    depths: list[float] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Node Number"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                depths.append(float(parts[2]))
            except ValueError:
                continue
    if not depths:
        raise ValueError(f"no depths parsed from {path}")
    return np.asarray(depths, dtype=np.float32)


# ---------------------------------------------------------------------------
# Raw-profile cache. The EXPENSIVE step is extracting the JCOPE T/S profiles at
# the OBC cells (reads the ~27 GB/year region archive). Those profiles depend
# only on the region archive + OBC-node (lat, lon) — NOT on the FVCOM depth /
# sigma — so they are identical for EVERY bathymetry variant of the same mesh +
# OBC list. Cache them once per year (~50 MB) and every later variant rebuild
# reads the cache + redoes only the cheap depth-dependent vertical interpolation.
# ---------------------------------------------------------------------------
def _profile_cache_path(cache_dir, year):
    return Path(cache_dir) / f"obc_raw_profiles_{year}.npz"


def _load_profile_cache(cache_dir, year, region_nc, obc_ids):
    """Return the cached raw profiles dict if a valid cache exists, else None.

    Invalidated when the OBC node list differs or the region archive's byte size
    changed (i.e. the JCOPE data was regenerated).
    """
    if not cache_dir:
        return None
    p = _profile_cache_path(cache_dir, year)
    if not p.exists():
        return None
    try:
        d = np.load(p)
    except Exception:
        return None
    if not np.array_equal(
        np.asarray(d["obc_nodes"]), np.asarray(obc_ids, dtype=np.int32)
    ):
        return None
    if int(d["region_size"]) != int(Path(region_nc).stat().st_size):
        return None  # region archive changed -> cache stale
    return {k[4:]: d[k] for k in d.files if k.startswith("var_")}


def _save_profile_cache(cache_dir, year, region_nc, obc_ids, raw):
    if not cache_dir:
        return
    p = _profile_cache_path(cache_dir, year)
    p.parent.mkdir(parents=True, exist_ok=True)
    st = Path(region_nc).stat()
    tmp = p.parent / (p.name + ".tmp")
    with open(tmp, "wb") as fh:  # pass a handle so np.savez does NOT append '.npz'
        np.savez(
            fh,
            obc_nodes=np.asarray(obc_ids, dtype=np.int32),
            region_size=np.int64(st.st_size),
            region_mtime=np.float64(st.st_mtime),
            **{f"var_{k}": v for k, v in raw.items()},
        )
    tmp.replace(p)  # atomic swap (each year writes its own file)


def _process_year(
    year,
    region_nc,
    basic_nc,
    obc_ids,
    obc_lat,
    obc_lon,
    obc_h_fvcom,
    n_siglay,
    siglay_per_node,
    siglev_per_node,
    no_tsobc,
    no_elevation,
    profile_cache_dir=None,
):
    """Interpolate one year's OBC — the per-year loop body, extracted so it can
    run as a joblib worker (``--jobs>1``) and share the profile-cache path. Each
    worker opens its own JcopeGrid + region archive, so years are independent;
    all args are picklable (numpy arrays / Path / scalars). Returns a plain dict
    keyed for reassembly in the parent. With ``profile_cache_dir`` the raw JCOPE
    profiles are loaded from / saved to the cache (skipping the 27 GB read on a
    hit); the output is identical either way.
    """
    grid = JcopeGrid(basic_nc)
    gen = JcopeObcGenerator(
        grid=grid,
        region_nc=region_nc,
        obc_nodes=obc_ids,
        obc_lat=obc_lat,
        obc_lon=obc_lon,
        obc_h_fvcom=obc_h_fvcom,
        n_siglay=n_siglay,
        siglay_per_node=siglay_per_node,
        siglev_per_node=siglev_per_node,
    )
    tsobc = None
    if not no_tsobc:
        if profile_cache_dir:
            raw = _load_profile_cache(profile_cache_dir, year, region_nc, obc_ids)
            hit = raw is not None
            if raw is None:
                raw = gen.read_raw_profiles()
                _save_profile_cache(profile_cache_dir, year, region_nc, obc_ids, raw)
            print(
                f"[cache] year {year}: profile cache {'HIT' if hit else 'MISS (built)'}",
                flush=True,
            )
            tsobc = gen.build_tsobc_arrays(raw_profiles=raw)
        else:
            tsobc = gen.build_tsobc_arrays()
    out = {
        "year": year,
        "mjd": gen.mjd.copy(),
        "siglev": gen.siglev,
        "siglay": gen.siglay,
        "obc_h_jcope": gen.obc_h_jcope.copy(),
        "tsobc": tsobc,
        "elevation": None if no_elevation else gen.build_elevation_array(),
    }
    gen.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate FVCOM OBC NetCDFs from a JCOPE-T DA archive via "
            "xfvcom.io.JcopeObcGenerator."
        ),
    )
    parser.add_argument(
        "--basic-nc",
        type=Path,
        default=Path(os.environ.get("DATA_DIR", "/var/empty"))
        / "jcopetda/grid/basic.nc",
        help="path to jcopetda basic.nc (default: $DATA_DIR/jcopetda/grid/basic.nc)",
    )
    parser.add_argument(
        "--region-nc",
        type=Path,
        default=None,
        help="path to jcopetda_region_YYYY.nc time-series archive "
        "(single-year mode; mutually exclusive with --region-pattern)",
    )
    parser.add_argument(
        "--region-pattern",
        type=str,
        default=None,
        help="path template containing {year}, e.g. "
        "'/data/jcopetda/region/.../jcopetda_region_{year}.nc' "
        "(required when --years selects more than one year)",
    )
    parser.add_argument(
        "--fvcom-grid", type=Path, required=True, help="path to FVCOM *_grd.dat"
    )
    parser.add_argument(
        "--fvcom-dep", type=Path, required=True, help="path to FVCOM *_dep.dat"
    )
    parser.add_argument(
        "--fvcom-obc",
        type=Path,
        required=True,
        help="path to FVCOM *_obc.dat (node ID list)",
    )
    parser.add_argument(
        "--utm-zone",
        type=int,
        default=54,
        help="UTM zone for the grd file (default 54 for Tokyo Bay)",
    )
    parser.add_argument(
        "--n-siglay",
        type=int,
        default=30,
        help="number of FVCOM sigma layers (uniform, default 30)",
    )
    parser.add_argument(
        "--sigma-dat",
        type=Path,
        default=None,
        help="SADAPT / sigma-z sigma.dat. When given, build a PER-NODE OBC "
        "vertical coordinate from it (overrides --n-siglay) so the OBC lands on "
        "the model's actual sigma-z layer depths. Requires --fvcom-grid + "
        "--fvcom-dep to match the grid the sigma.dat was built for.",
    )
    parser.add_argument("--tag", default="jcope", help="filename tag (default: jcope)")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel workers over years (joblib; default 1 = serial, "
        "byte-identical to the original path). Each worker reads one year's "
        "region NC independently; the deterministic time concatenation is "
        "unchanged, so N>1 only speeds up a multi-year build.",
    )
    parser.add_argument(
        "--profile-cache-dir",
        type=Path,
        default=None,
        help="directory for the raw-profile cache (obc_raw_profiles_<year>.npz). "
        "The JCOPE T/S profiles at the OBC cells are bathymetry-INDEPENDENT "
        "(fixed OBC node lat/lon), so caching them once lets every later "
        "depth-variant OBC rebuild skip the ~27 GB/year region read and redo "
        "only the cheap vertical interpolation. Cache is invalidated if the OBC "
        "node list or the region archive's byte size changes. Omit to disable.",
    )
    year_group = parser.add_mutually_exclusive_group(required=True)
    year_group.add_argument(
        "--year", type=int, help="single year; use --region-nc for the archive"
    )
    year_group.add_argument(
        "--years",
        type=str,
        help="year list ('2020,2022') or inclusive range ('2020-2023'); "
        "requires --region-pattern",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory to write tsobc / elevation NetCDFs to",
    )
    parser.add_argument(
        "--no-tsobc", action="store_true", help="skip writing the T/S OBC file"
    )
    parser.add_argument(
        "--no-elevation", action="store_true", help="skip writing the SSH OBC file"
    )
    args = parser.parse_args()

    # ---- resolve year list and region-NC source(s) ----
    if args.year is not None:
        years = [args.year]
    else:
        years = parse_year_spec(args.years)
    if not years:
        sys.exit("no years to process")

    if len(years) == 1 and args.region_nc is not None:
        region_paths = [args.region_nc]
    elif args.region_pattern is not None:
        region_paths = [Path(args.region_pattern.format(year=y)) for y in years]
    elif len(years) == 1:
        sys.exit("single-year run needs either --region-nc or --region-pattern")
    else:
        sys.exit("multi-year run requires --region-pattern")

    missing = [p for p in region_paths if not p.exists()]
    if missing:
        sys.exit("missing region NCs: " + ", ".join(str(p) for p in missing))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- load FVCOM-side metadata once ----
    print(f"[grid] reading FVCOM mesh from {args.fvcom_grid}")
    mesh = FvcomGrid.from_dat(args.fvcom_grid, utm_zone=args.utm_zone)
    if mesh.lat is None or mesh.lon is None:
        sys.exit("FvcomGrid lacks lon/lat — check utm_zone and the *_grd.dat file")
    n_nodes = mesh.node
    print(f"  nodes: {n_nodes}, elements: {mesh.nele}")

    print(f"[grid] reading FVCOM depth from {args.fvcom_dep}")
    h_all = parse_dep_dat(args.fvcom_dep)
    if h_all.size != n_nodes:
        sys.exit(
            f"depth file has {h_all.size} rows; grid has {n_nodes} nodes — mismatch"
        )

    print(f"[grid] reading OBC node list from {args.fvcom_obc}")
    obc_ids = parse_obc_dat(args.fvcom_obc)
    print(f"  OBC nodes ({obc_ids.size}): {obc_ids.tolist()}")

    idx0: NDArray[np.int64] = (obc_ids - 1).astype(np.int64)
    obc_lat = np.asarray(mesh.lat)[idx0]
    obc_lon = np.asarray(mesh.lon)[idx0]
    obc_h_fvcom = h_all[idx0]

    # ---- optional SADAPT / sigma-z per-node vertical coordinate ----
    # Without --sigma-dat the OBC uses the uniform --n-siglay grid (default).
    # With it, build the FVCOM sigma-z coordinate from the SADAPT sigma.dat and
    # take each OBC node's actual layer depths (coord.Z) so the OBC values land on
    # the model's per-node sigma-z column (FVCOM does NO vertical re-interpolation
    # of the OBC file). The grid/dep MUST be the ones the sigma.dat was built for.
    siglay_per_node = None
    siglev_per_node = None
    if args.sigma_dat is not None:
        from xfvcom.grid.gtsz_builder import build_gtsz, load_mesh
        from xfvcom.io import read_sigma_dat

        sf = read_sigma_dat(args.sigma_dat)
        if sf.gtsz is None:
            sys.exit(
                f"--sigma-dat {args.sigma_dat} is not a SIGMAZ file "
                f"(SIGMA COORDINATE TYPE = {sf.stype}); use --n-siglay for uniform σ"
            )
        gmesh = load_mesh(args.fvcom_grid, args.fvcom_dep)
        if gmesh.h.shape[0] != n_nodes:
            sys.exit(f"--sigma-dat mesh has {gmesh.h.shape[0]} nodes != grid {n_nodes}")
        coord = build_gtsz(gmesh, sf.gtsz)
        z_obc = np.asarray(coord.Z, dtype=np.float64)[idx0]  # (n_obc, KB) siglev
        siglev_per_node = z_obc.T  # (KB, n_obc)
        siglay_per_node = 0.5 * (
            siglev_per_node[:-1] + siglev_per_node[1:]
        )  # (KBM1, n_obc)
        print(
            f"[sigma-z] SADAPT per-node OBC coordinate from {args.sigma_dat.name}: "
            f"KB={sf.gtsz.kb} KBM1={siglay_per_node.shape[0]} "
            f"(uniform --n-siglay={args.n_siglay} IGNORED); "
            f"KBP range {int(coord.kbp[idx0].min())}-{int(coord.kbp[idx0].max())}"
        )

    print(f"[jcope] opening basic.nc at {args.basic_nc}")
    grid = JcopeGrid(args.basic_nc)
    print(
        f"  grid: {grid.im} x {grid.jm} x {grid.km}, "
        f"ocean fraction {grid.mask.mean()*100:.1f}%"
    )

    # ---- loop over years, accumulating arrays ----
    tsobc_fields_by_year: dict[int, dict[str, np.ndarray]] = {}
    elevation_by_year: dict[int, np.ndarray] = {}
    mjd_by_year: dict[int, np.ndarray] = {}
    siglev: NDArray[np.float64] | None = None
    siglay: NDArray[np.float64] | None = None
    obc_h_jcope_first: NDArray[np.float32] | None = None

    # Years are independent — process each via _process_year (raw-profile read /
    # cache + vertical interp). With --jobs>1 they run concurrently (joblib),
    # otherwise serially; the deterministic per-year concatenation below is
    # order-independent, so the output is identical either way.
    def _args_for(year, region_nc):
        return (
            year,
            region_nc,
            args.basic_nc,
            obc_ids,
            obc_lat,
            obc_lon,
            obc_h_fvcom,
            args.n_siglay,
            siglay_per_node,
            siglev_per_node,
            args.no_tsobc,
            args.no_elevation,
            args.profile_cache_dir,
        )

    if args.jobs and args.jobs > 1 and len(years) > 1:
        from joblib import Parallel, delayed

        print(
            f"[jcope] parallel over {len(years)} years with n_jobs={args.jobs} (joblib) ...",
            flush=True,
        )
        results = Parallel(n_jobs=args.jobs)(
            delayed(_process_year)(*_args_for(year, region_nc))
            for year, region_nc in zip(years, region_paths)
        )
    else:
        results = [
            _process_year(*_args_for(year, region_nc))
            for year, region_nc in zip(years, region_paths)
        ]
    for r in results:
        y = r["year"]
        mjd_by_year[y] = r["mjd"]
        if not args.no_tsobc:
            tsobc_fields_by_year[y] = r["tsobc"]
        if not args.no_elevation:
            elevation_by_year[y] = r["elevation"]
    siglev = results[0]["siglev"]
    siglay = results[0]["siglay"]
    obc_h_jcope_first = results[0]["obc_h_jcope"]

    # All three must be set after the loop runs at least once (years is
    # guaranteed non-empty above).
    assert siglev is not None
    assert siglay is not None
    assert obc_h_jcope_first is not None

    # ---- summary table (FVCOM h vs JCOPE h, computed once) ----
    print()
    print("OBC summary (node → FVCOM h vs JCOPE h):")
    print("  node  lat       lon        h_fvcom  h_jcope")
    for n, nid in enumerate(obc_ids):
        print(
            f"  {int(nid):4d}  {obc_lat[n]:8.4f}  {obc_lon[n]:8.4f}  "
            f"{float(obc_h_fvcom[n]):7.2f}  {float(obc_h_jcope_first[n]):7.2f}"
        )
    print()

    label = format_year_label(years)
    source_files = [Path(p).name for p in region_paths]

    if not args.no_tsobc:
        time_mjd = np.concatenate([mjd_by_year[y] for y in years])
        fields_concat = {
            v: np.concatenate([tsobc_fields_by_year[y][v] for y in years], axis=0)
            for v in tsobc_fields_by_year[years[0]]
        }
        out = args.output_dir / f"tb_tsobc_{args.tag}_{label}.nc"
        print(f"[write] {out}  (n_time={time_mjd.size})")
        write_tsobc_nc(
            out,
            time_mjd=time_mjd,
            obc_nodes=obc_ids,
            obc_h=obc_h_fvcom,
            siglev=siglev,
            siglay=siglay,
            fields=fields_concat,
            source_files=source_files,
        )

    if not args.no_elevation:
        time_mjd = np.concatenate([mjd_by_year[y] for y in years])
        elevation_concat = np.concatenate([elevation_by_year[y] for y in years], axis=0)
        out = args.output_dir / f"tb_julian_obc_{args.tag}_{label}.nc"
        print(f"[write] {out}  (n_time={time_mjd.size})")
        write_elevation_nc(
            out,
            time_mjd=time_mjd,
            obc_nodes=obc_ids,
            elevation=elevation_concat,
            source_files=source_files,
        )

    grid.close()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
