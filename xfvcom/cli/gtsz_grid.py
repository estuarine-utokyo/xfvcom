"""``xfvcom-gtsz-grid`` -- generate an FVCOM sigma-z (GTSZ) ``*_sigma.dat`` from a
mesh (``*_grd.dat`` + ``*_dep.dat``) and a set of coordinate knobs, with a
diagnostics report and optional figures.

This is the *general* (domain-agnostic) entry point. Tokyo-Bay-specific target
selection lives in the application (``TB-FVCOM/hydro/tuning/sigmaz/``), which
imports :mod:`xfvcom.grid`.

Example::

    xfvcom-gtsz-grid --single \\
        --grd TokyoBay_grd.dat --dep TokyoBay_dep_rfac0p2.dat \\
        --out tb_sigmaz_sadapt_sigma.dat \\
        --sadapt --k1 12 --smax 0.02 --zlev-auto --auto-kb \\
        --report --figdir figures/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="xfvcom-gtsz-grid",
        description="Generate an FVCOM sigma-z (GTSZ / SIGMAZ) *_sigma.dat from a mesh.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--single", action="store_true", help="generate a single grid (default mode)"
    )
    # --- mesh ---
    p.add_argument("--grd", type=Path, required=True, help="FVCOM *_grd.dat (mesh)")
    p.add_argument(
        "--dep", type=Path, required=True, help="FVCOM *_dep.dat (bathymetry)"
    )
    p.add_argument("--out", type=Path, required=True, help="output *_sigma.dat path")
    # --- coordinate family ---
    p.add_argument(
        "--sadapt",
        action="store_true",
        help="slope-adaptive reference-surface coordinate (requires --mask; default on)",
    )
    p.add_argument("--no-sadapt", dest="sadapt", action="store_false")
    p.set_defaults(sadapt=True)
    p.add_argument(
        "--mask",
        action="store_true",
        help="z-band masking (variable active layers + lateral wall); "
        "REQUIRED by --sadapt; default on",
    )
    p.add_argument("--no-mask", dest="mask", action="store_false")
    p.set_defaults(mask=True)
    p.add_argument(
        "--base",
        type=int,
        default=2,
        choices=(1, 2, 3),
        help="surface-band base stretch: 1=A1 uniform, 2=A2 double-exp, 3=A3 tanh",
    )
    p.add_argument("--p1", type=float, default=2.0, help="A2 exponent GTSZ P1")
    p.add_argument("--l1", type=float, default=1.0, help="A3 GTSZ L1")
    p.add_argument("--l2", type=float, default=1.0, help="A3 GTSZ L2")
    p.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="non-SADAPT z<->sigma blend width [m] (0=hard max)",
    )
    # --- knobs ---
    p.add_argument("--k1", type=int, default=12, help="surface sigma-band layer count")
    p.add_argument(
        "--smax",
        type=float,
        default=0.02,
        help="max sigma-slope for H_ref [m/m] (SADAPT); <=0 => flat min(H)",
    )
    p.add_argument(
        "--kb",
        type=int,
        default=None,
        help="total levels KB; omit with --auto-kb to size from K1+#zlev+1",
    )
    p.add_argument(
        "--auto-kb",
        action="store_true",
        help="size KB = K1 + max active z-levels + 1 (the deepest SADAPT column)",
    )
    # --- z-levels ---
    p.add_argument(
        "--zlev",
        type=str,
        default=None,
        help="explicit z-levels, space-separated negatives (descending), "
        'e.g. "-2 -4 -6 ... -40"',
    )
    p.add_argument(
        "--zlev-auto",
        action="store_true",
        help="auto geometrically-stretched z-levels (see --zlev-*)",
    )
    p.add_argument(
        "--zlev-top", type=float, default=-2.0, help="shallowest z-level [m]"
    )
    p.add_argument("--zlev-dz0", type=float, default=2.0, help="top interval [m]")
    p.add_argument("--zlev-stretch", type=float, default=1.12, help="growth ratio")
    p.add_argument("--zlev-dzmax", type=float, default=60.0, help="max interval [m]")
    p.add_argument(
        "--zlev-hmax",
        type=float,
        default=None,
        help="span z-levels to this depth [m] (default: mesh max depth)",
    )
    # --- diagnostics / output ---
    p.add_argument(
        "--report", action="store_true", help="print + save a diagnostics report"
    )
    p.add_argument(
        "--pge",
        action="store_true",
        help="also compute the offline sigma-PGE pre-screen (slower)",
    )
    p.add_argument(
        "--figdir", type=Path, default=None, help="write transect + map figures here"
    )
    p.add_argument(
        "--comment", type=str, default=None, help="header comment for the sigma.dat"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    from xfvcom.grid.gtsz import GtszSpec
    from xfvcom.grid.gtsz_builder import (
        active_zlev_count,
        auto_kb,
        build_gtsz,
        grassfire_href,
        load_mesh,
        suggest_zlev,
    )
    from xfvcom.grid.gtsz_diagnostics import coordinate_summary, offline_sigma_pge
    from xfvcom.io.sigma_dat import write_sigma_dat

    args = _build_parser().parse_args(argv)

    mesh = load_mesh(args.grd, args.dep)
    print(
        f"mesh: {mesh.n_node} nodes, {mesh.n_elem} elements; "
        f"H {mesh.h.min():.1f} .. {mesh.h.max():.1f} m"
    )

    # --- z-levels ---
    if args.zlev:
        zlev = np.asarray(
            [float(t) for t in args.zlev.replace(",", " ").split()], float
        )
    elif args.zlev_auto:
        hmax = args.zlev_hmax if args.zlev_hmax is not None else float(mesh.h.max())
        zlev = suggest_zlev(
            hmax,
            z_top=args.zlev_top,
            dz_shallow=args.zlev_dz0,
            stretch=args.zlev_stretch,
            dz_max=args.zlev_dzmax,
        )
        print(f"auto z-levels: {len(zlev)} from {zlev[0]:.0f} to {zlev[-1]:.0f} m")
    else:
        print("ERROR: provide --zlev or --zlev-auto", file=sys.stderr)
        return 2

    # --- KB (auto-sizes from the deepest SADAPT column) ---
    if args.auto_kb or args.kb is None:
        if args.sadapt:
            href = grassfire_href(mesh.h, mesh.nv, mesh.x, mesh.y, args.smax)
            kb = auto_kb(mesh.h, href, args.k1, zlev)
        else:
            kb = args.k1 + len(zlev) + 1
        print(f"auto KB = {kb} (K1={args.k1} + #zlev/active + 1)")
    else:
        kb = args.kb

    spec = GtszSpec(
        kb=kb,
        base=args.base,
        k1=args.k1,
        k2=kb,
        nz=len(zlev),
        zlev=tuple(zlev),
        p1=args.p1,
        l1=args.l1,
        l2=args.l2,
        smooth=args.smooth,
        mask=args.mask,
        sadapt=args.sadapt,
        smax=args.smax,
    )
    spec.validate()

    coord = build_gtsz(mesh, spec)
    write_sigma_dat(args.out, spec, header_comment=args.comment)
    print(f"wrote {args.out}")

    if args.report or args.figdir or args.pge:
        summ = coordinate_summary(coord)
        report = summ.as_text()
        if args.pge:
            pge = offline_sigma_pge(coord)
            report += "\n" + pge.as_text()
        print("---- diagnostics ----")
        print(report)
        rpath = args.out.with_suffix(".report.txt")
        rpath.write_text(report + "\n")
        print(f"wrote {rpath}")

    if args.figdir:
        import matplotlib

        from xfvcom.grid.gtsz_diagnostics import plot_maps, plot_transect

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        args.figdir.mkdir(parents=True, exist_ok=True)
        ax = plot_transect(
            coord,
            along="x",
            title=f"sigma-z transect (SADAPT={spec.sadapt}, KB={kb}, K1={spec.k1})",
        )
        ax.figure.savefig(
            args.figdir / "gtsz_transect.png", dpi=150, bbox_inches="tight"
        )
        plt.close(ax.figure)
        fig = plot_maps(coord)
        fig.savefig(args.figdir / "gtsz_maps.png", dpi=150)
        plt.close(fig)
        print(f"wrote figures to {args.figdir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
