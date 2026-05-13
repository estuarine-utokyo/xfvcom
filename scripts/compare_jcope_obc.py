#!/usr/bin/env python3
# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Compare two FVCOM TS-OBC NetCDFs node by node.

Used to inspect how the new ``xfvcom.io.JcopeObcGenerator`` pipeline
(real JCOPE bathymetry + real σ from ``basic.nc``) differs from the
legacy ``create_obc_from_jcope.py`` output (J-EGG500-averaged depth
+ uniform σ assumption) at the open-boundary nodes.

Both inputs must share dimensions ``(time, siglay, nobc)`` and the same
node order along ``nobc``. The script slices both to a single calendar
year, plots annual-mean T/S profiles per node, and prints a summary
table with per-node RMSE / bias (target z = the new file's column).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd


def mjd_year_bounds(year: int) -> tuple[float, float]:
    epoch = pd.Timestamp("1858-11-17")
    start = (pd.Timestamp(f"{year}-01-01") - epoch).total_seconds() / 86400
    end = (pd.Timestamp(f"{year + 1}-01-01") - epoch).total_seconds() / 86400
    return start, end


def load_tsobc(path: Path, year: int):
    """Return (time, T, S, h, siglay, nodes) for ``year`` from a tsobc NC."""
    with nc.Dataset(path) as ds:
        t = ds["time"][:].astype(np.float64)
        start, end = mjd_year_bounds(year)
        sel = (t >= start) & (t < end)
        if sel.sum() == 0:
            raise ValueError(f"{path} has no samples inside {year}")
        T = np.asarray(ds["obc_temp"][sel], dtype=np.float64)
        S = np.asarray(ds["obc_salinity"][sel], dtype=np.float64)
        h = np.asarray(ds["obc_h"][:], dtype=np.float64)
        siglay = np.asarray(ds["siglay"][:], dtype=np.float64)
        nodes = np.asarray(ds["obc_nodes"][:], dtype=np.int64)
    return t[sel], T, S, h, siglay, nodes


def interp_profile(
    values: np.ndarray, src_z: np.ndarray, dst_z: np.ndarray
) -> np.ndarray:
    """Re-interpolate a 1-D profile (values defined at src_z) onto dst_z.

    Both z arrays are negative downward. ``np.interp`` requires monotonically
    increasing x, so sort src_z ascending first.
    """
    order = np.argsort(src_z)
    return np.interp(dst_z, src_z[order], values[order])


# Months belonging to each meteorological season
SEASONS: dict[str, list[int]] = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}
MJD_EPOCH = pd.Timestamp("1858-11-17")


def months_from_mjd(mjd: np.ndarray) -> np.ndarray:
    """Return month-of-year (1..12) for an array of float MJD values."""
    dt = MJD_EPOCH + pd.to_timedelta(mjd, unit="D")
    return np.asarray(pd.DatetimeIndex(dt).month, dtype=np.int32)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", type=Path, required=True, help="legacy tsobc NetCDF")
    p.add_argument("--new", type=Path, required=True, help="new tsobc NetCDF")
    p.add_argument(
        "--year", type=int, required=True, help="calendar year to slice both files to"
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--label-old", default="legacy (J-EGG500 avg / uniform σ)")
    p.add_argument("--label-new", default="new (basic.nc real σ)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] old = {args.old}")
    t_o, T_o, S_o, h_o, slay_o, nd_o = load_tsobc(args.old, args.year)
    print(f"  n_time={t_o.size} n_obc={nd_o.size}")

    print(f"[load] new = {args.new}")
    t_n, T_n, S_n, h_n, slay_n, nd_n = load_tsobc(args.new, args.year)
    print(f"  n_time={t_n.size} n_obc={nd_n.size}")

    if not np.array_equal(nd_o, nd_n):
        sys.exit(f"OBC node order differs: old={nd_o.tolist()} new={nd_n.tolist()}")

    n_obc = nd_o.size
    n_lay_o = slay_o.shape[0]
    n_lay_n = slay_n.shape[0]

    # Per-node depths (negative downward)
    z_o = slay_o * h_o[None, :]  # (siglay_old, nobc)
    z_n = slay_n * h_n[None, :]  # (siglay_new, nobc)

    T_o_mean = T_o.mean(axis=0)  # (siglay_old, nobc)
    T_n_mean = T_n.mean(axis=0)
    S_o_mean = S_o.mean(axis=0)
    S_n_mean = S_n.mean(axis=0)

    # ---- profile-comparison helper ----
    def plot_profile(field_o, field_n, axis_label, title, fname):
        fig, axes = plt.subplots(4, 4, figsize=(14, 14))
        axes = axes.flatten()
        for n in range(n_obc):
            ax = axes[n]
            ax.plot(
                field_o[:, n],
                z_o[:, n],
                "r--",
                lw=1.5,
                alpha=0.85,
                label=args.label_old,
            )
            ax.plot(
                field_n[:, n],
                z_n[:, n],
                "b-",
                lw=1.5,
                alpha=0.85,
                label=args.label_new,
            )
            ax.axhline(-h_o[n], color="r", lw=0.5, ls=":", alpha=0.5)
            ax.axhline(-h_n[n], color="b", lw=0.5, ls=":", alpha=0.5)
            ax.set_title(
                f"node {int(nd_o[n])}: h_old={h_o[n]:.0f}m h_new={h_n[n]:.0f}m",
                fontsize=9,
            )
            ax.set_xlabel(axis_label, fontsize=8)
            ax.set_ylabel("z (m)", fontsize=8)
            ax.grid(alpha=0.3)
            if n == 0:
                ax.legend(fontsize=7, loc="lower right")
        for j in range(n_obc, len(axes)):
            axes[j].axis("off")
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = args.output_dir / fname
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  wrote {out}")

    print("[plot] annual-mean profiles")
    plot_profile(
        T_o_mean,
        T_n_mean,
        "temperature (°C)",
        f"OBC annual-mean temperature ({args.year})",
        f"tsobc_compare_T_annual_mean_{args.year}.png",
    )
    plot_profile(
        S_o_mean,
        S_n_mean,
        "salinity (PSU)",
        f"OBC annual-mean salinity ({args.year})",
        f"tsobc_compare_S_annual_mean_{args.year}.png",
    )

    # ---- seasonal-mean plots ----
    print("[plot] seasonal-mean profiles")
    mon_o = months_from_mjd(t_o)
    mon_n = months_from_mjd(t_n)
    season_means: dict[str, dict[str, np.ndarray]] = {}
    for season, months in SEASONS.items():
        sel_o = np.isin(mon_o, months)
        sel_n = np.isin(mon_n, months)
        if sel_o.sum() == 0 or sel_n.sum() == 0:
            print(f"  {season}: no samples — skipping")
            continue
        T_o_m = T_o[sel_o].mean(axis=0)
        T_n_m = T_n[sel_n].mean(axis=0)
        S_o_m = S_o[sel_o].mean(axis=0)
        S_n_m = S_n[sel_n].mean(axis=0)
        season_means[season] = {
            "T_old": T_o_m,
            "T_new": T_n_m,
            "S_old": S_o_m,
            "S_new": S_n_m,
        }
        plot_profile(
            T_o_m,
            T_n_m,
            "temperature (°C)",
            f"OBC {season} temperature ({args.year})",
            f"tsobc_compare_T_{season}_{args.year}.png",
        )
        plot_profile(
            S_o_m,
            S_n_m,
            "salinity (PSU)",
            f"OBC {season} salinity ({args.year})",
            f"tsobc_compare_S_{season}_{args.year}.png",
        )

    # ---- per-node statistics (annual + per-season, regrid old to new z) ----
    print()
    print(f"=== Per-node statistics ({args.year}, regridded to new z) ===")
    header = (
        f"{'season':>6}  {'node':>5}  {'h_old':>7} {'h_new':>7}  "
        f"{'RMSE_T':>7} {'bias_T':>7}  {'RMSE_S':>7} {'bias_S':>7}"
    )
    print(header)
    print("-" * len(header))

    def stats_row(season_label, T_o_m, T_n_m, S_o_m, S_n_m):
        for n in range(n_obc):
            T_o_on_new = interp_profile(T_o_m[:, n], z_o[:, n], z_n[:, n])
            S_o_on_new = interp_profile(S_o_m[:, n], z_o[:, n], z_n[:, n])
            d_T = T_n_m[:, n] - T_o_on_new
            d_S = S_n_m[:, n] - S_o_on_new
            rmse_T = float(np.sqrt(np.mean(d_T**2)))
            bias_T = float(d_T.mean())
            rmse_S = float(np.sqrt(np.mean(d_S**2)))
            bias_S = float(d_S.mean())
            print(
                f"{season_label:>6}  {int(nd_o[n]):>5}  "
                f"{h_o[n]:>7.1f} {h_n[n]:>7.1f}  "
                f"{rmse_T:>7.3f} {bias_T:>+7.3f}  "
                f"{rmse_S:>7.3f} {bias_S:>+7.3f}"
            )

    stats_row("ANNUAL", T_o_mean, T_n_mean, S_o_mean, S_n_mean)
    for season, vals in season_means.items():
        stats_row(
            season,
            vals["T_old"],
            vals["T_new"],
            vals["S_old"],
            vals["S_new"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
