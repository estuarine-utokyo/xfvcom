"""Meteorological forcing engine: station time series + spatial maps.

Region-agnostic, reusable plotting of the atmospheric forcing that FVCOM
*consumes*, read straight from the surface-forcing NetCDF (the metforce /
"wnd" file on the FVCOM mesh) rather than from a re-run model output.  As
with :mod:`xfvcom.plot.river_forcing`, working from the input file makes the
plots cheap to regenerate after every boundary-condition rebuild and exposes
fields the model does not echo back (air temperature, humidity, cloud,
precipitation).

The mesh layout follows the FVCOM convention: wind components live on
elements ``(time, nele)``; scalar fields (air temperature, radiation,
pressure, humidity, cloud, precipitation) live on nodes ``(time, node)``.

Two public entry points:

``plot_met_station_timeseries(nc, stations, var, obs_provider=...)``
    Sample the forcing at each station's nearest node/element, reduce to a
    daily mean, and stack one panel per station -- overlaying an observation
    series when the caller's ``obs_provider`` returns one.

``plot_met_map_monthly(nc, var, ...)`` / ``plot_met_map_annual(nc, var, ...)``
    Monthly (3x4) or annual surface-field maps on the unstructured mesh, with
    a coastline outline and optional station markers; wind may add a
    mean-vector quiver overlay.

Variables are described by :data:`MET_VARIABLES`; callers refer to them by
key (e.g. ``"wind_speed"``, ``"air_temperature"``) or pass a custom
:class:`MetVar`.  All figure text is ASCII-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.tri import Triangulation

from .source_map import extract_boundary_edges, order_boundary_nodes

_MJD_EPOCH_NS: int = int(np.datetime64("1858-11-17", "ns").astype("int64"))
_DAY_NS = np.int64(86_400_000_000_000)


def _load_coords(nc_path: str | Path) -> dict:
    """Load just the mesh coordinates a forcing file carries.

    Unlike :func:`xfvcom.plot.source_map.load_grid_coordinates`, this does
    not require a bathymetry ``h`` variable (surface-forcing files such as
    the metforce NC have none).  Returns node ``lon``/``lat``, element
    ``lonc``/``latc``, ``nv3`` (``(3, nele)`` 0-based, for boundary tracing)
    and ``tri`` (``(nele, 3)`` 0-based, for :class:`matplotlib.tri.Triangulation`).
    """
    import netCDF4 as nc

    with nc.Dataset(nc_path) as ds:
        out = {
            "lon": np.asarray(ds["lon"][:], dtype="float64"),
            "lat": np.asarray(ds["lat"][:], dtype="float64"),
            "lonc": np.asarray(ds["lonc"][:], dtype="float64"),
            "latc": np.asarray(ds["latc"][:], dtype="float64"),
        }
        nv = np.asarray(ds["nv"][:], dtype="int64") - 1  # 0-based
    nv3 = nv if nv.shape[0] == 3 else nv.T
    out["nv3"] = nv3
    out["tri"] = nv3.T
    return out


# ---------------------------------------------------------------------------
# Variable registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetVar:
    """Description of one meteorological forcing field.

    Attributes
    ----------
    key : str
        Lookup key (also the observation-provider key).
    label : str
        Human title, ASCII-only.
    unit : str
        Display unit (after ``scale``).
    grid : ``"node"`` | ``"element"``
        Where the source field lives on the FVCOM mesh.
    nc_vars : tuple[str, ...]
        Source NC variable name(s) (two for wind, one otherwise).
    kind : ``"scalar"`` | ``"wind_speed"`` | ``"wind_direction"``
        How to compose a value from ``nc_vars``.
    scale : float
        Multiplicative factor applied to the raw value (e.g. precip
        ``m/s -> mm/day``).
    map_ok : bool
        Whether a spatial map is meaningful (``False`` for wind direction).
    cmap : str
        Colormap for maps.
    """

    key: str
    label: str
    unit: str
    grid: str
    nc_vars: tuple[str, ...]
    kind: str = "scalar"
    scale: float = 1.0
    map_ok: bool = True
    cmap: str = "turbo"


MET_VARIABLES: dict[str, MetVar] = {
    "wind_speed": MetVar(
        "wind_speed",
        "Wind speed |U10|",
        "m/s",
        "element",
        ("uwind_speed", "vwind_speed"),
        kind="wind_speed",
    ),
    "wind_from_direction": MetVar(
        "wind_from_direction",
        "Wind direction (from)",
        "deg",
        "element",
        ("uwind_speed", "vwind_speed"),
        kind="wind_direction",
        map_ok=False,
        cmap="twilight",
    ),
    "air_temperature": MetVar(
        "air_temperature",
        "Air temperature",
        "deg C",
        "node",
        ("air_temperature",),
    ),
    "air_pressure": MetVar(
        "air_pressure",
        "Sea-level pressure",
        "hPa",
        "node",
        ("air_pressure",),
        cmap="viridis",
    ),
    "short_wave": MetVar(
        "short_wave",
        "Short-wave radiation",
        "W/m^2",
        "node",
        ("short_wave",),
    ),
    "long_wave": MetVar(
        "long_wave",
        "Long-wave radiation",
        "W/m^2",
        "node",
        ("long_wave",),
    ),
    "relative_humidity": MetVar(
        "relative_humidity",
        "Relative humidity",
        "%",
        "node",
        ("relative_humidity",),
        cmap="viridis",
    ),
    "cloud_cover": MetVar(
        "cloud_cover",
        "Cloud cover",
        "-",
        "node",
        ("cloud_cover",),
        cmap="Blues",
    ),
    "precipitation": MetVar(
        "precipitation",
        "Precipitation",
        "mm/day",
        "node",
        ("Precipitation",),
        scale=1000.0 * 86400.0,
        cmap="Blues",
    ),
}


def resolve_var(var: str | MetVar) -> MetVar:
    """Return a :class:`MetVar` from a registry key or pass one through."""
    if isinstance(var, MetVar):
        return var
    try:
        return MET_VARIABLES[var]
    except KeyError as exc:
        raise KeyError(
            f"unknown met variable {var!r}; known: {sorted(MET_VARIABLES)}"
        ) from exc


# ---------------------------------------------------------------------------
# NC helpers
# ---------------------------------------------------------------------------
def _decode_times(ds) -> np.ndarray:
    if "Itime" in ds.variables and "Itime2" in ds.variables:
        itime = np.asarray(ds["Itime"][:], dtype="int64")
        itime2 = np.asarray(ds["Itime2"][:], dtype="int64")
        return (_MJD_EPOCH_NS + itime * _DAY_NS + itime2 * np.int64(1_000_000)).view(
            "datetime64[ns]"
        )
    days = np.asarray(ds["time"][:], dtype="float64")
    return (_MJD_EPOCH_NS + np.round(days * _DAY_NS).astype("int64")).view(
        "datetime64[ns]"
    )


def _read(ds, name: str, col: int | None = None) -> np.ndarray:
    arr = ds[name][:, col] if col is not None else ds[name][:]
    if hasattr(arr, "filled"):
        arr = arr.filled(np.nan)
    return np.asarray(arr, dtype="float64")


def _nearest_index(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> int:
    """Nearest mesh point to ``(lon0, lat0)`` using a cos-lat metric."""
    dx = (lon - lon0) * np.cos(np.deg2rad(lat0))
    dy = lat - lat0
    return int(np.argmin(dx * dx + dy * dy))


def _wind_from_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Meteorological FROM-direction in degrees (0=N, 90=E)."""
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


# ---------------------------------------------------------------------------
# Station time series
# ---------------------------------------------------------------------------
def extract_station_series(
    ds, var: MetVar, lon0: float, lat0: float, coords: dict
) -> pd.Series:
    """Daily-mean forcing series for ``var`` at the mesh point nearest
    ``(lon0, lat0)``.  Wind direction is vector-averaged before reduction.
    """
    times = pd.DatetimeIndex(_decode_times(ds))
    if var.grid == "element":
        idx = _nearest_index(coords["lonc"], coords["latc"], lon0, lat0)
        u = _read(ds, "uwind_speed", idx)
        v = _read(ds, "vwind_speed", idx)
        if var.kind == "wind_direction":
            su = pd.Series(u, index=times).resample("1D").mean()
            sv = pd.Series(v, index=times).resample("1D").mean()
            return pd.Series(
                _wind_from_direction(su.to_numpy(), sv.to_numpy()), index=su.index
            )
        val = np.sqrt(u * u + v * v)
    else:
        idx = _nearest_index(coords["lon"], coords["lat"], lon0, lat0)
        val = _read(ds, var.nc_vars[0], idx) * var.scale
    return pd.Series(val, index=times).resample("1D").mean()


def plot_met_station_timeseries(
    nc_path: str | Path,
    stations: list[tuple[str, float, float]],
    var: str | MetVar,
    *,
    obs_provider=None,
    year: int | None = None,
    out: str | Path | None = None,
    title: str | None = None,
    model_color: str = "#1f5fbf",
    obs_color: str = "#cc2b2b",
    clip_pct: tuple[float, float] = (0.5, 99.5),
    dpi: int = 300,
):
    """Stacked per-station time series of one forcing variable.

    Parameters
    ----------
    stations : list[(name, lon, lat)]
        One panel per station; the model is sampled at the nearest mesh
        point and reduced to a daily mean.
    obs_provider : callable, optional
        ``fn(station_name, var_key, year) -> pd.Series | None`` giving a
        daily-mean observation overlay (returns ``None`` when none exists).
    year : int, optional
        Restrict both model and obs to this calendar year.

    Returns ``dict(fig=..., ax=...)``; writes ``out`` when given.
    """
    import netCDF4 as nc

    mv = resolve_var(var)
    coords = _load_coords(nc_path)

    lo_t = pd.Timestamp(f"{year}-01-01") if year is not None else None
    hi_t = pd.Timestamp(f"{year + 1}-01-01") if year is not None else None

    def _clip(s: pd.Series | None) -> pd.Series | None:
        if s is None or lo_t is None or hi_t is None:
            return s
        return s[(s.index >= lo_t) & (s.index < hi_t)]

    rows: list[tuple[str, pd.Series, pd.Series | None]] = []
    with nc.Dataset(nc_path) as ds:
        for name, lon0, lat0 in stations:
            model = extract_station_series(ds, mv, lon0, lat0, coords)
            obs: pd.Series | None = None
            if obs_provider is not None:
                try:
                    obs = obs_provider(name, mv.key, year)
                except Exception as exc:  # obs is best-effort
                    print(f"[met_forcing] obs failed {name}/{mv.key}: {exc}")
            model_clipped = _clip(model)
            assert model_clipped is not None  # model is never None
            rows.append((name, model_clipped, _clip(obs)))

    pool = []
    for _n, m, o in rows:
        for s in (m, o):
            if s is not None and s.notna().any():
                pool.append(s.dropna().to_numpy())
    if pool:
        merged = np.concatenate(pool)
        lo_y = float(np.percentile(merged, clip_pct[0]))
        hi_y = float(np.percentile(merged, clip_pct[1]))
        pad = max((hi_y - lo_y) * 0.05, 1e-6)
        ymin, ymax = lo_y - pad, hi_y + pad
    else:
        ymin, ymax = -1.0, 1.0

    n = len(rows)
    fig, axes = plt.subplots(
        n, 1, figsize=(11, 1.8 * n + 1.0), sharex=True, squeeze=False
    )
    axes_flat = axes.flatten()
    for ax, (name, model, obs) in zip(axes_flat, rows):
        if model.notna().any():
            ax.plot(
                model.index,
                model.to_numpy(),
                color=model_color,
                lw=1.0,
                label="forcing",
            )
        if obs is not None and obs.notna().any():
            ax.plot(obs.index, obs.to_numpy(), color=obs_color, lw=1.0, label="obs")
            ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.8)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(f"{name}\n{mv.unit}", fontsize=8)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
    axes_flat[-1].set_xlabel("month (UTC)")
    if title is None:
        title = f"{mv.label} ({mv.unit}) -- daily mean at stations"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi)
    return dict(fig=fig, ax=axes)


# ---------------------------------------------------------------------------
# Spatial maps
# ---------------------------------------------------------------------------
def _triangulation(coords: dict) -> Triangulation:
    return Triangulation(coords["lon"], coords["lat"], coords["tri"])


def _monthly_field(ds, var: MetVar) -> np.ndarray:
    """Per-(node|element) monthly mean, shape ``(12, M)`` reindexed to 1..12."""
    times = pd.DatetimeIndex(_decode_times(ds))
    if var.kind == "wind_speed":
        u = _read(ds, "uwind_speed")
        v = _read(ds, "vwind_speed")
        vals = np.sqrt(u * u + v * v)
    else:
        vals = _read(ds, var.nc_vars[0]) * var.scale
    df = pd.DataFrame(vals, index=times)
    monthly = df.groupby(df.index.month).mean().reindex(range(1, 13))
    return monthly.to_numpy()


def _coast_paths(coords: dict):
    return sorted(
        order_boundary_nodes(extract_boundary_edges(coords["nv3"])),
        key=len,
        reverse=True,
    )


def _draw_field(ax, coords, tri, values, is_element, vmin, vmax, cmap):
    if is_element:
        return ax.tripcolor(
            tri,
            facecolors=values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="flat",
        )
    return ax.tripcolor(tri, values, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")


def _decorate_map(ax, coords, paths, stations, xlim, ylim):
    lon, lat = coords["lon"], coords["lat"]
    for p in paths:
        ax.plot(lon[p], lat[p], color="#444444", lw=0.5, zorder=3)
    if stations:
        for _name, slon, slat in stations:
            ax.scatter(
                slon,
                slat,
                marker="v",
                s=28,
                c="white",
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    mid = float(np.mean(ax.get_ylim()))
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mid)))
    ax.set_xticks([])
    ax.set_yticks([])


def plot_met_map_monthly(
    nc_path: str | Path,
    var: str | MetVar,
    *,
    out: str | Path | None = None,
    title: str | None = None,
    stations: list[tuple[str, float, float]] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    clip_pct: tuple[float, float] = (1.0, 99.0),
    dpi: int = 300,
):
    """12-panel (3x4) monthly-mean surface map of a forcing variable.

    A shared colour scale spans the robust percentile range across all
    months; the coastline is outlined and any ``stations`` are marked.

    Returns ``dict(fig=..., ax=...)``; writes ``out`` when given.
    """
    import netCDF4 as nc

    mv = resolve_var(var)
    if not mv.map_ok:
        raise ValueError(f"{mv.key!r} has no meaningful scalar map")
    coords = _load_coords(nc_path)
    tri = _triangulation(coords)
    paths = _coast_paths(coords)
    is_element = mv.grid == "element"
    with nc.Dataset(nc_path) as ds:
        monthly = _monthly_field(ds, mv)

    vmin = float(np.nanpercentile(monthly, clip_pct[0]))
    vmax = float(np.nanpercentile(monthly, clip_pct[1]))
    fig, axes = plt.subplots(3, 4, figsize=(11, 12))
    last = None
    for ax, m in zip(axes.flat, range(1, 13)):
        last = _draw_field(
            ax, coords, tri, monthly[m - 1], is_element, vmin, vmax, mv.cmap
        )
        _decorate_map(ax, coords, paths, stations, xlim, ylim)
        ax.set_title(f"month {m:02d}", fontsize=9, pad=2)
    fig.subplots_adjust(
        left=0.02, right=0.90, top=0.94, bottom=0.02, wspace=0.04, hspace=0.10
    )
    cax = fig.add_axes((0.915, 0.10, 0.020, 0.78))
    fig.colorbar(last, cax=cax, label=f"{mv.label} [{mv.unit}]")
    fig.suptitle(
        title or f"{mv.label} -- monthly mean ({mv.unit})", fontsize=13, y=0.985
    )
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi)
    return dict(fig=fig, ax=axes)


def plot_met_map_annual(
    nc_path: str | Path,
    var: str | MetVar,
    *,
    out: str | Path | None = None,
    title: str | None = None,
    stations: list[tuple[str, float, float]] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    wind_quiver: bool = True,
    clip_pct: tuple[float, float] = (1.0, 99.0),
    dpi: int = 300,
):
    """Single-panel annual-mean surface map of a forcing variable.

    For ``wind_speed`` with ``wind_quiver=True`` the mean wind vector field
    is overlaid as a sparse element quiver.

    Returns ``dict(fig=..., ax=...)``; writes ``out`` when given.
    """
    import netCDF4 as nc

    mv = resolve_var(var)
    if not mv.map_ok:
        raise ValueError(f"{mv.key!r} has no meaningful scalar map")
    coords = _load_coords(nc_path)
    tri = _triangulation(coords)
    paths = _coast_paths(coords)
    is_element = mv.grid == "element"
    with nc.Dataset(nc_path) as ds:
        if mv.kind == "wind_speed":
            u_t = _read(ds, "uwind_speed")
            v_t = _read(ds, "vwind_speed")
            # Colour = mean wind SPEED (mean of the magnitude); the quiver
            # uses the mean wind VECTOR (mean of the components) -- distinct
            # quantities (the vector mean is smaller where direction varies).
            field = np.nanmean(np.sqrt(u_t * u_t + v_t * v_t), axis=0)
            u = np.nanmean(u_t, axis=0)
            v = np.nanmean(v_t, axis=0)
        else:
            field = np.nanmean(_read(ds, mv.nc_vars[0]), axis=0) * mv.scale
            u = v = None

    vmin = float(np.nanpercentile(field, clip_pct[0]))
    vmax = float(np.nanpercentile(field, clip_pct[1]))
    fig, ax = plt.subplots(figsize=(7, 9))
    tpc = _draw_field(ax, coords, tri, field, is_element, vmin, vmax, mv.cmap)
    if mv.kind == "wind_speed" and wind_quiver and u is not None:
        step = max(1, coords["lonc"].size // 350)
        sl = slice(None, None, step)
        ax.quiver(
            coords["lonc"][sl],
            coords["latc"][sl],
            u[sl],
            v[sl],
            color="black",
            scale=30,
            width=0.003,
            alpha=0.7,
            zorder=4,
        )
    _decorate_map(ax, coords, paths, stations, xlim, ylim)
    fig.colorbar(tpc, ax=ax, fraction=0.04, pad=0.02, label=f"{mv.label} [{mv.unit}]")
    ax.set_title(title or f"{mv.label} -- annual mean ({mv.unit})", fontsize=12)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return dict(fig=fig, ax=ax)
