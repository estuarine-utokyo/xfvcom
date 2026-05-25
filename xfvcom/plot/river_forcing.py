"""River / sewer (STP) forcing time-series engine.

Region-agnostic, reusable plotting of the freshwater forcing that FVCOM
*consumes* -- read straight from the river forcing NetCDF
(``river_flux`` / ``river_temp`` / ``river_salt`` over ``(time, rivers)``)
rather than from a re-run model output.  Reading the input file makes the
plots cheap to regenerate every time the boundary condition is rebuilt
(no FVCOM re-run required), which is the intended use: re-run after every
BC change.

The companion :mod:`xfvcom.plot.freshwater_map` draws *where* the sources
are; this module draws *what* each source injects over time.

Three layers, increasingly high level:

``load_river_forcing(nc, nml=...)``
    Decode the NC into a :class:`RiverForcing` (times, names, flux, temp,
    salt, per-entry ``RIVER_FLUX_SCALE_LOCAL`` already applied to flux).

``classify_outflows(names, river_map_yaml)`` / ``aggregate_entities(...)``
    Join the FVCOM outflow names to the authoritative schema-v4 river/sewer
    map so each outflow is tagged ``River``/``Sewer`` and grouped under its
    logical ``source.entity`` (e.g. four ``*Arakawa`` outflows -> one
    ``Arakawa`` entity: flux summed, temperature flux-weighted).

``plot_river_forcing(...)`` / ``plot_river_forcing_grid(...)``
    Draw a panel grid (one panel per entity or per outflow) of discharge or
    temperature, with a shared y-range and a per-panel mean annotation.

All figure text is ASCII-only (per project convention).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

_MJD_EPOCH_NS: int = int(np.datetime64("1858-11-17", "ns").astype("int64"))
_DAY_NS = np.int64(86_400_000_000_000)
_MS_NS = np.int64(1_000_000)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass
class RiverForcing:
    """Decoded FVCOM river/sewer forcing.

    Attributes
    ----------
    times : np.ndarray
        ``datetime64[ns]`` time axis, shape ``(T,)`` (UTC).
    names : list[str]
        FVCOM outflow names, length ``N`` (the ``rivers`` dimension).
    flux : np.ndarray
        Runoff volume flux ``(T, N)`` in m^3/s, with the per-entry
        ``RIVER_FLUX_SCALE_LOCAL`` already applied when a namelist is given.
    temp : np.ndarray
        Runoff temperature ``(T, N)`` in deg C.
    salt : np.ndarray | None
        Runoff salinity ``(T, N)`` in PSU, when present.
    scales : np.ndarray
        The applied ``RIVER_FLUX_SCALE_LOCAL`` per entry, shape ``(N,)``
        (all ``1.0`` when no namelist was supplied).
    """

    times: np.ndarray
    names: list[str]
    flux: np.ndarray
    temp: np.ndarray
    salt: np.ndarray | None
    scales: np.ndarray


def _decode_times(ds) -> np.ndarray:
    """Decode an FVCOM time axis to ``datetime64[ns]``.

    Prefers the integer ``Itime`` / ``Itime2`` pair (exact); falls back to
    the floating ``time`` (days since the MJD epoch).
    """
    if "Itime" in ds.variables and "Itime2" in ds.variables:
        itime = np.asarray(ds["Itime"][:], dtype="int64")
        itime2 = np.asarray(ds["Itime2"][:], dtype="int64")
        return (_MJD_EPOCH_NS + itime * _DAY_NS + itime2 * _MS_NS).view(
            "datetime64[ns]"
        )
    days = np.asarray(ds["time"][:], dtype="float64")
    return (_MJD_EPOCH_NS + np.round(days * _DAY_NS).astype("int64")).view(
        "datetime64[ns]"
    )


def _decode_names(arr) -> list[str]:
    out: list[str] = []
    for row in arr[:]:
        out.append(b"".join(row).strip().decode("utf-8", errors="replace"))
    return out


def parse_river_scales(nml_path: str | Path, n: int) -> np.ndarray:
    """Return per-entry ``RIVER_FLUX_SCALE_LOCAL`` (length ``n``).

    Entries with no matching scale line default to ``1.0``.  The namelist
    lists one ``&NML_RIVER`` block per entry in NC order, so the i-th match
    maps to the i-th outflow.
    """
    scales: np.ndarray = np.ones(n, dtype="float64")
    nml_path = Path(nml_path)
    if not nml_path.exists():
        return scales
    matches = re.findall(
        r"RIVER_FLUX_SCALE_LOCAL\s*=\s*([0-9.+\-eE]+)", nml_path.read_text()
    )
    for i, m in enumerate(matches[:n]):
        scales[i] = float(m)
    return scales


def load_river_forcing(
    nc_path: str | Path, *, nml_path: str | Path | None = None
) -> RiverForcing:
    """Load an FVCOM river/sewer forcing NetCDF into a :class:`RiverForcing`.

    Parameters
    ----------
    nc_path : path
        FVCOM river forcing file (``river_flux``/``river_temp`` over
        ``(time, rivers)``, ``river_names`` over ``(rivers, namelen)``).
    nml_path : path, optional
        ``RIVERS_NAMELIST`` to read ``RIVER_FLUX_SCALE_LOCAL`` from; when
        given, the scale is applied to ``flux`` (so the plot shows what
        FVCOM actually injects).  When omitted, scales are ``1.0``.
    """
    import netCDF4 as nc  # local import: keeps module import light

    with nc.Dataset(nc_path) as ds:
        times = _decode_times(ds)
        names = _decode_names(ds["river_names"])
        flux = np.asarray(ds["river_flux"][:], dtype="float64")
        temp = np.asarray(ds["river_temp"][:], dtype="float64")
        salt = (
            np.asarray(ds["river_salt"][:], dtype="float64")
            if "river_salt" in ds.variables
            else None
        )
    n = flux.shape[1]
    scales = parse_river_scales(nml_path, n) if nml_path is not None else np.ones(n)
    flux = flux * scales[np.newaxis, :]
    return RiverForcing(
        times=times, names=names, flux=flux, temp=temp, salt=salt, scales=scales
    )


# ---------------------------------------------------------------------------
# Classification / aggregation from the authoritative river-map YAML
# ---------------------------------------------------------------------------
def classify_outflows(
    names: list[str],
    river_map_yaml: str | Path,
    *,
    sewer_domain: str = "wastewater",
) -> dict[str, dict]:
    """Map each FVCOM outflow name to ``{"entity", "kind"}``.

    The schema-v4 river/sewer map keys rows by their FVCOM ``name`` and
    carries ``source.{domain, entity}`` plus an optional ``kind``.  An entry
    is ``"Sewer"`` when ``source.domain == sewer_domain`` (default
    ``"wastewater"``) or ``kind == "constant"``; otherwise ``"River"``.  The
    grouping ``entity`` is ``source.entity`` (falling back to the FVCOM name).

    Outflow names absent from the YAML fall back to
    ``{"entity": <name>, "kind": "River"}`` so the caller still gets a
    complete mapping.
    """
    with Path(river_map_yaml).open() as fh:
        doc = yaml.safe_load(fh)
    by_name: dict[str, dict] = {}
    for e in doc.get("rivers", []):
        src = e.get("source", {}) or {}
        is_sewer = src.get("domain") == sewer_domain or e.get("kind") == "constant"
        entity = e["name"] if e.get("kind") == "constant" else src.get("entity")
        by_name[e["name"]] = {
            "entity": entity or e["name"],
            "kind": "Sewer" if is_sewer else "River",
        }
    return {
        name: by_name.get(name, {"entity": name, "kind": "River"}) for name in names
    }


def aggregate_entities(
    forcing: RiverForcing, classification: dict[str, dict]
) -> dict[str, dict]:
    """Roll up per-outflow series to per-entity series.

    Returns ``{entity: {"kind", "names", "flux", "temp"}}`` where ``flux`` is
    the sum over the entity's outflows and ``temp`` is the flux-weighted mean
    (falling back to the simple mean at times when the entity's total flux is
    ~0).  Entity order follows first appearance in ``forcing.names``.
    """
    order: list[str] = []
    members: dict[str, list[int]] = {}
    for i, name in enumerate(forcing.names):
        ent = classification[name]["entity"]
        if ent not in members:
            members[ent] = []
            order.append(ent)
        members[ent].append(i)

    out: dict[str, dict] = {}
    for ent in order:
        idx = members[ent]
        flux = forcing.flux[:, idx]
        temp = forcing.temp[:, idx]
        flux_sum = np.nansum(flux, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            wmean = np.nansum(flux * temp, axis=1) / flux_sum
        simple = np.nanmean(temp, axis=1)
        temp_agg = np.where(np.isfinite(wmean) & (flux_sum > 0), wmean, simple)
        out[ent] = {
            "kind": classification[forcing.names[idx[0]]]["kind"],
            "names": [forcing.names[i] for i in idx],
            "flux": flux_sum,
            "temp": temp_agg,
        }
    return out


def river_annual_summary(
    forcing: RiverForcing,
    classification: dict[str, dict],
    *,
    year: int | None = None,
) -> pd.DataFrame:
    """Per-outflow annual summary (the tracked R5-style table).

    Columns: ``name, entity, kind, scale_local, Q_mean_m3s, Q_annual_m3,
    T_mean_C``.  ``Q_annual_m3`` integrates flux over the (optionally
    year-masked) record using the mean sampling interval.
    """
    t = forcing.times
    mask = np.ones(t.shape, dtype=bool)
    if year is not None:
        y0 = np.datetime64(f"{year}-01-01", "ns")
        y1 = np.datetime64(f"{year + 1}-01-01", "ns")
        mask = (t >= y0) & (t < y1)
    t_sec = t[mask].astype("int64") / 1e9
    dt = float(np.mean(np.diff(t_sec))) if t_sec.size > 1 else 3600.0

    flux = forcing.flux[mask, :]
    temp = forcing.temp[mask, :]
    rows = []
    for i, name in enumerate(forcing.names):
        info = classification[name]
        rows.append(
            {
                "name": name,
                "entity": info["entity"],
                "kind": info["kind"].lower(),
                "scale_local": float(forcing.scales[i]),
                "Q_mean_m3s": round(float(np.nanmean(flux[:, i])), 4),
                "Q_annual_m3": round(float(np.nansum(flux[:, i]) * dt), 0),
                "T_mean_C": round(float(np.nanmean(temp[:, i])), 3),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_river_forcing_grid(
    times: np.ndarray,
    series: list[tuple[str, np.ndarray]],
    *,
    out: str | Path | None = None,
    title: str = "",
    ylabel: str = "",
    n_cols: int = 4,
    color: str = "#1f5fbf",
    clip_pct: tuple[float, float] = (0.5, 99.5),
    annotate_mean: bool = True,
    mean_fmt: str = "mean={:.3g}",
    dpi: int = 300,
):
    """Panel grid of per-source time series sharing one y-range.

    Parameters
    ----------
    times : np.ndarray
        Shared ``datetime64`` x-axis, length ``T``.
    series : list[(label, values)]
        One ``(label, (T,) array)`` per panel; panels are laid out row-major.
    clip_pct : (lo, hi)
        Percentiles used to set the common y-range (robust to spikes).
    annotate_mean : bool
        Annotate each panel with its finite mean (top-left).

    Returns ``dict(fig=..., ax=...)``.
    """
    n = len(series)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.1 * n_cols, 1.7 * n_rows + 1.0),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    stacked = np.concatenate([v[np.isfinite(v)] for _, v in series if v.size])
    if stacked.size:
        lo = float(np.percentile(stacked, clip_pct[0]))
        hi = float(np.percentile(stacked, clip_pct[1]))
        pad = max((hi - lo) * 0.05, 1e-9)
        ymin, ymax = lo - pad, hi + pad
    else:
        ymin, ymax = 0.0, 1.0

    for ax, (label, v) in zip(axes_flat, series):
        ax.plot(times, v, color=color, lw=0.8)
        ax.set_title(label, fontsize=8)
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
        if annotate_mean and v.size and np.isfinite(v).any():
            ax.text(
                0.03,
                0.92,
                mean_fmt.format(float(np.nanmean(v))),
                transform=ax.transAxes,
                fontsize=7,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )
    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(f"{title}  ({ylabel})" if ylabel else title, fontsize=12)
    fig.supxlabel("month (UTC)", fontsize=9)
    fig.tight_layout(rect=(0, 0.0, 1, 0.98))
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi)
    return dict(fig=fig, ax=axes)


def plot_river_forcing(
    forcing: RiverForcing,
    classification: dict[str, dict],
    *,
    value: str = "flux",
    kind: str | None = None,
    aggregate: bool = True,
    out: str | Path | None = None,
    title: str | None = None,
    n_cols: int | None = None,
    river_color: str = "#1f5fbf",
    sewer_color: str = "#cc2b2b",
    dpi: int = 300,
):
    """High-level: one figure of discharge or temperature for a source kind.

    Parameters
    ----------
    value : ``"flux"`` | ``"temp"``
        Which series to draw.
    kind : ``"River"`` | ``"Sewer"`` | ``None``
        Restrict to one source kind (``None`` keeps both).
    aggregate : bool
        Group outflows into logical entities (``True``, default) or draw one
        panel per FVCOM outflow (``False``).
    n_cols : int, optional
        Panel columns; defaults to 4 for rivers / 2 for sewers when ``kind``
        is set, else 4.

    Returns ``dict(fig=..., ax=...)``; writes ``out`` when given.
    """
    if value not in ("flux", "temp"):
        raise ValueError(f"value must be 'flux' or 'temp', got {value!r}")

    if aggregate:
        ent = aggregate_entities(forcing, classification)
        items = [
            (name, d["kind"], d[value])
            for name, d in ent.items()
            if kind is None or d["kind"] == kind
        ]
    else:
        col = forcing.flux if value == "flux" else forcing.temp
        items = [
            (name, classification[name]["kind"], col[:, i])
            for i, name in enumerate(forcing.names)
            if kind is None or classification[name]["kind"] == kind
        ]

    series = [(label, v) for label, _k, v in items]
    color = sewer_color if kind == "Sewer" else river_color
    if n_cols is None:
        n_cols = 2 if kind == "Sewer" else 4

    ylabel = "Q [m^3/s]" if value == "flux" else "T [deg C]"
    if title is None:
        what = {"River": "River", "Sewer": "Sewer (STP)", None: "Source"}[kind]
        kind_word = "discharge" if value == "flux" else "temperature"
        scope = "entity" if aggregate else "outflow"
        title = f"{what} {kind_word} ({len(series)} {scope}s)"

    return plot_river_forcing_grid(
        forcing.times,
        series,
        out=out,
        title=title,
        ylabel=ylabel,
        n_cols=n_cols,
        color=color,
        clip_pct=(1.0, 99.0) if value == "temp" else (0.5, 99.5),
        dpi=dpi,
    )
