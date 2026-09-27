"""river_dl-based discharge source for FVCOM river/sewer forcing.

Reads a per-river or per-plant ``discharge_hourly.nc`` produced by the
``river_dl`` toolkit (e.g. ``$DATA_DIR/river/discharge/<River>/<Station>/``
or ``$DATA_DIR/wastewater/<Plant>/``) and supplies the ``flux`` series
to :class:`xfvcom.io.river_nc_generator.RiverNetCDFGenerator`.

Expected NetCDF schema (river_dl Phase AM/AN/AO+, 2026-04-21):

================  =====================================================
``time(time)``    int64, ``"hours since YYYY-MM-DD HH:MM:SS"`` (proleptic)
``discharge(time)`` float, units ``m3/s``
``qc_flag(time)`` byte (optional, ignored here; available for audit)
================  =====================================================

Time zone (2026-09-27; TB-FVCOM hydro/docs/obc_jcope_temperature_bias.md S18.121). river_dl products carry no
zone attribute, and their time stamps are **JST** (river_dl ``docs/ersem_fvcom_usage.md``: "All timestamps are JST").
Until this date the stamps were compared with the UTC FVCOM timeline as wall-clock values, so every river/sewer
discharge series entered FVCOM **9 h late**. The source zone is now resolved per file, in order: an explicit
``source_tz`` argument; a ``time_zone`` attribute on ``time`` or on the file; ``UTC`` for the ``LegacyF04v3``
extracts (daily values decoded from an FVCOM forcing file, hence already UTC); otherwise ``Asia/Tokyo`` (the
river_dl product contract). Source instants are converted to UTC before interpolation. A DAILY product (one value
per calendar day stamped 00:00, e.g. Tamagawa/Ishihara ``discharge_daily.nc``) is the mean over that source-zone day
and is placed at its middle (12:00 source zone) before conversion.

The source intentionally returns *raw* observed/estimated discharge — no
Optuna #123 per-river scaling or Trial #16 seasonal modulation. Those are
runtime knobs applied via ``RIVERS_NAMELIST.RIVER_FLUX_SCALE_LOCAL`` and
the launcher's seasonal-cosine wrapper (see
``TB-FVCOM/hydro/docs/bc_construction_protocol.md`` rule R2).
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from .base import BaseForcingSource

_REQUIRED_VARS: Final[tuple[str, ...]] = ("discharge", "time")


class RiverDLNetCDFSource(BaseForcingSource):
    """Discharge time series from a river_dl ``discharge_hourly.nc``.

    Provides:

    * ``flux``  — linearly interpolated from ``discharge`` (m³ s-1).
    * ``temp``  — constant fallback (``temp_const``, default 15 °C).
    * ``salt``  — constant fallback (``salt_const``, default 0 PSU;
      0 is correct for both river and sewer outlets in TB-FVCOM).

    Parameters
    ----------
    nc_path : Path or str
        Path to the river_dl NetCDF.
    scale : float, optional
        Multiplicative factor applied to ``discharge`` before returning.
        Defaults to 1.0; production should leave it at 1.0 and apply
        runtime calibration in NML. Exposed only for static studies that
        bypass the run-script wrapper.
    temp_const : float, optional
        Constant temperature [°C] returned for ``temp``. Default 15.
    salt_const : float, optional
        Constant salinity [PSU] returned for ``salt``. Default 0.
    fill_nan : bool, optional
        If True (default), source-side NaN values (qc_flag = 2 "remaining
        gaps" in river_dl Phase AN/AO) are dropped before time interp, so
        the FVCOM-side timeline never carries NaN flux. FVCOM crashes on
        NaN forcing, so leaving this on is mandatory for production runs;
        set False only for diagnostic introspection where preserving raw
        NaN is desired.

    Raises
    ------
    FileNotFoundError
        If *nc_path* does not exist.
    KeyError
        If required NetCDF variables (``discharge``, ``time``) are missing.
    """

    def __init__(
        self,
        nc_path: Path | str,
        *,
        scale: float = 1.0,
        temp_const: float = 15.0,
        salt_const: float = 0.0,
        fill_nan: bool = True,
        source_tz: str | None = None,
        time_support: str | None = None,
    ) -> None:
        self._path = Path(nc_path)
        if not self._path.exists():
            raise FileNotFoundError(f"river_dl NetCDF not found: {self._path}")

        self._ds: xr.Dataset = xr.open_dataset(self._path)
        for v in _REQUIRED_VARS:
            if v not in self._ds.variables:
                raise KeyError(
                    f"{self._path.name}: missing required variable {v!r}; "
                    f"available: {sorted(self._ds.variables)}"
                )

        self._scale = float(scale)
        self._temp_const = float(temp_const)
        self._salt_const = float(salt_const)
        self._fill_nan = bool(fill_nan)

        self._source_tz = self._resolve_source_tz(source_tz)
        self._daily_mean = self._is_daily_mean(time_support)
        # Source instants as NAIVE UTC (the generator's timeline is converted the same way in _interp_flux).
        self._src_time = self._to_naive_utc(pd.DatetimeIndex(self._ds["time"].values))

    # ------------------------------------------------------------------
    # Time zone
    # ------------------------------------------------------------------
    LEGACY_UTC_DIRS: Final[tuple[str, ...]] = ("LegacyF04v3",)

    ZONE_ATTRS: Final[tuple[str, ...]] = ("time_zone", "timezone", "tz")
    JST_ALIASES: Final[tuple[str, ...]] = (
        "JST",
        "UTC+9",
        "UTC+09",
        "UTC+09:00",
        "+09:00",
        "ASIA/TOKYO",
    )
    UTC_ALIASES: Final[tuple[str, ...]] = (
        "UTC",
        "GMT",
        "Z",
        "UTC+0",
        "UTC+00:00",
        "+00:00",
        "ETC/UTC",
    )

    def _canon(self, tz: str) -> str:
        up = str(tz).strip().upper()
        if up in self.JST_ALIASES:
            return "Asia/Tokyo"
        if up in self.UTC_ALIASES:
            return "UTC"
        return str(tz).strip()

    def _resolve_source_tz(self, explicit: str | None) -> str:
        """Civil zone of the product (whose calendar defines its days). Order: explicit argument; a zone attribute on
        ``time`` / the file (conflicting attributes raise); ``UTC`` for LegacyF04v3; the river_dl contract (JST).
        Separately, ``self._decoded_utc`` records whether the CF units carry an offset -- then xarray has already
        decoded the values to UTC instants and they are never localised again; an offset that disagrees with the
        product zone raises (review job 6979769, amendment 1)."""
        units = str(
            self._ds["time"].encoding.get(
                "units", self._ds["time"].attrs.get("units", "")
            )
        ).strip()
        m = re.search(r"(Z|[+-]\d{2}:?\d{2}|\bUTC)$", units)
        self._decoded_utc = bool(m)
        declared = {}
        for where, attrs in (
            ("time", self._ds["time"].attrs),
            ("file", self._ds.attrs),
        ):
            for key in self.ZONE_ATTRS:
                if attrs.get(key):
                    declared[f"{where}:{key}"] = self._canon(attrs[key])
        if len(set(declared.values())) > 1:
            raise ValueError(f"{self._path}: conflicting zone declarations {declared}")
        if explicit:
            zone, self._zone_basis = self._canon(explicit), "explicit"
            if declared and zone != next(iter(declared.values())):
                raise ValueError(
                    f"{self._path}: source_tz {explicit!r} contradicts the file's {declared}"
                )
        elif declared:
            zone, self._zone_basis = next(iter(declared.values())), "attribute"
        elif self._path.parent.name in self.LEGACY_UTC_DIRS:
            zone, self._zone_basis = "UTC", "LegacyF04v3"
        elif self._decoded_utc:
            zone, self._zone_basis = "UTC", "units offset"
        else:
            zone, self._zone_basis = "Asia/Tokyo", "river_dl contract"
        if self._decoded_utc and m.group(1) not in ("Z", "UTC"):
            off = m.group(1).replace(":", "")
            want = (
                pd.Timestamp(self._ds["time"].values[0])
                .tz_localize("UTC")
                .tz_convert(zone)
                .strftime("%z")
            )
            if self._zone_basis in ("explicit", "attribute") and off != want:
                raise ValueError(
                    f"{self._path}: units offset {m.group(1)} contradicts zone {zone!r} ({want})"
                )
        return zone

    def _to_naive_utc(self, idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        # values -> aware instants: already UTC when the units carried an offset, else wall clock in the source zone
        if idx.tz is None:
            idx = idx.tz_localize("UTC" if self._decoded_utc else self._source_tz)
        if self._daily_mean:
            # the mean over a source-zone calendar day sits at the middle of that day (12:00 source zone)
            idx = idx.tz_convert(self._source_tz).normalize() + pd.Timedelta(hours=12)
        return idx.tz_convert("UTC").tz_localize(None)

    def _is_daily_mean(self, time_support: str | None) -> bool:
        """True for a product whose value is the mean over a source-zone calendar day. Explicit ``time_support``
        (argument or attribute: "daily_mean" / "instant") or ``cell_methods`` "time: mean" decide; otherwise the
        cadence rule (midnight stamps, 1-day step) applies ONLY to products under the river_dl contract. Midnight is
        tested in the SOURCE calendar (a +09:00-encoded JST day decodes to 15:00 UTC); an explicit daily_mean may
        have a single record (review job 6979769, amendment 2)."""
        t = pd.DatetimeIndex(self._ds["time"].values)
        local = t.tz_localize(
            "UTC" if self._decoded_utc else self._source_tz
        ).tz_convert(self._source_tz)
        midnight = bool((local.normalize() == local).all())
        daily_step = len(t) < 2 or bool(
            pd.Series(local[1:] - local[:-1]).median() == pd.Timedelta(days=1)
        )
        sup = (
            time_support
            or self._ds["time"].attrs.get("time_support")
            or self._ds.attrs.get("time_support")
        )
        if sup:
            sup = str(sup).lower()
            if sup not in ("daily_mean", "instant"):
                raise ValueError(
                    f"{self._path}: time_support {sup!r} (expected 'daily_mean' or 'instant')"
                )
            if sup == "daily_mean" and not (midnight and daily_step):
                raise ValueError(
                    f"{self._path}: time_support daily_mean but the stamps are not source-zone midnights"
                )
            return sup == "daily_mean"
        inferred = len(t) >= 2 and midnight and daily_step
        if (
            "time: mean" in str(self._ds["discharge"].attrs.get("cell_methods", ""))
            and inferred
        ):
            return True
        return inferred and self._zone_basis == "river_dl contract"

    @property
    def source_tz(self) -> str:
        """Zone the file's time stamps were interpreted in (see module docstring)."""
        return self._source_tz

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def variables(self) -> tuple[str, ...]:
        return ("flux", "temp", "salt")

    @property
    def river_dl_path(self) -> Path:
        return self._path

    @property
    def attrs(self) -> dict[str, str]:
        """Convenience access to the file's global attributes (str-cast)."""
        return {k: str(v) for k, v in self._ds.attrs.items()}

    # ------------------------------------------------------------------
    # BaseForcingSource interface
    # ------------------------------------------------------------------
    def get_series(  # type: ignore[override]
        self, var_name: str, times: pd.DatetimeIndex
    ) -> NDArray[np.float32]:
        n = pd.DatetimeIndex(times).size
        if var_name == "flux":
            return self._interp_flux(times)
        if var_name == "temp":
            return np.full(n, self._temp_const, dtype=np.float32)
        if var_name == "salt":
            return np.full(n, self._salt_const, dtype=np.float32)
        raise KeyError(
            f"Unsupported variable {var_name!r}; expected one of " f"{self.variables}"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _interp_flux(self, times: pd.DatetimeIndex) -> NDArray[np.float32]:
        target = pd.DatetimeIndex(times)
        if target.tz is not None:
            target = target.tz_convert("UTC").tz_localize(None)

        da = self._ds["discharge"]

        # ``fill_nan`` drops source-side NaN before interp so short gaps
        # (qc_flag = 2 in river_dl Phase AN/AO) are linearly bridged from
        # the surrounding valid samples. Disable only for diagnostics.
        if self._fill_nan:
            da_clean = da.dropna(dim="time")
            src_time_clean = self._to_naive_utc(
                pd.DatetimeIndex(da_clean["time"].values)
            )
        else:
            da_clean = da
            src_time_clean = self._src_time
        # interpolate on the UTC instants, not on the file's own (JST) wall clock
        da_clean = da_clean.assign_coords(time=src_time_clean.values)

        # Skip interpolation when the requested timeline is an exact
        # match of the cleaned source (the common case for hourly FVCOM
        # runs that align with a gap-free river_dl cadence).
        if target.size == src_time_clean.size and bool(
            (target == src_time_clean).all()
        ):
            arr = np.asarray(da_clean.values, dtype=np.float32)
        else:
            arr = np.asarray(
                da_clean.interp(time=target, method="linear").values,
                dtype=np.float32,
            )
            arr = self._hold_edges(
                arr, target, src_time_clean, np.asarray(da_clean.values)
            )
        return arr * np.float32(self._scale)

    # Converting JST stamps to UTC moves the end of a product 9 h earlier: a product that ends 2024-12-31 23:00 JST
    # stops at 14:00 UTC, short of the FVCOM year-end bookend. Targets outside the source coverage by at most this many
    # hours take the nearest valid value (persistence), with a warning; anything further raises.
    EDGE_HOLD_MAX_H: Final[float] = 24.0

    def _hold_edges(self, arr, target, src_t, src_v):
        if not self._fill_nan or len(src_t) == 0:
            return arr
        before, after = target < src_t[0], target > src_t[-1]
        if not (before.any() or after.any()):
            return arr
        gap_h = max(
            (
                ((src_t[0] - target[before].min()) / pd.Timedelta(hours=1))
                if before.any()
                else 0.0
            ),
            (
                ((target[after].max() - src_t[-1]) / pd.Timedelta(hours=1))
                if after.any()
                else 0.0
            ),
        )
        if gap_h > self.EDGE_HOLD_MAX_H:
            raise ValueError(
                f"{self._path}: requested timeline extends {gap_h:.1f} h beyond the source coverage "
                f"({src_t[0]} .. {src_t[-1]} UTC, source zone {self._source_tz}); max edge hold "
                f"{self.EDGE_HOLD_MAX_H} h"
            )
        warnings.warn(
            f"{self._path.name}: {int(before.sum() + after.sum())} target step(s) up to {gap_h:.1f} h outside the "
            f"source coverage (ends {src_t[-1]} UTC); held at the nearest valid value",
            stacklevel=3,
        )
        arr = arr.copy()
        arr[before] = np.float32(src_v[0])
        arr[after] = np.float32(src_v[-1])
        return arr


__all__ = ["RiverDLNetCDFSource"]
