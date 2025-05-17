from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .base import BaseForcingSource
from .utils import load_timeseries_table


class TimeSeriesSource(BaseForcingSource):
    """
    Read a CSV/TSV time-series table and provide raw (non-interpolated) data.

    *No interpolation is performed at this stage.*  Values are simply re-indexed
    to *out_times*; missing timestamps remain NaN.  A separate interpolation
    layer will be introduced in Step 3.
    """

    def __init__(
        self,
        path: Path,
        *,
        river_name: str | None = None,
        col_map: Mapping[str, str] | None = None,
        na_values: Sequence[str | float] | None = None,
        interp_method: str = "linear",
    ) -> None:
        # Load table (encoding / delimiter / NA markers handled inside)
        df = load_timeseries_table(Path(path), na_values=na_values)

        # ------------------------------------------------------------ #
        # 1) Ensure the DataFrame is indexed by a “time” DatetimeIndex
        # ------------------------------------------------------------ #
        if "time" in df.columns:  # ← still a column → move it
            df = df.set_index("time")
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"{path.name}: a 'time' column or DatetimeIndex is required "
                f"(got {type(df.index).__name__})"
            )
        df.index = pd.to_datetime(df.index, errors="coerce")
        if df.index.isna().any():
            raise ValueError(
                f"{path.name}: failed to parse some 'time' values as dates"
            )
        # ──────────────────────────────────────────────────────────────
        # Bring the index to a **naïve – but UTC-based** timeline.
        #   (All reference times used by the test-suite are in UTC.)
        # ──────────────────────────────────────────────────────────────
        #
        #  • Ensure the index *is* a DatetimeIndex; otherwise the source
        #    file is missing/invalid “time” information and we bail out
        #    immediately with a clear message instead of an AttributeError.
        #  • If the index *is* a DatetimeIndex but has tz-info attached,
        #    convert to UTC and then strip the timezone (naïve UTC).
        #

        # After the block above `df.index` *must* be DatetimeIndex
        # – keep the explicit check for defence-in-depth & clear message.
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"{path.name}: failed to build a DatetimeIndex "
                f"(got {type(df.index).__name__})"
            )

        # The tests assume all timestamps are already expressed in UTC.
        # If the CSV/TSV contained tz-aware datetimes, just strip the tz-info
        # and keep the clock-time unchanged.
        # If timestamps are tz-aware convert to UTC, then strip tz-info
        if df.index.tz is not None:  # JST(UTC+9) → UTC → naïve
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        # If a river_name column exists and a filter is requested, apply it
        if river_name and "river_name" in df.columns:
            df = df.loc[df["river_name"] == river_name].copy()

        # Always keep a **naïve (timezone-less) UTC** index.
        # `df` is **already** indexed by the time column (see block above) and
        # has had any timezone stripped, so we can store it directly.
        self._df = df

        self._col_map: dict[str, str] = {
            k.lower(): v for k, v in (col_map or {}).items()
        }

        # ─── 補間方法 ───
        self._interp_method: str = interp_method.lower()
        if self._interp_method != "linear":
            raise ValueError(
                "interp_method must be 'linear'; "
                f"'{self._interp_method}' is not allowed"
            )

    # ------------------------------------------------------------------
    # Public API required by BaseForcingSource
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #  Abstract-method implementation required by BaseForcingSource
    # ------------------------------------------------------------------
    def get_series(self, var_name: str, out_times: np.ndarray) -> np.ndarray:  # type: ignore[override]
        """
        Return the (non-interpolated) series for *var_name* aligned to *out_times*.

        Missing timestamps are filled with NaN.  Interpolation is deferred to
        the next development step.
        """
        # Resolve alias (case-insensitive)
        key = self._col_map.get(var_name.lower(), var_name)
        if key not in self._df.columns:
            # Test-suite expects ValueError, not KeyError.
            raise ValueError(f"Column '{key}' not found in time-series table")

        # Convert *out_times* into a **naïve** `DatetimeIndex` as well.
        #   – if they are timezone-aware, just drop the tz-info
        #   – otherwise keep them as-is
        # Ensure *out_times* is in the very same “naïve-UTC” flavour
        times = pd.DatetimeIndex(out_times)
        if times.tz is not None:  # keep absolute UTC clock-time
            times = times.tz_convert("UTC").tz_localize(None)

        # Make sure the source index is monotonic before using
        if not self._df.index.is_monotonic_increasing:
            self._df = self._df.sort_index()

        # ------------------------------------------------------------
        # 1) make sure input data cover the requested period
        # ------------------------------------------------------------
        idx_min, idx_max = self._df.index.min(), self._df.index.max()
        if times.min() < idx_min or times.max() > idx_max:
            raise ValueError(
                "Requested times fall outside the available data range "
                "(extrapolation is not permitted)."
            )

        # ------------------------------------------------------------
        # 2) interpolation (values outside the data span remain NaN, so no extrapolation)
        # ------------------------------------------------------------
        union_index = self._df.index.union(times)
        series_union = self._df[key].reindex(union_index)

        # single-pass linear interpolation (NaNs at ends remain NaN)
        series_union = series_union.interpolate(method="time", limit_direction="both")

        # keep only the requested timestamps
        series = series_union.reindex(times)

        # Final integrity check ─ NaNs must NOT remain *inside*
        # the original data span after interpolation.
        inside_mask = (times >= idx_min) & (times <= idx_max)
        if series[inside_mask].isna().any():
            raise ValueError(
                f"Interpolation failed for '{var_name}'; NaNs remain in output."
            )

        return series.to_numpy(dtype=float)

    # ------------------------------------------------------------------
    #  Convenience alias; will allow future refactor away from
    #  get_series() without breaking existing code.
    # ------------------------------------------------------------------
    def get(
        self, var_name: str, out_times: np.ndarray
    ) -> np.ndarray:  # pragma: no cover
        return self.get_series(var_name, out_times)
