"""river_dl-based discharge source for FVCOM river/sewer forcing.

Reads a per-river or per-plant ``discharge_hourly.nc`` produced by the
``river_dl`` toolkit (e.g. ``$DATA_DIR/river_dl/discharge/<River>/<Station>/``
or ``$DATA_DIR/river_dl/sewage/<Plant>/``) and supplies the ``flux`` series
to :class:`xfvcom.io.river_nc_generator.RiverNetCDFGenerator`.

Expected NetCDF schema (river_dl Phase AM/AN/AO+, 2026-04-21):

================  =====================================================
``time(time)``    int64, ``"hours since YYYY-MM-DD HH:MM:SS"`` (proleptic)
``discharge(time)`` float, units ``m3/s``
``qc_flag(time)`` byte (optional, ignored here; available for audit)
================  =====================================================

The source intentionally returns *raw* observed/estimated discharge — no
Optuna #123 per-river scaling or Trial #16 seasonal modulation. Those are
runtime knobs applied via ``RIVERS_NAMELIST.RIVER_FLUX_SCALE_LOCAL`` and
the launcher's seasonal-cosine wrapper (see
``TB-FVCOM/hydro/docs/bc_construction_protocol.md`` rule R2).
"""

from __future__ import annotations

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

        # Cache source time as naive datetime64 for fast comparison.
        self._src_time = pd.DatetimeIndex(self._ds["time"].values)
        if self._src_time.tz is not None:
            self._src_time = self._src_time.tz_localize(None)

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
            src_time_clean = pd.DatetimeIndex(da_clean["time"].values)
            if src_time_clean.tz is not None:
                src_time_clean = src_time_clean.tz_localize(None)
        else:
            da_clean = da
            src_time_clean = self._src_time

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
        return arr * np.float32(self._scale)


__all__ = ["RiverDLNetCDFSource"]
