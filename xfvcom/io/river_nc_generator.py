from __future__ import annotations

import tempfile
from datetime import timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, cast

import netCDF4 as nc
import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from xfvcom.io.sources.base import BaseForcingSource
from xfvcom.io.sources.timeseries import TimeSeriesSource

from .base_generator import BaseGenerator
from .rivers_nml_parser import parse_rivers_nml


# ------------------------------------------------------------------
#  Lightweight constant-source (per single variable)
# ------------------------------------------------------------------
class _ScalarConstantSource(BaseForcingSource):
    """Return the same constant value for exactly one variable."""

    def __init__(self, var: str, value: float):
        self._var = var
        self._val = float(value)

    def get_series(self, var_name: str, out_times: np.ndarray) -> np.ndarray:  # type: ignore[override]
        if var_name != self._var:
            raise KeyError(
                f"Unsupported variable {var_name!r} (expected {self._var!r})"
            )
        return np.full(out_times.size, self._val, dtype=float)


# ----------------------------------------------------------------------
# Helper: choose appropriate Source for each variable
# ----------------------------------------------------------------------
def _choose_source(
    var: str,
    const_val: float | None,
    out_times: np.ndarray,
    *,
    river_name: str | None = None,
    ts_map: dict[str, str],
    const_map: dict[str, dict[str, float]],
    interp_opts: dict[str, str | bool],
) -> BaseForcingSource:
    """
    Return a constant-value source for *var*.

    Priority:
    1. YAML time-series for the given river
    2. YAML constants for the given river
    3. CLI constant (--flux/--temp/--salt)
    4. Fallback 0.0
    """
    # 1) YAML time-series (skip if river_name is None)
    if river_name and river_name in ts_map:
        file_part = ts_map[river_name]
        path, _, vars_part = file_part.partition(":")
        vars_list = [v.strip() for v in vars_part.split(",")] if vars_part else []
        if vars_list and var not in vars_list:
            # Column is not present in file → fall back to next priority
            pass
        else:
            return TimeSeriesSource(
                Path(path),
                river_name=river_name,
                interp_method=str(interp_opts.get("method", "linear")),
            )

    # ------------------------------------------------------------------
    # GLOBAL time-series fallback  (applies to every river)
    # ------------------------------------------------------------------
    if "GLOBAL" in ts_map:
        file_part = ts_map["GLOBAL"]
        path, _, vars_part = file_part.partition(":")
        vars_list = [v.strip() for v in vars_part.split(",")] if vars_part else []

        # If vars_part is empty → file provides *all* variables
        # If vars_part is given → use only when it contains *var*
        if (not vars_list) or (var in vars_list):
            return TimeSeriesSource(
                Path(path),
                river_name=None,  # global
                interp_method=str(interp_opts.get("method", "linear")),
            )

    # 2) YAML const priority
    if river_name and river_name in const_map and var in const_map[river_name]:
        return _ScalarConstantSource(var, const_map[river_name][var])

    # 2-bis) YAML/CLI GLOBAL const (applies to all rivers)
    if "GLOBAL" in const_map and var in const_map["GLOBAL"]:
        return _ScalarConstantSource(var, const_map["GLOBAL"][var])

    # 3) CLI constant
    if const_val is not None:
        return _ScalarConstantSource(var, const_val)

    # 4) Fallback 0.0
    return _ScalarConstantSource(var, 0.0)


class RiverNetCDFGenerator(BaseGenerator):
    """Generate NetCDF-4 river forcing file from NML and constant sources."""

    def __init__(
        self,
        nml_path: Path,
        start: str,
        end: str,
        dt_seconds: int,
        default_flux: float = 0.0,
        default_temp: float = 20.0,
        default_salt: float = 0.0,
        ts_specs: list[str] | None = None,  # ← NEW (from --ts)
        const_specs: list[str] | None = None,  # ← NEW (from --const)
        config: Path | None = None,
    ) -> None:
        # store path & call BaseGenerator
        self.nml_path = nml_path
        super().__init__(nml_path)

        # ------------------------------------------------------------
        # CLI 既定値で一度初期化しておく（YAML があれば後で上書き）
        # ------------------------------------------------------------
        cfg_flux = default_flux
        cfg_temp = default_temp
        cfg_salt = default_salt

        self._ts_map: dict[str, str] = {}
        self._const_map: dict[str, dict[str, float]] = {}
        # interpolation options (currently: just “method”)
        self._interp_opts: dict[str, Any] = {
            "method": "linear",
        }

        self.start = pd.Timestamp(start, tz="UTC")
        self.end = pd.Timestamp(end, tz="UTC")
        self.dt = dt_seconds

        # timeline used by render()
        self.timeline = pd.date_range(
            self.start,
            self.end,
            freq=f"{self.dt}s",
            inclusive="both",
            tz="UTC",
        )

        # ------------------------------------------------------------
        # Initial river list from NML (fallback ["river1"] if missing)
        # ------------------------------------------------------------
        self.rivers: list[str] = self._extract_river_names(self.nml_path)

        if config:
            with Path(config).open("r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)

            # --------------------------------------------------------
            # Global defaults (cast to float – YAML may give str)
            # --------------------------------------------------------
            defs = cfg.get("defaults", {})
            if "flux" in defs:
                cfg_flux = float(defs["flux"])
            if "temp" in defs:
                cfg_temp = float(defs["temp"])
            if "salt" in defs:
                cfg_salt = float(defs["salt"])

            # Each river
            for rv in cfg.get("rivers", []):
                name = rv["name"]
                if "ts" in rv:
                    self._ts_map[name] = rv["ts"]
                if "const" in rv:
                    # flux/temp/salt を部分的に持っていても良い
                    self._const_map[name] = {
                        k: float(v)
                        for k, v in (
                            rv["const"]
                            if isinstance(rv["const"], dict)
                            else _parse_keyvals(rv["const"])
                        ).items()
                    }

            # Interpolation
            self._interp_opts.update(cfg.get("interp", {}))

        # ------------------------------------------------------------------
        # Merge CLI --ts / --const if they exist (lower precedence than YAML)
        # ------------------------------------------------------------------
        # --- merge CLI --ts (lower precedence than YAML) -----------------
        if ts_specs:
            for rv, spec in _parse_ts_spec(ts_specs).items():
                self._ts_map.setdefault(rv, spec)

        # --- merge CLI --const ------------------------------------------
        if const_specs:
            cli_map = _parse_const_spec(const_specs)
            for rv, kv in cli_map.items():
                merged = {**self._const_map.get(rv, {}), **kv}
                self._const_map[rv] = merged

        # ------------------------------------------------------------
        # Ensure rivers list contains every key in ts/const maps
        # (after merges, before render)
        # ------------------------------------------------------------
        # Build final river list:
        #   1) every river appearing in ts/const maps (in that order)
        #   2) any names read from NML that are *not* duplicates
        map_order = [rv for rv in self._ts_map.keys() if rv != "GLOBAL"]
        map_order += [
            rv for rv in self._const_map.keys() if rv not in ("GLOBAL", *map_order)
        ]

        nml_names = self._extract_river_names(self.nml_path)
        self.rivers = map_order or nml_names  # prefer explicit list

        # If map_order existed, append NML-only names (avoid duplicates)
        for name in nml_names:
            if name not in self.rivers:
                self.rivers.append(name)

        # ------------------------------------------------------------
        # Finalise default constants *after* YAML and CLI overrides
        # ------------------------------------------------------------
        self.default_flux = cfg_flux
        self.default_temp = cfg_temp
        self.default_salt = cfg_salt

    # --------------------------------------------------------------- #
    # Abstract-method overrides                                      #
    # --------------------------------------------------------------- #
    # --------------------------- helpers ---------------------------- #
    @staticmethod
    def _to_mjd(times: pd.DatetimeIndex) -> NDArray[np.float64]:
        """Return Modified Julian Day as a NumPy array (float64)."""
        mjd0 = pd.Timestamp("1858-11-17T00:00:00Z")
        return ((times - mjd0) / pd.Timedelta("1D")).to_numpy("f8")

    @staticmethod
    def _times_char(times: pd.DatetimeIndex) -> np.ndarray:
        """Return Times char array (time, DateStrLen=26)."""
        strs = times.strftime("%Y-%m-%dT%H:%M:%S.000000")
        return np.asarray([list(s.ljust(26)) for s in strs], dtype="S1")

    # ------------------------------------------------------------------
    # Helper: extract river names from a minimal NML file
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_river_names(path: Path) -> list[str]:
        """
        Very lightweight parser to pull `river_name = ...` entries out of an
        NML file.  Falls back to a single placeholder if the file contains no
        names.  Raises ``FileNotFoundError`` when *path* does not exist.
        """
        import re

        names: list[str] = []
        key = re.compile(r"river_name", re.IGNORECASE)

        if not path.exists():
            raise FileNotFoundError(path)

        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not key.search(line):
                    continue

                # drop comments after "!"
                line = line.split("!")[0]
                if "=" not in line:
                    continue

                raw = line.split("=", 1)[1]
                for token in raw.split(","):
                    token = token.strip().strip('"').strip("'")
                    if token:
                        names.append(token)

                # assume all names are on one line in minimal NML
                if names:
                    break

        return names or ["river1"]

    def load(self) -> None:
        """Parse rivers.nml and build timeline."""
        self.rivers = parse_rivers_nml(self.source)
        self.timeline = pd.date_range(
            self.start, self.end, freq=f"{self.dt}s", inclusive="both", tz="UTC"
        )

    def validate(self) -> None:
        if not self.rivers:
            raise ValueError("No river entries found in NML.")

    # ------------------------------------------------------------------
    # Low-level NetCDF writer
    # ------------------------------------------------------------------
    def render(self) -> bytes:
        """
        Build a river-forcing NetCDF-4 file that matches the original
        MATLAB/FVCOM layout *byte-for-byte*, and return its binary content.
        """
        # ---- 1. Pre-compute helper arrays --------------------------------
        nr = len(self.rivers)
        nt = self.timeline.size

        # Modified Julian Day (float32) and its split parts
        time_mjd_f32: NDArray[np.float32] = self._to_mjd(self.timeline).astype("f4")
        itime_i32: NDArray[np.int32] = time_mjd_f32.astype("i4")
        itime2_i32: NDArray[np.int32] = (
            (time_mjd_f32 - itime_i32) * 86_400_000
        ).astype("i4")

        # char arrays
        times_char = self._times_char(self.timeline)  # (nt, 26) S1
        rname_char = np.asarray(  # (nr, 80) S1
            [list(name.ljust(80)) for name in self.rivers], dtype="S1"
        )

        # ------------------------------------------------------------------
        # Build (nt, nr) matrices – loop over rivers
        # ------------------------------------------------------------------
        flux_f4: NDArray[np.float32] = np.empty((nt, nr), dtype="f4")
        temp_f4: NDArray[np.float32] = np.empty((nt, nr), dtype="f4")
        salt_f4: NDArray[np.float32] = np.empty((nt, nr), dtype="f4")

        for j, river_name in enumerate(self.rivers):
            # pick source objects (priority: ts → const → CLI default → 0)
            src_flux = _choose_source(
                "flux",
                self.default_flux,
                self.timeline.to_numpy(),
                river_name=river_name,
                ts_map=self._ts_map,
                const_map=self._const_map,
                interp_opts=self._interp_opts,
            )
            src_temp = _choose_source(
                "temp",
                self.default_temp,
                self.timeline.to_numpy(),
                river_name=river_name,
                ts_map=self._ts_map,
                const_map=self._const_map,
                interp_opts=self._interp_opts,
            )
            src_salt = _choose_source(
                "salt",
                self.default_salt,
                self.timeline.to_numpy(),
                river_name=river_name,
                ts_map=self._ts_map,
                const_map=self._const_map,
                interp_opts=self._interp_opts,
            )

            # write column j
            # mypy expects the second argument to be a `pd.DatetimeIndex`
            flux_f4[:, j] = src_flux.get_series("flux", self.timeline)
            temp_f4[:, j] = src_temp.get_series("temp", self.timeline)
            salt_f4[:, j] = src_salt.get_series("salt", self.timeline)

        # ---- 2. Write with netCDF4 - low level ---------------------------
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        with nc.Dataset(tmp_path, "w", format="NETCDF4_CLASSIC") as ds:
            # (a) dimensions — order is significant
            ds.createDimension("namelen", 80)
            ds.createDimension("rivers", nr)
            ds.createDimension("time", None)
            ds.createDimension("DateStrLen", 26)

            # (b) global attributes
            ds.type = "FVCOM RIVER FORCING FILE"
            ds.title = "Constant river forcing (prototype)"
            ds.history = "generated by xfvcom"
            ds.info = "flux in m3/s, temp in degC, salinity in PSU"

            # (c) coordinate & helper variables  --------------------------
            v_rnames = ds.createVariable("river_names", "S1", ("rivers", "namelen"))
            v_rnames[:, :] = rname_char

            v_time = ds.createVariable("time", "f4", ("time",))
            v_time[:] = time_mjd_f32
            v_time.long_name = "time"
            v_time.units = "days since 1858-11-17 00:00:00"
            v_time.format = "modified julian day (MJD)"
            v_time.time_zone = "UTC"

            v_itime = ds.createVariable("Itime", "i4", ("time",))
            v_itime[:] = itime_i32
            v_itime.units = v_time.units
            v_itime.format = v_time.format
            v_itime.time_zone = "UTC"

            v_itime2 = ds.createVariable("Itime2", "i4", ("time",))
            v_itime2[:] = itime2_i32
            v_itime2.units = "msec since 00:00:00"
            v_itime2.time_zone = "UTC"

            v_times = ds.createVariable("Times", "S1", ("time", "DateStrLen"))
            v_times[:, :] = times_char
            v_times.time_zone = "UTC"

            # (d) data variables  ----------------------------------------
            def _make(name: str, data: NDArray, long: str, unit: str) -> None:
                var = ds.createVariable(name, "f4", ("time", "rivers"), fill_value=None)
                var[:, :] = data
                var.long_name = long
                var.units = unit

            _make("river_flux", flux_f4, "river runoff volume flux", "m^3s^-1")
            _make("river_temp", temp_f4, "river runoff temperature", "Celsius")
            _make("river_salt", salt_f4, "river runoff salinity", "PSU")

        # read back binary and delete temp file
        binary = tmp_path.read_bytes()
        tmp_path.unlink(missing_ok=True)
        return binary


def _parse_ts_spec(tokens: list[str]) -> dict[str, str]:
    """
    Parse CLI --ts tokens.

    Examples
    --------
    Arakawa=rivers.csv:flux,temp  -> {"Arakawa": "rivers.csv:flux,temp"}
    rivers.csv                    -> {"GLOBAL": "rivers.csv"}
    """
    out: dict[str, str] = {}
    for tok in tokens:
        if "=" in tok:
            river, spec = tok.split("=", 1)
            out[river.strip()] = spec.strip()
        else:
            out["GLOBAL"] = tok.strip()
    return out


def _parse_keyvals(expr: str) -> dict[str, float]:
    """
    Parse "VAR=VAL[,VAR=VAL…]" strings into a dict with float values.
    """
    out: dict[str, float] = {}
    for tok in expr.split(","):
        if "=" not in tok:
            raise ValueError(f"Invalid key=value pair: {tok!r}")
        k, v = tok.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def _parse_const_spec(tokens: list[str]) -> dict[str, dict[str, float]]:
    """
    Parse CLI --const tokens.

    Examples
    --------
    Sumidagawa.flux=130  -> {"Sumidagawa": {"flux": 130}}
    temp=15              -> {"GLOBAL": {"temp": 15}}
    """
    out: dict[str, dict[str, float]] = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"Invalid const spec: {tok!r}")
        lhs, val = tok.split("=", 1)
        if "." in lhs:
            river, var = lhs.split(".", 1)
        else:
            river, var = "GLOBAL", lhs
        out.setdefault(river.strip(), {})[var.strip()] = float(val)
    return out
