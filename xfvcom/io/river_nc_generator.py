from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping

import netCDF4 as nc
import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray

from xfvcom.io.sources.base import BaseForcingSource
from xfvcom.io.sources.river_dl import RiverDLNetCDFSource
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
    data_tz: str,
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
                input_tz=data_tz,
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
                input_tz=data_tz,
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
        *,
        start_tz: str = "UTC",
        data_tz: str = "Asia/Tokyo",
        river_dl_map: Mapping[str, Mapping[str, Any]] | None = None,
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

        t0 = pd.Timestamp(start)
        t1 = pd.Timestamp(end)
        if t0.tzinfo is None:
            t0 = t0.tz_localize(start_tz)
        if t1.tzinfo is None:
            t1 = t1.tz_localize(start_tz)
        self.start = t0.tz_convert("UTC")
        self.end = t1.tz_convert("UTC")
        self.dt = dt_seconds
        self._data_tz = data_tz

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
        # river_dl per-river NetCDF sources (constructed once; reused
        # for every requested variable on that river).
        #
        # An entry may declare ``kind: constant`` in lieu of ``source:``
        # to represent a source with no upstream NetCDF (e.g. a sewer
        # plant whose discharge is approximated by a fixed annual mean
        # while a real observation feed is unavailable).  Constant
        # entries are stored separately in ``self._constant_sources``
        # and consumed by the render() loop before the river_dl path.
        # ------------------------------------------------------------
        self._river_dl_sources: dict[str, RiverDLNetCDFSource] = {}
        self._constant_sources: dict[str, dict[str, float]] = {}
        # Schema-v3 per-river temperature source.  Each value is a dict
        # already validated upstream by _validate_temp_source.  When
        # populated, render() computes the river_temp column from this
        # spec and ignores any temp_const that a RiverDLNetCDFSource was
        # constructed with.  Schema v3 makes this dict mandatory for
        # every river_dl_map entry; legacy callers (kind: constant or
        # kind: river_dl with a constant 'temp:' field) still work for
        # backward compat in the generator API itself.
        self._temp_sources: dict[str, dict[str, Any]] = {}
        if river_dl_map:
            for name, entry in river_dl_map.items():
                kind = str(entry.get("kind", "river_dl"))
                if kind == "constant":
                    if "source" in entry:
                        raise ValueError(
                            f"river_dl_map entry {name!r} declares "
                            f"kind: constant but also carries a "
                            f"'source' key; remove one or the other"
                        )
                    if "flux" not in entry:
                        raise ValueError(
                            f"river_dl_map entry {name!r} declares "
                            f"kind: constant but is missing required "
                            f"key 'flux'"
                        )
                    if "scale" in entry:
                        raise ValueError(
                            f"river_dl_map entry {name!r} declares "
                            f"kind: constant; 'scale' is meaningless "
                            f"for a constant source (the value is "
                            f"already in physical units)"
                        )
                    self._constant_sources[name] = {
                        "flux": float(entry["flux"]),
                        "temp": float(entry.get("temp", cfg_temp)),
                        "salt": float(entry.get("salt", cfg_salt)),
                    }
                elif kind == "river_dl":
                    if "source" not in entry:
                        raise ValueError(
                            f"river_dl_map entry {name!r} missing "
                            f"required key 'source' (path to "
                            f"discharge_hourly.nc)"
                        )
                    self._river_dl_sources[name] = RiverDLNetCDFSource(
                        nc_path=Path(entry["source"]),
                        scale=float(entry.get("scale", 1.0)),
                        temp_const=float(entry.get("temp", cfg_temp)),
                        salt_const=float(entry.get("salt", cfg_salt)),
                    )
                else:
                    raise ValueError(
                        f"river_dl_map entry {name!r} has unsupported "
                        f"kind: {kind!r} (expected 'river_dl' or "
                        f"'constant')"
                    )
                if "temp_source" in entry:
                    ts = entry["temp_source"]
                    if not isinstance(ts, dict) or "kind" not in ts:
                        raise ValueError(
                            f"river_dl_map entry {name!r} temp_source must "
                            f"be a dict with a 'kind' key"
                        )
                    self._temp_sources[name] = dict(ts)
            # Ensure the rivers list contains every named entry,
            # preserving the order in which they were supplied (mirrors
            # the existing ts_map / const_map merge logic above).
            for name in river_dl_map.keys():
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
        delta = times - mjd0
        seconds = np.asarray(delta.total_seconds(), dtype=np.float64)
        return seconds / 86400.0

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
    # Schema-v3 temp_source evaluation
    # ------------------------------------------------------------------
    def _evaluate_temp_source(
        self, river_name: str, spec: dict[str, Any]
    ) -> NDArray[np.float32]:
        """Return a (nt,) float32 array of river_temp values for the
        given river, computed from the temp_source spec dict.

        Supported kinds:
          * ``air_regression``: open the metforce NC(s) per requested
            year, look up hourly T2 at the (air_lat, air_lon) nearest
            grid cell, apply ``T_water = slope * T_air + intercept``,
            optional clip to [min_temp, max_temp].
          * ``monthly_climatology``: index ``monthly_means[t.month-1]``
            for each timestep in the timeline.
        """
        kind = str(spec["kind"])
        if kind == "monthly_climatology":
            means = np.asarray(spec["monthly_means"], dtype=np.float32)
            months = np.asarray(self.timeline.month, dtype=np.int32) - 1
            return means[months].astype(np.float32)
        if kind == "air_regression":
            return self._evaluate_air_regression(river_name, spec)
        raise ValueError(
            f"river {river_name!r}: temp_source.kind={kind!r} not supported"
        )

    def _evaluate_air_regression(
        self, river_name: str, spec: dict[str, Any]
    ) -> NDArray[np.float32]:
        """Implement ``T_water = slope * T_air + intercept`` at hourly
        cadence from a metforce-style ``fvcom_forcing_<year>.nc`` archive.

        The template ``spec["air_nc_template"]`` substitutes ``{year}``
        from the time axis; multi-year ranges open each year's NC and
        concatenate on time.
        """
        import netCDF4 as nc4

        template = str(spec["air_nc_template"])
        var = str(spec["air_var"])
        air_lat = float(spec["air_lat"])
        air_lon = float(spec["air_lon"])
        slope = float(spec["slope"])
        intercept = float(spec["intercept"])

        years = sorted(set(self.timeline.year))
        # Per-year segment: open NC, find nearest cell, build a pd.Series.
        series_parts: list[pd.Series] = []
        for year in years:
            path = Path(template.format(year=year))
            if not path.exists():
                raise FileNotFoundError(
                    f"river {river_name!r} air_regression: metforce NC "
                    f"not found for year {year}: {path}"
                )
            ds = nc4.Dataset(path)
            try:
                lats = ds.variables["lat"][:].astype(float)
                lons = ds.variables["lon"][:].astype(float)
                ilat = int(np.argmin(np.abs(lats - air_lat)))
                ilon = int(np.argmin(np.abs(lons - air_lon)))
                v = ds.variables[var]
                arr = np.asarray(v[:, ilat, ilon], dtype=np.float64)
                tv = ds.variables["time"]
                t_idx = pd.DatetimeIndex(
                    nc4.num2date(tv[:], tv.units, only_use_cftime_datetimes=False)
                )
            finally:
                ds.close()
            # Make timezone-aware UTC for compatibility with self.timeline.
            if t_idx.tz is None:
                t_idx = t_idx.tz_localize("UTC")
            series_parts.append(pd.Series(arr, index=t_idx))

        if not series_parts:
            raise RuntimeError(
                f"river {river_name!r}: no metforce NC opened for " f"years={years}"
            )
        air_series = pd.concat(series_parts).sort_index()
        # Reindex onto the timeline; linear interp where needed (the
        # metforce hourly axis is the same hourly cadence we use here,
        # so this is mostly a no-op except at endpoints).
        air_on_timeline = air_series.reindex(self.timeline).interpolate(
            method="time", limit_direction="both"
        )
        if air_on_timeline.isna().any():
            raise RuntimeError(
                f"river {river_name!r}: metforce {var} has NaN on the "
                f"requested timeline; refusing to silently fill"
            )
        water = slope * air_on_timeline.to_numpy(dtype=np.float64) + intercept
        if "min_temp" in spec:
            water = np.clip(water, float(spec["min_temp"]), None)
        if "max_temp" in spec:
            water = np.clip(water, None, float(spec["max_temp"]))
        return water.astype(np.float32)

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
            # kind: constant entries from river_dl_map: broadcast a
            # fixed value across the time axis for each of the three
            # variables.  Checked before river_dl because a source can
            # only be one or the other (validated at __init__ time).
            if river_name in self._constant_sources:
                const = self._constant_sources[river_name]
                flux_f4[:, j] = np.float32(const["flux"])
                temp_f4[:, j] = np.float32(const["temp"])
                salt_f4[:, j] = np.float32(const["salt"])
                if river_name in self._temp_sources:
                    temp_f4[:, j] = self._evaluate_temp_source(
                        river_name, self._temp_sources[river_name]
                    )
                continue

            # river_dl per-river NetCDF takes priority over ts/const maps
            # when present; one source object supplies all three variables.
            if river_name in self._river_dl_sources:
                rdl = self._river_dl_sources[river_name]
                flux_f4[:, j] = rdl.get_series("flux", self.timeline)
                temp_f4[:, j] = rdl.get_series("temp", self.timeline)
                salt_f4[:, j] = rdl.get_series("salt", self.timeline)
                if river_name in self._temp_sources:
                    temp_f4[:, j] = self._evaluate_temp_source(
                        river_name, self._temp_sources[river_name]
                    )
                continue

            # pick source objects (priority: ts → const → CLI default → 0)
            src_flux = _choose_source(
                "flux",
                self.default_flux,
                self.timeline.to_numpy(),
                river_name=river_name,
                ts_map=self._ts_map,
                const_map=self._const_map,
                interp_opts=self._interp_opts,
                data_tz=self._data_tz,
            )
            src_temp = _choose_source(
                "temp",
                self.default_temp,
                self.timeline.to_numpy(),
                river_name=river_name,
                ts_map=self._ts_map,
                const_map=self._const_map,
                interp_opts=self._interp_opts,
                data_tz=self._data_tz,
            )
            src_salt = _choose_source(
                "salt",
                self.default_salt,
                self.timeline.to_numpy(),
                river_name=river_name,
                ts_map=self._ts_map,
                const_map=self._const_map,
                interp_opts=self._interp_opts,
                data_tz=self._data_tz,
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
