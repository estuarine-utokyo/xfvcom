# -*- coding: utf-8 -*-
"""Validator for FVCOM meteorological forcing NetCDF files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import netCDF4 as nc
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm


@dataclass
class PhysicalBounds:
    """Physical bounds for a meteorological variable.

    Parameters
    ----------
    min_val : float | None
        Minimum physically plausible value (None = no lower bound).
    max_val : float | None
        Maximum physically plausible value (None = no upper bound).
    units : str
        Units for display purposes.
    """

    min_val: float | None
    max_val: float | None
    units: str


@dataclass
class AnomalyRecord:
    """Single anomalous value record.

    Parameters
    ----------
    index : int
        1-based (Fortran) node or element index.
    time_idx : int
        0-based time index.
    time : pd.Timestamp | None
        Actual timestamp (None if time conversion fails).
    lon : float
        Longitude of the location.
    lat : float
        Latitude of the location.
    value : float
        The anomalous value (may be NaN or Inf).
    reason : str
        Reason for anomaly: "nan", "inf", "below_min", "above_max".
    """

    index: int
    time_idx: int
    time: pd.Timestamp | None
    lon: float
    lat: float
    value: float
    reason: str


@dataclass
class NonUniformExample:
    """Example of a non-uniform value at a specific location.

    Parameters
    ----------
    index : int
        1-based (Fortran) node or element index.
    value : float
        The value at this location.
    """

    index: int
    value: float


@dataclass
class UniformityReport:
    """Report of spatial uniformity check for a single variable.

    Parameters
    ----------
    var_name : str
        Name of the variable.
    location_type : Literal["node", "element"]
        Whether indices refer to nodes or elements.
    is_uniform : bool
        True if the variable is spatially uniform at all time steps.
    examples : list[NonUniformExample]
        Up to 3 examples of non-uniform values (only if is_uniform=False).
    reference_value : float | None
        The reference value used for comparison (first location's value).
    first_nonuniform_time_idx : int | None
        0-based time index where non-uniformity was first detected.
    """

    var_name: str
    location_type: Literal["node", "element"]
    is_uniform: bool
    examples: list[NonUniformExample] = field(default_factory=list)
    reference_value: float | None = None
    first_nonuniform_time_idx: int | None = None


@dataclass
class AnomalyReport:
    """Report of anomalous values for a single variable.

    Parameters
    ----------
    var_name : str
        Name of the variable.
    location_type : Literal["node", "element"]
        Whether indices refer to nodes or elements.
    total_nan : int
        Total count of NaN values in the variable.
    total_inf : int
        Total count of Inf values in the variable.
    total_below_min : int
        Total count of values below minimum bound.
    total_above_max : int
        Total count of values above maximum bound.
    anomalies : list[AnomalyRecord]
        Detailed records (may be truncated by max_reports).
    bounds : PhysicalBounds | None
        Physical bounds used for validation.
    """

    var_name: str
    location_type: Literal["node", "element"]
    total_nan: int = 0
    total_inf: int = 0
    total_below_min: int = 0
    total_above_max: int = 0
    anomalies: list[AnomalyRecord] = field(default_factory=list)
    bounds: PhysicalBounds | None = None

    @property
    def total_anomalies(self) -> int:
        """Total count of all anomalies."""
        return (
            self.total_nan
            + self.total_inf
            + self.total_below_min
            + self.total_above_max
        )

    @property
    def is_clean(self) -> bool:
        """True if no anomalies found."""
        return self.total_anomalies == 0


# Default physical bounds for meteorological variables
DEFAULT_BOUNDS: dict[str, PhysicalBounds] = {
    "uwind_speed": PhysicalBounds(-100.0, 100.0, "m/s"),
    "vwind_speed": PhysicalBounds(-100.0, 100.0, "m/s"),
    "air_temperature": PhysicalBounds(-60.0, 60.0, "deg C"),
    "cloud_cover": PhysicalBounds(0.0, 1.0, "-"),
    "short_wave": PhysicalBounds(0.0, 1500.0, "W/m2"),
    "long_wave": PhysicalBounds(0.0, 600.0, "W/m2"),
    "relative_humidity": PhysicalBounds(0.0, 100.0, "%"),
    "air_pressure": PhysicalBounds(850.0, 1100.0, "hPa"),
    "net_heat_flux": PhysicalBounds(-1500.0, 2000.0, "W/m2"),
}

# Variables defined on elements (wind) vs nodes (everything else)
ELEMENT_VARIABLES = {"uwind_speed", "vwind_speed"}


class MetValidator:
    """Validator for FVCOM meteorological forcing files.

    Parameters
    ----------
    nc_path : Path | str
        Path to the meteorological forcing NetCDF file.

    Examples
    --------
    >>> validator = MetValidator("met_forcing.nc")
    >>> reports = validator.validate()
    >>> print(validator.summary(reports))
    """

    def __init__(self, nc_path: Path | str) -> None:
        self.nc_path = Path(nc_path)
        if not self.nc_path.exists():
            raise FileNotFoundError(f"File not found: {self.nc_path}")

        # Load coordinate and time data
        with nc.Dataset(self.nc_path, "r") as ds:
            # Node coordinates
            self.lon: NDArray[np.float64] = np.asarray(ds.variables["lon"][:])
            self.lat: NDArray[np.float64] = np.asarray(ds.variables["lat"][:])
            self.n_nodes = len(self.lon)

            # Element coordinates
            self.lonc: NDArray[np.float64] = np.asarray(ds.variables["lonc"][:])
            self.latc: NDArray[np.float64] = np.asarray(ds.variables["latc"][:])
            self.n_elements = len(self.lonc)

            # Time
            time_var = ds.variables["time"]
            self.time_values: NDArray[np.float64] = np.asarray(time_var[:])
            self.n_times = len(self.time_values)

            # Parse time units for conversion
            time_units = time_var.units  # e.g., "days since 1858-11-17 00:00:00"
            self.timestamps = self._parse_time(self.time_values, time_units)

            # Get list of data variables (exclude coordinates and mesh)
            self.data_vars: list[str] = []
            coord_vars = {
                "x",
                "y",
                "lon",
                "lat",
                "xc",
                "yc",
                "lonc",
                "latc",
                "nv",
                "time",
            }
            for var_name in ds.variables:
                if var_name not in coord_vars:
                    var = ds.variables[var_name]
                    if "time" in var.dimensions:
                        self.data_vars.append(var_name)

    def _parse_time(
        self, time_values: NDArray[np.float64], units: str
    ) -> list[pd.Timestamp | None]:
        """Convert CF time values to pandas Timestamps."""
        try:
            times = nc.num2date(time_values, units, calendar="standard")
            return [pd.Timestamp(t.isoformat()) for t in times]
        except Exception:
            return [None] * len(time_values)

    def _get_coords(
        self, var_name: str, indices: NDArray[np.intp]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return lon/lat for node or element indices (0-based input)."""
        if var_name in ELEMENT_VARIABLES:
            return self.lonc[indices], self.latc[indices]
        else:
            return self.lon[indices], self.lat[indices]

    def _get_location_type(self, var_name: str) -> Literal["node", "element"]:
        """Return whether variable is node-based or element-based."""
        return "element" if var_name in ELEMENT_VARIABLES else "node"

    def validate(
        self,
        check_nan: bool = True,
        check_inf: bool = True,
        check_bounds: bool = True,
        custom_bounds: dict[str, PhysicalBounds] | None = None,
        max_reports_per_var: int = 100,
        variables: list[str] | None = None,
        show_progress: bool = False,
    ) -> list[AnomalyReport]:
        """Run validation and return reports for each variable.

        Parameters
        ----------
        check_nan : bool
            Check for NaN values.
        check_inf : bool
            Check for Inf values.
        check_bounds : bool
            Check for values outside physical bounds.
        custom_bounds : dict[str, PhysicalBounds] | None
            Override default bounds for specific variables.
        max_reports_per_var : int
            Maximum detailed anomaly records per variable (0 = unlimited).
        variables : list[str] | None
            Specific variables to check (None = all data variables).
        show_progress : bool
            Display progress bar for each variable.

        Returns
        -------
        list[AnomalyReport]
            Validation reports for each checked variable.
        """
        bounds_map = DEFAULT_BOUNDS.copy()
        if custom_bounds:
            bounds_map.update(custom_bounds)

        vars_to_check = variables if variables else self.data_vars
        reports: list[AnomalyReport] = []

        with nc.Dataset(self.nc_path, "r") as ds:
            # Create iterator with optional progress bar
            var_iter: list[str] | tqdm[str]
            if show_progress:
                var_iter = tqdm(
                    vars_to_check,
                    desc="Checking variables",
                    unit="var",
                    ncols=80,
                )
            else:
                var_iter = vars_to_check

            for var_name in var_iter:
                if var_name not in ds.variables:
                    continue

                # Update progress bar description with current variable
                if show_progress and isinstance(var_iter, tqdm):
                    var_iter.set_postfix_str(var_name)

                data: NDArray[np.float64] = np.asarray(ds.variables[var_name][:])
                bounds = bounds_map.get(var_name) if check_bounds else None

                report = self._check_variable(
                    var_name,
                    data,
                    bounds,
                    check_nan=check_nan,
                    check_inf=check_inf,
                    max_reports=max_reports_per_var,
                )
                reports.append(report)

        return reports

    def _check_variable(
        self,
        var_name: str,
        data: NDArray[np.float64],
        bounds: PhysicalBounds | None,
        check_nan: bool,
        check_inf: bool,
        max_reports: int,
    ) -> AnomalyReport:
        """Check single variable for anomalies.

        Parameters
        ----------
        var_name : str
            Variable name.
        data : NDArray
            Data array with shape (time, node/element).
        bounds : PhysicalBounds | None
            Physical bounds to check against.
        check_nan : bool
            Whether to check for NaN.
        check_inf : bool
            Whether to check for Inf.
        max_reports : int
            Maximum detailed records (0 = unlimited).

        Returns
        -------
        AnomalyReport
            Validation report for this variable.
        """
        location_type = self._get_location_type(var_name)
        report = AnomalyReport(
            var_name=var_name,
            location_type=location_type,
            bounds=bounds,
        )

        anomaly_records: list[AnomalyRecord] = []

        # Check NaN
        if check_nan:
            nan_mask = np.isnan(data)
            report.total_nan = int(np.sum(nan_mask))
            if report.total_nan > 0:
                self._collect_anomalies(
                    var_name, data, nan_mask, "nan", anomaly_records, max_reports
                )

        # Check Inf
        if check_inf:
            inf_mask = np.isinf(data)
            report.total_inf = int(np.sum(inf_mask))
            if report.total_inf > 0:
                remaining = max_reports - len(anomaly_records) if max_reports > 0 else 0
                if max_reports == 0 or remaining > 0:
                    self._collect_anomalies(
                        var_name,
                        data,
                        inf_mask,
                        "inf",
                        anomaly_records,
                        remaining if max_reports > 0 else 0,
                    )

        # Check bounds
        if bounds is not None:
            # Exclude NaN/Inf for bounds checking
            valid_data = data.copy()
            valid_data[np.isnan(data) | np.isinf(data)] = 0  # Neutral value

            if bounds.min_val is not None:
                below_mask = (data < bounds.min_val) & ~np.isnan(data) & ~np.isinf(data)
                report.total_below_min = int(np.sum(below_mask))
                if report.total_below_min > 0:
                    remaining = (
                        max_reports - len(anomaly_records) if max_reports > 0 else 0
                    )
                    if max_reports == 0 or remaining > 0:
                        self._collect_anomalies(
                            var_name,
                            data,
                            below_mask,
                            "below_min",
                            anomaly_records,
                            remaining if max_reports > 0 else 0,
                        )

            if bounds.max_val is not None:
                above_mask = (data > bounds.max_val) & ~np.isnan(data) & ~np.isinf(data)
                report.total_above_max = int(np.sum(above_mask))
                if report.total_above_max > 0:
                    remaining = (
                        max_reports - len(anomaly_records) if max_reports > 0 else 0
                    )
                    if max_reports == 0 or remaining > 0:
                        self._collect_anomalies(
                            var_name,
                            data,
                            above_mask,
                            "above_max",
                            anomaly_records,
                            remaining if max_reports > 0 else 0,
                        )

        report.anomalies = anomaly_records
        return report

    def _collect_anomalies(
        self,
        var_name: str,
        data: NDArray[np.float64],
        mask: NDArray[np.bool_],
        reason: str,
        records: list[AnomalyRecord],
        max_new: int,
    ) -> None:
        """Collect anomaly records from masked positions.

        Parameters
        ----------
        var_name : str
            Variable name.
        data : NDArray
            Full data array.
        mask : NDArray
            Boolean mask of anomalous positions.
        reason : str
            Reason string for these anomalies.
        records : list[AnomalyRecord]
            List to append records to (modified in place).
        max_new : int
            Maximum new records to add (0 = unlimited).
        """
        # Find indices where mask is True
        time_indices, loc_indices = np.where(mask)

        n_to_add = (
            len(time_indices) if max_new == 0 else min(len(time_indices), max_new)
        )

        if n_to_add == 0:
            return

        # Get coordinates for these locations
        lons, lats = self._get_coords(var_name, loc_indices[:n_to_add])

        for i in range(n_to_add):
            t_idx = int(time_indices[i])
            loc_idx = int(loc_indices[i])

            records.append(
                AnomalyRecord(
                    index=loc_idx + 1,  # Convert to 1-based
                    time_idx=t_idx,
                    time=(
                        self.timestamps[t_idx] if t_idx < len(self.timestamps) else None
                    ),
                    lon=float(lons[i]),
                    lat=float(lats[i]),
                    value=float(data[t_idx, loc_idx]),
                    reason=reason,
                )
            )

    def summary(self, reports: list[AnomalyReport]) -> str:
        """Generate human-readable summary.

        Parameters
        ----------
        reports : list[AnomalyReport]
            Validation reports from validate().

        Returns
        -------
        str
            Formatted summary string.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("Meteorological Forcing Validation Report")
        lines.append("=" * 60)
        lines.append(f"File: {self.nc_path.name}")
        lines.append(f"Path: {self.nc_path}")

        if self.timestamps and self.timestamps[0] and self.timestamps[-1]:
            lines.append(f"Time range: {self.timestamps[0]} to {self.timestamps[-1]}")
        lines.append(
            f"Nodes: {self.n_nodes}, Elements: {self.n_elements}, "
            f"Time steps: {self.n_times}"
        )
        lines.append("")

        total_nan = 0
        total_inf = 0
        total_bounds = 0
        vars_with_issues: list[str] = []

        for report in reports:
            lines.append(f"--- {report.var_name} ({report.location_type}-based) ---")

            # NaN status
            if report.total_nan == 0:
                lines.append("  [OK] No NaN values")
            else:
                lines.append(f"  [ERROR] {report.total_nan} NaN values found")
                total_nan += report.total_nan
                if report.var_name not in vars_with_issues:
                    vars_with_issues.append(report.var_name)

            # Inf status
            if report.total_inf == 0:
                lines.append("  [OK] No Inf values")
            else:
                lines.append(f"  [ERROR] {report.total_inf} Inf values found")
                total_inf += report.total_inf
                if report.var_name not in vars_with_issues:
                    vars_with_issues.append(report.var_name)

            # Bounds status
            if report.bounds is not None:
                bounds_issues = report.total_below_min + report.total_above_max
                if bounds_issues == 0:
                    lines.append(
                        f"  [OK] Values within bounds "
                        f"[{report.bounds.min_val}, {report.bounds.max_val}] "
                        f"{report.bounds.units}"
                    )
                else:
                    if report.total_below_min > 0:
                        lines.append(
                            f"  [WARNING] {report.total_below_min} values "
                            f"below min ({report.bounds.min_val} {report.bounds.units})"
                        )
                    if report.total_above_max > 0:
                        lines.append(
                            f"  [WARNING] {report.total_above_max} values "
                            f"above max ({report.bounds.max_val} {report.bounds.units})"
                        )
                    total_bounds += bounds_issues
                    if report.var_name not in vars_with_issues:
                        vars_with_issues.append(report.var_name)

            # Show detailed anomalies
            for rec in report.anomalies:
                loc_label = "Node" if report.location_type == "node" else "Element"
                time_str = (
                    rec.time.isoformat() if rec.time else f"time_idx={rec.time_idx}"
                )
                if np.isnan(rec.value):
                    val_str = "NaN"
                elif np.isinf(rec.value):
                    val_str = "Inf" if rec.value > 0 else "-Inf"
                else:
                    val_str = f"{rec.value:.4g}"
                lines.append(
                    f"    {loc_label} {rec.index} (lon={rec.lon:.4f}, "
                    f"lat={rec.lat:.4f}): {val_str} at {time_str}"
                )

            lines.append("")

        # Summary
        lines.append("=" * 60)
        lines.append("Summary")
        lines.append("=" * 60)
        total_all = total_nan + total_inf + total_bounds
        lines.append(f"Total anomalies: {total_all}")
        lines.append(f"  NaN: {total_nan}")
        lines.append(f"  Inf: {total_inf}")
        lines.append(f"  Out of bounds: {total_bounds}")

        if vars_with_issues:
            lines.append(f"Variables with issues: {', '.join(vars_with_issues)}")

        if total_all == 0:
            lines.append("Status: PASSED (no anomalies found)")
        else:
            lines.append("Status: FAILED (anomalies found)")

        return "\n".join(lines)

    def to_dataframe(self, reports: list[AnomalyReport]) -> pd.DataFrame:
        """Convert anomaly reports to a pandas DataFrame.

        Parameters
        ----------
        reports : list[AnomalyReport]
            Validation reports from validate().

        Returns
        -------
        pd.DataFrame
            DataFrame with all anomaly records.
        """
        rows: list[dict] = []
        for report in reports:
            for rec in report.anomalies:
                rows.append(
                    {
                        "var_name": report.var_name,
                        "location_type": report.location_type,
                        "index_1based": rec.index,
                        "time_idx": rec.time_idx,
                        "timestamp": rec.time,
                        "lon": rec.lon,
                        "lat": rec.lat,
                        "value": rec.value,
                        "reason": rec.reason,
                    }
                )
        return pd.DataFrame(rows)

    def check_uniformity(
        self,
        variables: list[str] | None = None,
        rtol: float = 1e-9,
        atol: float = 1e-12,
        show_progress: bool = False,
    ) -> list[UniformityReport]:
        """Check spatial uniformity of each variable.

        For each variable and each time step, check if all spatial values
        are uniform (identical within tolerance). This is useful to verify
        that spatially uniform forcing data is actually uniform and won't
        cause interpolation failures.

        Parameters
        ----------
        variables : list[str] | None
            Specific variables to check (None = all data variables).
        rtol : float
            Relative tolerance for uniformity check (default: 1e-9).
        atol : float
            Absolute tolerance for uniformity check (default: 1e-12).
        show_progress : bool
            Display progress bar for each variable.

        Returns
        -------
        list[UniformityReport]
            Uniformity reports for each checked variable.
        """
        vars_to_check = variables if variables else self.data_vars
        reports: list[UniformityReport] = []

        with nc.Dataset(self.nc_path, "r") as ds:
            # Create iterator with optional progress bar
            var_iter: list[str] | tqdm[str]
            if show_progress:
                var_iter = tqdm(
                    vars_to_check,
                    desc="Checking uniformity",
                    unit="var",
                    ncols=80,
                )
            else:
                var_iter = vars_to_check

            for var_name in var_iter:
                if var_name not in ds.variables:
                    continue

                # Update progress bar description with current variable
                if show_progress and isinstance(var_iter, tqdm):
                    var_iter.set_postfix_str(var_name)

                data: NDArray[np.float64] = np.asarray(ds.variables[var_name][:])
                report = self._check_variable_uniformity(var_name, data, rtol, atol)
                reports.append(report)

        return reports

    def _check_variable_uniformity(
        self,
        var_name: str,
        data: NDArray[np.float64],
        rtol: float,
        atol: float,
    ) -> UniformityReport:
        """Check spatial uniformity for a single variable.

        Parameters
        ----------
        var_name : str
            Variable name.
        data : NDArray
            Data array with shape (time, node/element).
        rtol : float
            Relative tolerance.
        atol : float
            Absolute tolerance.

        Returns
        -------
        UniformityReport
            Uniformity report for this variable.
        """
        location_type = self._get_location_type(var_name)

        # Check each time step for spatial uniformity
        # Data shape is (time, space)
        for t_idx in range(data.shape[0]):
            time_slice = data[t_idx, :]

            # Skip if all NaN
            valid_mask = ~np.isnan(time_slice) & ~np.isinf(time_slice)
            valid_values = time_slice[valid_mask]

            if len(valid_values) == 0:
                continue

            # Use first valid value as reference
            ref_value = valid_values[0]

            # Check if all values are close to reference
            if not np.allclose(valid_values, ref_value, rtol=rtol, atol=atol):
                # Non-uniform: collect up to 3 examples of different values
                examples: list[NonUniformExample] = []
                seen_values: set[float] = set()

                # Always include the reference value as first example
                ref_idx = int(np.where(valid_mask)[0][0])
                examples.append(
                    NonUniformExample(index=ref_idx + 1, value=float(ref_value))
                )
                seen_values.add(float(ref_value))

                # Find other distinct values
                for loc_idx in range(len(time_slice)):
                    if not valid_mask[loc_idx]:
                        continue
                    val = float(time_slice[loc_idx])
                    # Check if this value is different from seen values
                    is_new = True
                    for seen_val in seen_values:
                        if np.isclose(val, seen_val, rtol=rtol, atol=atol):
                            is_new = False
                            break
                    if is_new:
                        examples.append(NonUniformExample(index=loc_idx + 1, value=val))
                        seen_values.add(val)
                        if len(examples) >= 3:
                            break

                return UniformityReport(
                    var_name=var_name,
                    location_type=location_type,
                    is_uniform=False,
                    examples=examples,
                    reference_value=float(ref_value),
                    first_nonuniform_time_idx=t_idx,
                )

        # All time steps are uniform
        # Get reference value from first time step
        first_valid = data[~np.isnan(data) & ~np.isinf(data)]
        ref_val = float(first_valid[0]) if len(first_valid) > 0 else None

        return UniformityReport(
            var_name=var_name,
            location_type=location_type,
            is_uniform=True,
            examples=[],
            reference_value=ref_val,
            first_nonuniform_time_idx=None,
        )

    def uniformity_summary(self, reports: list[UniformityReport]) -> str:
        """Generate human-readable uniformity summary.

        Parameters
        ----------
        reports : list[UniformityReport]
            Uniformity reports from check_uniformity().

        Returns
        -------
        str
            Formatted summary string.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("Spatial Uniformity Check")
        lines.append("=" * 60)
        lines.append(f"File: {self.nc_path.name}")
        lines.append(
            f"Nodes: {self.n_nodes}, Elements: {self.n_elements}, "
            f"Time steps: {self.n_times}"
        )
        lines.append("")

        uniform_vars: list[str] = []
        non_uniform_vars: list[str] = []

        for report in reports:
            loc_label = "node" if report.location_type == "node" else "element"

            if report.is_uniform:
                val_str = (
                    f"{report.reference_value:.6g}"
                    if report.reference_value is not None
                    else "N/A"
                )
                lines.append(f"{report.var_name}: Uniform (value={val_str})")
                uniform_vars.append(report.var_name)
            else:
                lines.append(f"{report.var_name}: Non-Uniform ({loc_label}-based)")
                # Show examples
                if report.first_nonuniform_time_idx is not None:
                    lines.append(
                        f"  First detected at time index: "
                        f"{report.first_nonuniform_time_idx}"
                    )
                lines.append(f"  Example values (1-based {loc_label} index):")
                for ex in report.examples:
                    lines.append(
                        f"    {loc_label.capitalize()} {ex.index}: {ex.value:.6g}"
                    )
                non_uniform_vars.append(report.var_name)

            lines.append("")

        # Summary
        lines.append("=" * 60)
        lines.append("Summary")
        lines.append("=" * 60)
        lines.append(f"Uniform variables: {len(uniform_vars)}")
        lines.append(f"Non-uniform variables: {len(non_uniform_vars)}")

        if non_uniform_vars:
            lines.append(f"Non-uniform: {', '.join(non_uniform_vars)}")

        return "\n".join(lines)
