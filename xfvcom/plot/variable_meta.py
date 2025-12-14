"""Variable metadata for FVCOM input/output files.

This module provides standardized variable names, units, and LaTeX-formatted
labels for publication-quality plots. It handles both FVCOM output variables
and forcing file variables (meteorological, river, groundwater).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr


@dataclass(frozen=True)
class VariableMeta:
    """Metadata for a single variable.

    Attributes
    ----------
    long_name : str
        Human-readable variable name
    units : str
        Physical units (plain text)
    units_latex : str
        LaTeX-formatted units for publication plots
    symbol : str
        Variable symbol (e.g., 'T' for temperature)
    """

    long_name: str
    units: str
    units_latex: str
    symbol: str = ""

    def label(self, use_latex: bool = True) -> str:
        """Generate axis label with units.

        Parameters
        ----------
        use_latex : bool
            If True, use LaTeX-formatted units (default: True)

        Returns
        -------
        str
            Formatted label like "Temperature [$^\\circ$C]"
        """
        units_str = self.units_latex if use_latex else self.units
        if units_str and units_str != "-":
            return f"{self.long_name} [{units_str}]"
        return self.long_name


# =============================================================================
# FVCOM Meteorological Forcing Variables
# =============================================================================
MET_VARIABLES: dict[str, VariableMeta] = {
    "uwind_speed": VariableMeta(
        long_name="Eastward Wind Speed",
        units="m/s",
        units_latex=r"m s$^{-1}$",
        symbol="u",
    ),
    "vwind_speed": VariableMeta(
        long_name="Northward Wind Speed",
        units="m/s",
        units_latex=r"m s$^{-1}$",
        symbol="v",
    ),
    "air_temperature": VariableMeta(
        long_name="Air Temperature",
        units="Celsius",
        units_latex=r"$^\circ$C",
        symbol="T_a",
    ),
    "cloud_cover": VariableMeta(
        long_name="Cloud Cover",
        units="-",
        units_latex="-",
        symbol="C",
    ),
    "short_wave": VariableMeta(
        long_name="Shortwave Radiation",
        units="W/m2",
        units_latex=r"W m$^{-2}$",
        symbol="Q_sw",
    ),
    "long_wave": VariableMeta(
        long_name="Longwave Radiation",
        units="W/m2",
        units_latex=r"W m$^{-2}$",
        symbol="Q_lw",
    ),
    "relative_humidity": VariableMeta(
        long_name="Relative Humidity",
        units="%",
        units_latex=r"\%",
        symbol="RH",
    ),
    "air_pressure": VariableMeta(
        long_name="Air Pressure",
        units="hPa",
        units_latex="hPa",
        symbol="P_a",
    ),
    "Precipitation": VariableMeta(
        long_name="Precipitation",
        units="m/s",
        units_latex=r"m s$^{-1}$",
        symbol="P",
    ),
    "net_heat_flux": VariableMeta(
        long_name="Net Heat Flux",
        units="W/m2",
        units_latex=r"W m$^{-2}$",
        symbol="Q_net",
    ),
}

# =============================================================================
# FVCOM River Forcing Variables
# =============================================================================
RIVER_VARIABLES: dict[str, VariableMeta] = {
    "river_flux": VariableMeta(
        long_name="River Discharge",
        units="m3/s",
        units_latex=r"m$^3$ s$^{-1}$",
        symbol="Q",
    ),
    "river_temp": VariableMeta(
        long_name="River Temperature",
        units="Celsius",
        units_latex=r"$^\circ$C",
        symbol="T_r",
    ),
    "river_salt": VariableMeta(
        long_name="River Salinity",
        units="PSU",
        units_latex="PSU",
        symbol="S_r",
    ),
}

# =============================================================================
# FVCOM Groundwater Forcing Variables
# =============================================================================
GROUNDWATER_VARIABLES: dict[str, VariableMeta] = {
    "groundwater_flux": VariableMeta(
        long_name="Groundwater Flux",
        units="m/s",
        units_latex=r"m s$^{-1}$",
        symbol="q_gw",
    ),
    "groundwater_temp": VariableMeta(
        long_name="Groundwater Temperature",
        units="Celsius",
        units_latex=r"$^\circ$C",
        symbol="T_gw",
    ),
    "groundwater_salt": VariableMeta(
        long_name="Groundwater Salinity",
        units="PSU",
        units_latex="PSU",
        symbol="S_gw",
    ),
}

# =============================================================================
# FVCOM Output Variables
# =============================================================================
OUTPUT_VARIABLES: dict[str, VariableMeta] = {
    "temp": VariableMeta(
        long_name="Temperature",
        units="Celsius",
        units_latex=r"$^\circ$C",
        symbol="T",
    ),
    "salinity": VariableMeta(
        long_name="Salinity",
        units="PSU",
        units_latex="PSU",
        symbol="S",
    ),
    "zeta": VariableMeta(
        long_name="Sea Surface Height",
        units="m",
        units_latex="m",
        symbol=r"$\zeta$",
    ),
    "u": VariableMeta(
        long_name="Eastward Velocity",
        units="m/s",
        units_latex=r"m s$^{-1}$",
        symbol="u",
    ),
    "v": VariableMeta(
        long_name="Northward Velocity",
        units="m/s",
        units_latex=r"m s$^{-1}$",
        symbol="v",
    ),
    "ww": VariableMeta(
        long_name="Vertical Velocity",
        units="m/s",
        units_latex=r"m s$^{-1}$",
        symbol="w",
    ),
    "ua": VariableMeta(
        long_name="Depth-averaged Eastward Velocity",
        units="m/s",
        units_latex=r"m s$^{-1}$",
        symbol=r"$\bar{u}$",
    ),
    "va": VariableMeta(
        long_name="Depth-averaged Northward Velocity",
        units="m/s",
        units_latex=r"m s$^{-1}$",
        symbol=r"$\bar{v}$",
    ),
    "h": VariableMeta(
        long_name="Bathymetric Depth",
        units="m",
        units_latex="m",
        symbol="h",
    ),
    "dye": VariableMeta(
        long_name="Dye Concentration",
        units="kg/m3",
        units_latex=r"kg m$^{-3}$",
        symbol="C",
    ),
    "O2_o": VariableMeta(
        long_name="Dissolved Oxygen",
        units="mg/L",
        units_latex=r"mg L$^{-1}$",
        symbol="DO",
    ),
}

# =============================================================================
# Combined registry
# =============================================================================
VARIABLE_REGISTRY: dict[str, VariableMeta] = {
    **MET_VARIABLES,
    **RIVER_VARIABLES,
    **GROUNDWATER_VARIABLES,
    **OUTPUT_VARIABLES,
}


def get_variable_meta(var_name: str) -> VariableMeta | None:
    """Look up variable metadata by name.

    Parameters
    ----------
    var_name : str
        Variable name (e.g., 'air_temperature', 'temp')

    Returns
    -------
    VariableMeta or None
        Metadata if found, None otherwise
    """
    return VARIABLE_REGISTRY.get(var_name)


def get_label(
    da: xr.DataArray,
    var_name: str | None = None,
    use_latex: bool = True,
) -> str:
    """Get formatted axis label for a DataArray.

    Tries to get metadata from:
    1. Variable registry (known FVCOM variables)
    2. DataArray attributes (long_name, units)
    3. Variable name as fallback

    Parameters
    ----------
    da : xr.DataArray
        Data array to get label for
    var_name : str, optional
        Variable name override. If None, uses da.name
    use_latex : bool
        If True, use LaTeX-formatted units (default: True)

    Returns
    -------
    str
        Formatted label for axis
    """
    name = var_name or da.name or "value"

    # Try registry first
    meta = get_variable_meta(name)
    if meta is not None:
        return meta.label(use_latex=use_latex)

    # Fall back to DataArray attributes
    long_name = da.attrs.get("long_name", name)
    units = da.attrs.get("units", "")

    # Clean up common unit formats
    if use_latex and units:
        units = _to_latex_units(units)

    if units and units != "-":
        return f"{long_name} [{units}]"
    return long_name


def _to_latex_units(units: str) -> str:
    """Convert common unit strings to LaTeX format.

    Parameters
    ----------
    units : str
        Plain text units

    Returns
    -------
    str
        LaTeX-formatted units
    """
    # Common substitutions
    conversions = {
        "m/s": r"m s$^{-1}$",
        "m s-1": r"m s$^{-1}$",
        "m3/s": r"m$^3$ s$^{-1}$",
        "m^3/s": r"m$^3$ s$^{-1}$",
        "W/m2": r"W m$^{-2}$",
        "W m-2": r"W m$^{-2}$",
        "Watts meter-2": r"W m$^{-2}$",
        "kg/m3": r"kg m$^{-3}$",
        "kg m-3": r"kg m$^{-3}$",
        "mg/L": r"mg L$^{-1}$",
        "Celsius": r"$^\circ$C",
        "Celsius Degree": r"$^\circ$C",
        "degrees Celsius": r"$^\circ$C",
        "degC": r"$^\circ$C",
        "percentage": r"\%",
        "percent": r"\%",
    }
    return conversions.get(units, units)
