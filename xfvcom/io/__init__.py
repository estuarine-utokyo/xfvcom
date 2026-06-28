# xfvcom/io/__init__.py
"""I/O sub-package of xfvcom.

Includes data loaders, force file generators, and GWO-AMD reader.
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# Public symbol table – declare *before* the imports so mypy is happy
# ---------------------------------------------------------------------
__all__: list[str] = []  # explicit annotation for mypy

# ---------------------------------------------------------------------
# Re-export everything from the former `io.py`
# ---------------------------------------------------------------------
from .core import *  # noqa: F401,F403

# core.py may itself define __all__; if so, merge it.
_core_all: list[str] | None = globals().get("__all__")  # after star-import
if _core_all:  # truthy & already a list[str]
    __all__.extend(_core_all)

from .groundwater_nc_generator import GroundwaterNetCDFGenerator  # noqa: F401
from .input_loader import FvcomInputLoader  # noqa: F401
from .jcope_grid import JcopeGrid  # noqa: F401
from .jcope_obc_generator import (  # noqa: F401
    JcopeObcGenerator,
    write_elevation_nc,
    write_tsobc_nc,
)
from .netcdf_utils import (  # noqa: F401
    decode_fvcom_time,
    encode_fvcom_time,
    extend_river_nc_file,
    read_fvcom_river_nc,
    to_mjd,
    write_fvcom_river_nc,
)

# ---------------------------------------------------------------------
# New helper(s)
# ---------------------------------------------------------------------
from .nml_parser import NamelistParser, parse_member_namelist  # noqa: F401
from .river_nml import parse_river_namelist  # noqa: F401

__all__.append("parse_river_namelist")
__all__.append("GroundwaterNetCDFGenerator")
__all__.append("FvcomInputLoader")
__all__.append("JcopeGrid")
__all__.append("JcopeObcGenerator")
__all__.append("write_elevation_nc")
__all__.append("write_tsobc_nc")
__all__.append("NamelistParser")
__all__.append("parse_member_namelist")

# ---------------------------------------------------------------------
# GWO-AMD meteorological data reader
# ---------------------------------------------------------------------
from .gwo_reader import (  # noqa: F401
    GWOForcingSource,
    GWOReader,
    parse_period,
    parse_station_map,
)

__all__.append("GWOReader")
__all__.append("GWOForcingSource")
__all__.append("parse_station_map")
__all__.append("parse_period")

# ---------------------------------------------------------------------
# GWO-AMD gap filling and correlations
# ---------------------------------------------------------------------
from .gwo_correlations import (  # noqa: F401
    StationCorrelations,
    get_station_coordinates,
    parse_fallback_stations,
)
from .gwo_gap_filler import (  # noqa: F401
    GapFiller,
    GapFillResult,
    MissingValueInfo,
    print_gap_fill_report,
    print_missing_value_report,
)

__all__.append("StationCorrelations")
__all__.append("parse_fallback_stations")
__all__.append("get_station_coordinates")
__all__.append("GapFiller")
__all__.append("GapFillResult")
__all__.append("MissingValueInfo")
__all__.append("print_gap_fill_report")
__all__.append("print_missing_value_report")

# Solar estimation (optional, requires pvlib)
try:
    from .solar_estimation import (  # noqa: F401
        SolarEstimator,
        estimate_solar_radiation,
    )

    __all__.append("SolarEstimator")
    __all__.append("estimate_solar_radiation")
except ImportError:
    pass  # pvlib not installed

# ---------------------------------------------------------------------
# MPOS (Tokyo Bay observation network) data loader
# ---------------------------------------------------------------------
from .mpos import MposLoader, MposStationConfig  # noqa: F401

__all__.append("MposLoader")
__all__.append("MposStationConfig")

# ---------------------------------------------------------------------
# FVCOM ascii vertical-coordinate / bathymetry I/O (sigma-z grid tool)
# ---------------------------------------------------------------------
from .dep_reader import DepData, read_dep, write_dep  # noqa: F401
from .sigma_dat import (  # noqa: F401
    SigmaFile,
    gtsz_to_lines,
    read_sigma_dat,
    write_sigma_dat,
)

__all__.append("DepData")
__all__.append("read_dep")
__all__.append("write_dep")
__all__.append("SigmaFile")
__all__.append("read_sigma_dat")
__all__.append("write_sigma_dat")
__all__.append("gtsz_to_lines")
