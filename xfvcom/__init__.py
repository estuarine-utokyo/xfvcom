# ------------------------------------------------------------------
# 0) lightweight first: pure-utility modules (no heavy imports)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 1) core analysis / I/O
# ------------------------------------------------------------------
from .analysis import FvcomAnalyzer
from .decorators import precedence
from .grid import FvcomGrid, get_grid, read_grid
from .io import (  # noqa: F401
    FvcomDataLoader,
    FvcomInputLoader,
    decode_fvcom_time,
    encode_fvcom_time,
    extend_river_nc_file,
    parse_river_namelist,
    read_fvcom_river_nc,
    to_mjd,
    write_fvcom_river_nc,
)

# ------------------------------------------------------------------
# 2) plotting sub-package (ordered: config → core → utils)
# ------------------------------------------------------------------
from .plot.config import FvcomPlotConfig
from .plot.core import FvcomPlotter
from .plot.markers import make_node_marker_post  # ← 直接 import
from .plot.utils import create_anim_2d_plot
from .plot_options import FvcomPlotOptions
from .utils.helpers import (
    FrameGenerator,
    PlotHelperMixin,
    create_gif,
    create_gif_from_frames,
)
from .utils.helpers_utils import (
    apply_xlim_ylim,
    evaluate_model_scores,
    generate_test_data,
)
from .utils.timeseries_utils import (
    extend_timeseries_ffill,
    extend_timeseries_linear,
    extend_timeseries_seasonal,
    interpolate_missing_values,
    resample_timeseries,
)

# ------------------------------------------------------------------
# 3) public symbol table
# ------------------------------------------------------------------
__all__: list[str] = [
    # plotting
    "FvcomPlotter",
    "FvcomPlotConfig",
    "FvcomPlotOptions",
    "create_anim_2d_plot",
    "make_node_marker_post",
    # analysis / I/O
    "FvcomDataLoader",
    "FvcomInputLoader",
    "FvcomAnalyzer",
    "parse_river_namelist",
    # NetCDF utilities
    "decode_fvcom_time",
    "encode_fvcom_time",
    "extend_river_nc_file",
    "read_fvcom_river_nc",
    "to_mjd",
    "write_fvcom_river_nc",
    # helpers
    "FrameGenerator",
    "PlotHelperMixin",
    "create_gif",
    "create_gif_from_frames",
    "apply_xlim_ylim",
    "evaluate_model_scores",
    "generate_test_data",
    # timeseries utilities
    "extend_timeseries_ffill",
    "extend_timeseries_linear",
    "extend_timeseries_seasonal",
    "interpolate_missing_values",
    "resample_timeseries",
    # misc
    "precedence",
    # grid
    "FvcomGrid",
    "get_grid",
    "read_grid",
]

__version__ = "0.2.0"
