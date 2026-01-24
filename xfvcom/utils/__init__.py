from .time_utils import sliced_time_label, time_label  # re-export for convenience
from .timeseries_utils import (
    extend_timeseries_ffill,
    extend_timeseries_linear,
    extend_timeseries_seasonal,
    interpolate_missing_values,
    resample_timeseries,
)
from .vertical import (
    compute_depth_at_siglay,
    compute_depth_range_average,
    parse_depth_range,
)

__all__ = [
    # existing exports ...
    "time_label",
    "sliced_time_label",
    # new timeseries utilities
    "extend_timeseries_ffill",
    "extend_timeseries_linear",
    "extend_timeseries_seasonal",
    "interpolate_missing_values",
    "resample_timeseries",
    # vertical coordinate utilities
    "compute_depth_at_siglay",
    "compute_depth_range_average",
    "parse_depth_range",
]
