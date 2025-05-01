"""
xfvcom top-level package
Re-export public APIs after file split.
"""

# ------------------------------------------------------------------
# 1) new canonical location  (xfvcom.plot.core)
# ------------------------------------------------------------------
from .plot.config import FvcomPlotConfig
from .analysis    import FvcomAnalyzer
from .io          import FvcomDataLoader
from .plot.core   import FvcomPlotter
from .plot.config import FvcomPlotConfig

# ------------------------------------------------------------------
# 2) helpers / utilities (unchanged)
# ------------------------------------------------------------------
from .utils.helpers import (
    PlotHelperMixin, FrameGenerator,
    create_gif, create_gif_from_frames,
)
from .utils.helpers_utils import (
    apply_xlim_ylim, evaluate_model_scores, generate_test_data,
)

from .plot.utils import create_anim_2d_plot
from .plot_options import FvcomPlotOptions
from .decorators import precedence

__all__ = [
    "FvcomPlotter", "FvcomDataLoader", "FvcomAnalyzer", "FvcomPlotConfig",
    "PlotHelperMixin", "FrameGenerator",
    "create_gif", "create_gif_from_frames",
    "apply_xlim_ylim", "evaluate_model_scores", "generate_test_data",
    "create_anim_2d_plot", "FvcomPlotOptions", "precedence",
]
