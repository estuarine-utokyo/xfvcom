# tests/test_options_postinit.py
from xfvcom.plot_options import FvcomPlotOptions


def test_post_init_normalization():
    opts = FvcomPlotOptions(arrow_scale="auto")
    assert opts.arrow_scale is None
    assert isinstance(opts.vec_reduce, dict) and opts.vec_reduce == {}
    assert isinstance(opts.scalar_reduce, dict)
