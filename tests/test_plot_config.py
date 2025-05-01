from xfvcom.plot.config import FvcomPlotConfig

def test_dataclass_defaults():
    cfg = FvcomPlotConfig()
    assert cfg.figsize == (8.0, 6.0)
    assert cfg._private_cache == {}
