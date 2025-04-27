# xfvcom/plot_options.py
from dataclasses import dataclass, field
from typing import Any, Sequence
import cartopy.crs as ccrs

@dataclass
class FvcomPlotOptions:
    # --- 共通 ---
    cmap: str = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    levels: int | Sequence[float] = 20
    date_fmt: str = "%Y-%m-%d"

    # 2D／メッシュ
    with_mesh: bool = False
    coastlines: bool = False
    obclines: bool = False
    add_tiles: bool = False
    plot_grid: bool = False
    projection: ccrs.Projection = ccrs.Mercator()

    # ベクトル
    arrow_width : float = 0.002
    headlength  : int   = 5
    headwidth   : int   = 3

    # 余剰 kwargs
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_kwargs(cls, **kwargs):
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        core = {k: kwargs.pop(k) for k in list(kwargs) if k in field_names}
        return cls(**core, extra=kwargs)
