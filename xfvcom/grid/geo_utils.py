from __future__ import annotations

"""Geographic helper utilities for *xfvcom.grid*.

Exposes :pyfunc:`utm_to_lonlat`, converting UTM coordinates to geographic
longitude/latitude (degrees, WGS-84; EPSG:4326).
"""

from functools import lru_cache
from typing import Tuple

import numpy as np
from pyproj import Transformer

__all__ = ["utm_to_lonlat"]


@lru_cache(maxsize=None)
def _get_transformer(zone: int, northern: bool) -> Transformer:
    if not 1 <= zone <= 60:
        raise ValueError("UTM zone must be 1–60")
    epsg = f"326{zone:02d}" if northern else f"327{zone:02d}"
    return Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)


def utm_to_lonlat(
    x: np.ndarray,
    y: np.ndarray,
    *,
    zone: int,
    northern: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    transformer = _get_transformer(zone, northern)
    lon, lat = transformer.transform(x, y)
    return np.asarray(lon), np.asarray(lat)
