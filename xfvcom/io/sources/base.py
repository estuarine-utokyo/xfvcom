from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class RiverTimeSeriesSource(ABC):
    """Abstract base class for river time-series providers."""

    @abstractmethod
    def get_series(self, variable: str, times: pd.DatetimeIndex) -> np.ndarray:
        """
        Return 1-D array aligned to *times*.
        NaN is allowed and will be filled later.
        """
        ...


class ConstantSource(RiverTimeSeriesSource):
    """Provide constant values for every variable."""

    def __init__(self, flux: float, temp: float, salt: float) -> None:
        self.constants = {"flux": flux, "temp": temp, "salt": salt}

    def get_series(self, variable: str, times: pd.DatetimeIndex) -> np.ndarray:
        val = self.constants[variable]
        return np.full(times.size, val, dtype="f4")
