"""Statistical metrics for model-observation comparison.

Standard metrics for ocean model validation:
  - Willmott's d (index of agreement)
  - RMSE (root mean squared error)
  - Bias (mean error)
  - Pearson correlation coefficient
"""

from __future__ import annotations

import numpy as np


def willmott_d(
    obs: np.ndarray,
    mod: np.ndarray,
    min_pairs: int = 30,
) -> float:
    """Compute Willmott's index of agreement (d).

    Parameters
    ----------
    obs : array-like
        Observation values (may contain NaN).
    mod : array-like
        Model values (may contain NaN).
    min_pairs : int
        Minimum number of valid (non-NaN) pairs required.
        Returns NaN if fewer pairs are available.

    Returns
    -------
    float
        d value in [0, 1], where 1 = perfect agreement.
    """
    obs = np.asarray(obs, dtype=np.float64)
    mod = np.asarray(mod, dtype=np.float64)
    mask = np.isfinite(obs) & np.isfinite(mod)
    n = int(np.sum(mask))
    if n < min_pairs:
        return np.nan
    o = obs[mask]
    m = mod[mask]
    mean_o = np.mean(o)
    ss_res: np.floating = np.sum((m - o) ** 2)
    ss_denom: np.floating = np.sum((np.abs(m - mean_o) + np.abs(o - mean_o)) ** 2)
    if ss_denom == 0:
        return np.nan
    return float(1.0 - ss_res / ss_denom)


def calc_rmse(
    obs: np.ndarray,
    mod: np.ndarray,
    min_pairs: int = 30,
) -> float:
    """Compute root mean squared error.

    Parameters
    ----------
    obs, mod : array-like
        Observation and model values (may contain NaN).
    min_pairs : int
        Minimum number of valid pairs required.

    Returns
    -------
    float
        RMSE value (>= 0).
    """
    obs = np.asarray(obs, dtype=np.float64)
    mod = np.asarray(mod, dtype=np.float64)
    mask = np.isfinite(obs) & np.isfinite(mod)
    if int(np.sum(mask)) < min_pairs:
        return np.nan
    return float(np.sqrt(np.mean((obs[mask] - mod[mask]) ** 2)))


def calc_bias(
    obs: np.ndarray,
    mod: np.ndarray,
    min_pairs: int = 30,
) -> float:
    """Compute mean bias (model - observation).

    Parameters
    ----------
    obs, mod : array-like
        Observation and model values (may contain NaN).
    min_pairs : int
        Minimum number of valid pairs required.

    Returns
    -------
    float
        Bias value. Positive = model overestimates.
    """
    obs = np.asarray(obs, dtype=np.float64)
    mod = np.asarray(mod, dtype=np.float64)
    mask = np.isfinite(obs) & np.isfinite(mod)
    if int(np.sum(mask)) < min_pairs:
        return np.nan
    return float(np.mean(mod[mask] - obs[mask]))


def pearson_r(
    obs: np.ndarray,
    mod: np.ndarray,
    min_pairs: int = 30,
) -> float:
    """Compute Pearson correlation coefficient.

    Parameters
    ----------
    obs, mod : array-like
        Observation and model values (may contain NaN).
    min_pairs : int
        Minimum number of valid pairs required.

    Returns
    -------
    float
        Correlation coefficient in [-1, 1].
    """
    obs = np.asarray(obs, dtype=np.float64)
    mod = np.asarray(mod, dtype=np.float64)
    mask = np.isfinite(obs) & np.isfinite(mod)
    if int(np.sum(mask)) < min_pairs:
        return np.nan
    o = obs[mask]
    m = mod[mask]
    if np.std(o) == 0 or np.std(m) == 0:
        return np.nan
    return float(np.corrcoef(o, m)[0, 1])
