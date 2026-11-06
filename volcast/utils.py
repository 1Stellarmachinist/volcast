"""Small numeric helpers shared across the package."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-12


def clean_float(x: Any) -> float | None:
    """Return a finite float, or None for NaN/inf/None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def safe_sigma(sigma: Any, floor: float = 1e-6) -> float:
    """Coerce a volatility estimate to a finite, strictly positive float.

    A NaN/inf/zero sigma silently produces zero-size positions downstream, so the
    floor is applied rather than letting the value through.
    """
    v = clean_float(sigma)
    if v is None or v <= 0:
        return floor
    return max(v, floor)


def ffill(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill column-wise.

    pandas 2.2 deprecated the ``method`` argument of ``fillna`` and pandas 3.x
    removed it outright, so this helper exists to keep a single call site that is
    known-good on both versions.
    """
    return df.ffill()


def zscore(s: pd.Series, window: int | None = None) -> pd.Series:
    """Z-score a series, scalar (whole-sample) or rolling.

    A zero dispersion means the z-score is undefined, so it is returned as NaN
    rather than infinity -- callers then ``fillna(0.0)`` and get a neutral signal
    instead of ``inf`` leaking into the sizing step.
    """
    if window is None:
        mu = s.mean()
        sd = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        if not np.isfinite(sd) or sd <= 0.0:
            return pd.Series(np.nan, index=s.index, dtype=float)
        return (s - mu) / sd

    min_periods = max(2, window // 2)
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=1)
    scaled = sd.where(sd > 0.0)
    return (s - mu) / scaled


def annualise_vol(daily_sigma: float, periods: int = 252) -> float:
    return safe_sigma(daily_sigma) * math.sqrt(periods)


def deannualise_vol(annual_sigma: float, periods: int = 252) -> float:
    return safe_sigma(annual_sigma) / math.sqrt(periods)
