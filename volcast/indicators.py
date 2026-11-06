"""Technical indicators used as cross-sectional features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import EPS


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().dropna(how="all")


def realised_vol(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rolling standard deviation of daily returns (daily units)."""
    return returns.rolling(window, min_periods=max(3, window // 2)).std(ddof=1)


def parkinson_vol(high: pd.DataFrame, low: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    hl = np.log(high / low.replace(0.0, np.nan)) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    return np.sqrt(factor * hl.rolling(window, min_periods=max(3, window // 2)).mean())


def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = prices.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    avg_up = up.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_down = down.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_up / (avg_down + EPS)
    return 100.0 - 100.0 / (1.0 + rs)


def momentum(prices: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    return prices.pct_change(window)


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).stack(),
            (high - prev_close).abs().stack(),
            (low - prev_close).abs().stack(),
        ],
        axis=1,
    ).max(axis=1)
    tr = tr.unstack()
    return tr.rolling(window, min_periods=max(3, window // 2)).mean()


def ewma_vol(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """RiskMetrics-style exponentially weighted volatility."""
    var = returns.pow(2).ewm(alpha=1.0 - lam, adjust=False, min_periods=10).mean()
    return np.sqrt(var.clip(lower=0.0))


def vol_ratio(returns: pd.DataFrame, fast: int = 10, slow: int = 60) -> pd.DataFrame:
    """Short vol over long vol - >1 means vol expanding."""
    f = realised_vol(returns, fast)
    s = realised_vol(returns, slow)
    return f / s.replace(0.0, np.nan)


def feature_table(prices: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """Flatten a few per-ticker features into a long DataFrame for screening."""
    rets = log_returns(prices)
    rv = realised_vol(rets, vol_window)
    mom = momentum(prices, 63)
    z = (prices - prices.rolling(60, min_periods=20).mean()) / prices.rolling(
        60, min_periods=20
    ).std(ddof=1).replace(0.0, np.nan)

    last = -1
    rows = []
    for t in prices.columns:
        rows.append(
            {
                "ticker": t,
                "last": float(prices[t].iloc[last]),
                "vol_20": float(rv[t].iloc[last]) if not np.isnan(rv[t].iloc[last]) else np.nan,
                "mom_63": float(mom[t].iloc[last]) if not np.isnan(mom[t].iloc[last]) else np.nan,
                "z_60": float(z[t].iloc[last]) if not np.isnan(z[t].iloc[last]) else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("ticker")
