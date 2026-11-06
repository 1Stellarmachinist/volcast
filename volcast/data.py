"""Price/volume data access with an on-disk parquet/CSV cache.

yfinance is optional at runtime: if it is missing or the network is down, the
deterministic synthetic generator keeps every downstream tool runnable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ffill


def cache_dir(root: str | Path = ".volcast/cache") -> Path:
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _key(tickers: list[str], period: str) -> str:
    raw = "|".join(sorted(tickers)) + f"::{period}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def synthetic_prices(
    tickers: list[str],
    periods: int = 400,
    seed: int = 7,
    base_vol: float = 0.02,
) -> pd.DataFrame:
    """Deterministic GBM-ish panel used when real data is unavailable.

    Each ticker gets a slightly different volatility so cross-sectional
    ranking has something to bite on.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)
    out = {}
    for i, t in enumerate(tickers):
        vol = base_vol * (0.7 + 0.15 * i)
        drift = 0.0003
        shocks = rng.normal(drift, vol, size=periods)
        out[t] = 100.0 * np.exp(np.cumsum(shocks))
    return pd.DataFrame(out, index=idx)


def _download(tickers: list[str], period: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        raw = yf.download(
            list(tickers),
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        return None
    if raw is None or len(raw) == 0:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return None
        close = raw["Close"]
    else:
        close = raw[["Close"]]
        if len(tickers) == 1:
            close.columns = [tickers[0]]
    close = close.dropna(how="all")
    return close if len(close) else None


def load_prices(
    tickers: list[str],
    period: str = "2y",
    use_cache: bool = True,
    allow_synthetic: bool = True,
    cache_root: str | Path = ".volcast/cache",
    seed: int = 7,
) -> pd.DataFrame:
    """Return a wide close-price DataFrame (index=dates, columns=tickers)."""
    tickers = [t.upper() for t in tickers]
    if not tickers:
        raise ValueError("tickers must not be empty")

    path = cache_dir(cache_root) / f"px_{_key(tickers, period)}.csv"
    if use_cache and path.is_file():
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if len(df) and set(tickers).issubset(df.columns):
                return ffill(df[tickers])
        except Exception:
            pass

    df = _download(tickers, period)

    if df is None or not set(tickers).issubset(df.columns):
        if not allow_synthetic:
            missing = sorted(set(tickers) - set(df.columns)) if df is not None else tickers
            raise RuntimeError(f"price download failed or missing tickers: {missing}")
        df = synthetic_prices(tickers, periods=400, seed=seed)

    df = ffill(df[tickers]).dropna(how="any")
    if use_cache:
        try:
            df.to_csv(path)
        except Exception:
            pass
    return df


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")
