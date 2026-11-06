"""Panel construction: align a wide price frame into returns + features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators
from .utils import ffill


@dataclass
class Panel:
    prices: pd.DataFrame
    returns: pd.DataFrame
    features: dict[str, pd.DataFrame]

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)

    def last_features(self) -> pd.DataFrame:
        out = {}
        for name, frame in self.features.items():
            if len(frame):
                out[name] = frame.ffill().iloc[-1]
        return pd.DataFrame(out)

    def slice_dates(self, start: str | pd.Timestamp | None = None,
                    end: str | pd.Timestamp | None = None) -> Panel:
        p = self.prices
        r = self.returns
        f = self.features
        if start is not None:
            p = p.loc[p.index >= pd.Timestamp(start)]
            r = r.loc[r.index >= pd.Timestamp(start)]
            f = {k: v.loc[v.index >= pd.Timestamp(start)] for k, v in f.items()}
        if end is not None:
            p = p.loc[p.index <= pd.Timestamp(end)]
            r = r.loc[r.index <= pd.Timestamp(end)]
            f = {k: v.loc[v.index <= pd.Timestamp(end)] for k, v in f.items()}
        return Panel(prices=p, returns=r, features=f)


def build_panel(prices: pd.DataFrame, vol_window: int = 20) -> Panel:
    """Build indicator features once; all downstream code reuses this."""
    prices = ffill(prices).dropna(how="any")
    rets = indicators.log_returns(prices)

    feats = {
        "vol": indicators.realised_vol(rets, vol_window),
        "ewma_vol": indicators.ewma_vol(rets),
        "vol_ratio": indicators.vol_ratio(rets, 10, 60),
        "momentum": indicators.momentum(prices, 63),
        "rsi": indicators.rsi(prices, 14),
        "zscore": (prices - prices.rolling(60, min_periods=20).mean())
        / prices.rolling(60, min_periods=20).std(ddof=1).replace(0.0, np.nan),
    }
    # Replace any inf introduced by ratios.
    for k, v in feats.items():
        feats[k] = v.replace([np.inf, -np.inf], np.nan)

    return Panel(prices=prices, returns=rets, features=feats)
