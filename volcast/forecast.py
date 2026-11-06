"""Forecasting engines.

Two engines exist:

``drift``    - analytic baseline. sigma_h = realised_vol * sqrt(h). No torch.
``chronos``  - Amazon Chronos-2. Imported lazily; unusable on a 2-core CPU box,
               so it is an optional extra and never a hard dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import clean_float, safe_sigma


@dataclass
class Forecast:
    ticker: str
    sigma: float          # daily units (per-bar volatility)
    sigma_h: float        # over the horizon
    mu: float             # expected return over the horizon
    engine: str
    horizon: int

    def as_row(self) -> dict:
        return {
            "ticker": self.ticker,
            "sigma": self.sigma,
            "sigma_h": self.sigma_h,
            "mu": self.mu,
            "engine": self.engine,
            "horizon": self.horizon,
            "ann_vol": self.sigma_h * math.sqrt(252.0 / max(self.horizon, 1)),
        }


def drift_forecast(
    returns: pd.DataFrame | pd.Series,
    horizon: int = 5,
    window: int = 20,
    tickers: list[str] | None = None,
) -> list[Forecast]:
    """Baseline engine: constant-drift / random-walk-vol with sqrt-time scaling.

    BUG #1 FIX: sigma_h must be a *finite positive* number. The old code did
    ``np.std(...)`` on a slice that was often empty (or all-NaN) and returned NaN,
    and the sizer then silently produced zero positions. ``safe_sigma`` floors it.
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    if tickers is None:
        tickers = list(returns.columns)

    h = max(int(horizon), 1)
    sqrt_h = math.sqrt(float(h))
    out: list[Forecast] = []

    for t in tickers:
        if t not in returns.columns:
            sigma = safe_sigma(np.nan)
            out.append(Forecast(t, sigma, sigma * sqrt_h, 0.0, "drift", h))
            continue

        s = returns[t].dropna()
        tail = s.iloc[-int(window):] if len(s) else s
        raw = float(tail.std(ddof=1)) if len(tail) >= 2 else float("nan")

        sigma = safe_sigma(raw)
        sigma_h = safe_sigma(sigma * sqrt_h)
        mu = float(tail.mean()) * h if len(tail) else 0.0
        if not math.isfinite(mu):
            mu = 0.0

        out.append(Forecast(t, sigma, sigma_h, mu, "drift", h))

    return out


def chronos_forecast(
    prices: pd.DataFrame,
    horizon: int = 5,
    context_len: int = 256,
    model_id: str = "amazon/chronos-2",
) -> list[Forecast]:
    """Chronos-2 engine. Lazy import; raises a clear error if not installed."""
    try:
        from chronos import Chronos2Pipeline  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "chronos engine requested but chronos-forecasting is not installed. "
            "Install the 'chronos' extra (pulls torch) and ensure a GPU; on this "
            "machine use engine='drift'."
        ) from exc

    pipe = Chronos2Pipeline.from_pretrained(model_id)  # pragma: no cover
    out: list[Forecast] = []  # pragma: no cover
    for t in prices.columns:  # pragma: no cover
        ctx = prices[t].dropna().iloc[-context_len:]
        res = pipe.predict_quantiles(
            context=ctx.to_numpy(dtype=float),
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        lo, mid, hi = (np.asarray(r, dtype=float) for r in res[0])
        sigma = safe_sigma(float(np.std([lo[-1], mid[-1], hi[-1]])) / 2.563)
        out.append(Forecast(t, sigma, safe_sigma(sigma * math.sqrt(horizon)),
                            float(mid[-1]), "chronos", horizon))
    return out


def forecast_panel(
    returns: pd.DataFrame,
    horizon: int = 5,
    window: int = 20,
    engine: str = "drift",
    prices: pd.DataFrame | None = None,
    context_len: int = 256,
) -> pd.DataFrame:
    """Return a tidy forecast table (one row per ticker)."""
    eng = (engine or "drift").lower()

    if eng == "drift":
        fc = drift_forecast(returns, horizon=horizon, window=window)
    elif eng == "chronos":
        if prices is None:
            raise ValueError("chronos engine needs a price panel")
        fc = chronos_forecast(prices, horizon=horizon, context_len=context_len)
    else:
        raise ValueError(f"unknown engine: {engine!r} (expected 'drift' or 'chronos')")

    df = pd.DataFrame([f.as_row() for f in fc])
    # Final defensive pass: nothing non-finite may leave the library.
    for col in ("sigma", "sigma_h", "mu", "ann_vol"):
        df[col] = [clean_float(v) if clean_float(v) is not None else 0.0 for v in df[col]]
    return df
