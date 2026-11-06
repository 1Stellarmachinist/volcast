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


_PIPELINE_CACHE: dict[tuple[str, str], object] = {}


def _load_pipeline(model_id: str, device: str):
    """Load (and memoise) a Chronos-2 pipeline.

    Building the pipeline from_pretrained takes seconds and ~500MB of weights. A
    backtest re-forecasts at every rebalance (~26 times over 126 days), so without
    this cache the model would be reloaded from disk 26 times per run -- and the
    GPU would sit idle while the disk did all the work.
    """
    key = (model_id, device)
    if key not in _PIPELINE_CACHE:
        from chronos import Chronos2Pipeline  # type: ignore

        _PIPELINE_CACHE[key] = Chronos2Pipeline.from_pretrained(model_id, device_map=device)
    return _PIPELINE_CACHE[key]


def clear_pipeline_cache() -> None:
    _PIPELINE_CACHE.clear()


def _pick_device() -> str:
    """Prefer CUDA when it is actually present."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch is optional
        return "cpu"


def chronos_forecast(
    prices: pd.DataFrame,
    horizon: int = 5,
    context_len: int = 256,
    model_id: str = "amazon/chronos-2",
    device_map: str | None = None,
    batch_size: int = 16,
) -> list[Forecast]:
    """Chronos-2 engine (zero-shot).

    The model is *fed log-returns*, not prices. Chronos-2's quantile spread is then
    already in return units, so the implied sigma needs no price-scale conversion.
    For a Gaussian, q90 - q10 == 2 * 1.2816 * sigma, which gives a calibration-free
    volatility estimate straight from the model's own predictive distribution.

    API note: Chronos-2 exposes ``Chronos2Pipeline.predict_df(...)``. The older
    ``predict_quantiles(context=...)`` signature belonged to Chronos/T5 v1 and does
    NOT exist in 2.x.
    """
    try:
        # Availability probe only; _load_pipeline performs the real import.
        from chronos import Chronos2Pipeline  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "chronos engine requested but chronos-forecasting is not installed. "
            "Install the 'chronos' extra (pip install -e '.[chronos]') on a machine "
            "with a GPU; on a 2-core CPU box use engine='drift'."
        ) from exc

    device = device_map or _pick_device()
    pipeline = _load_pipeline(model_id, device)

    h = max(int(horizon), 1)
    rets = np.log(prices).diff()

    # Long format: one row per (ticker, timestamp), which is what predict_df wants.
    frames = []
    for ticker in prices.columns:
        s = rets[ticker].dropna().iloc[-int(context_len):]
        if len(s) < 8:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "id": ticker,
                    "timestamp": s.index,
                    "target": s.to_numpy(dtype=float),
                }
            )
        )

    if not frames:
        return [
            Forecast(t, safe_sigma(np.nan), safe_sigma(np.nan), 0.0, "chronos", h)
            for t in prices.columns
        ]

    context_df = pd.concat(frames, ignore_index=True)

    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=h,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
        batch_size=batch_size,
    )

    # predict_df returns one row per (id, step) with columns "0.1", "0.5", "0.9".
    q10, q50, q90 = "0.1", "0.5", "0.9"
    out: list[Forecast] = []

    for ticker in prices.columns:
        sub = pred_df[pred_df["id"] == ticker] if "id" in pred_df.columns else pred_df
        if len(sub) == 0 or not {q10, q50, q90}.issubset(sub.columns):
            out.append(
                Forecast(ticker, safe_sigma(np.nan), safe_sigma(np.nan), 0.0, "chronos", h)
            )
            continue

        spread = (sub[q90].to_numpy(dtype=float) - sub[q10].to_numpy(dtype=float)) / (
            2.0 * 1.2815515655446004
        )
        # Per-step sigma; combine to the horizon by summing variances.
        sigma_h = safe_sigma(float(np.sqrt(np.nansum(spread**2))))
        sigma_1 = safe_sigma(float(np.nanmean(spread)))
        mu = float(np.nansum(sub[q50].to_numpy(dtype=float)))
        if not math.isfinite(mu):
            mu = 0.0

        out.append(Forecast(ticker, sigma_1, sigma_h, mu, "chronos", h))

    return out


def forecast_panel(
    returns: pd.DataFrame,
    horizon: int = 5,
    window: int = 20,
    engine: str = "drift",
    prices: pd.DataFrame | None = None,
    context_len: int = 256,
    device_map: str | None = None,
) -> pd.DataFrame:
    """Return a tidy forecast table (one row per ticker)."""
    eng = (engine or "drift").lower()

    if eng == "drift":
        fc = drift_forecast(returns, horizon=horizon, window=window)
    elif eng == "chronos":
        if prices is None:
            raise ValueError("chronos engine needs a price panel")
        fc = chronos_forecast(
            prices, horizon=horizon, context_len=context_len, device_map=device_map
        )
    else:
        raise ValueError(f"unknown engine: {engine!r} (expected 'drift' or 'chronos')")

    df = pd.DataFrame([f.as_row() for f in fc])
    # Final defensive pass: nothing non-finite may leave the library.
    for col in ("sigma", "sigma_h", "mu", "ann_vol"):
        df[col] = [clean_float(v) if clean_float(v) is not None else 0.0 for v in df[col]]
    return df
