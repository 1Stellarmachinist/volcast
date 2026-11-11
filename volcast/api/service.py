"""Service layer: JSON sanitisation and market computations for the API."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..data import load_prices
from ..forecast import forecast_panel
from ..indicators import feature_table
from ..panel import build_panel
from ..signals import cross_sectional_weights
from ..sizing import target_positions


# --------------------------------------------------------------------------- #
# BUG #2 FIX: the JSON sanitiser
# --------------------------------------------------------------------------- #
def _clean(obj: Any) -> Any:
    """Recursively replace NaN/inf/-inf with ``None`` so the payload is valid JSON.

    Without this, ``json.dumps`` emits bare ``NaN``/``Infinity`` (allowed by
    Python, invalid per RFC 8259) and the browser's ``JSON.parse`` throws
    "Unexpected token N". ``None`` serialises to JSON ``null``, which the client
    renders as a dash instead of crashing the whole screen.
    """
    if obj is None:
        return None

    # numpy / pandas scalars -> python scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return [_clean(v) for v in obj.tolist()]

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, bool) or isinstance(obj, int) or isinstance(obj, str):
        return obj

    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_clean(v) for v in obj]

    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()

    if isinstance(obj, pd.Series):
        return _clean(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return _clean(obj.to_dict(orient="records"))
    if isinstance(obj, pd.NA.__class__):  # pragma: no cover - defensive
        return None

    if isinstance(obj, float):
        return obj

    # Anything pydantic can model, or unknown -> best-effort string
    if hasattr(obj, "model_dump"):
        try:
            return _clean(obj.model_dump())
        except Exception:  # pragma: no cover
            return str(obj)

    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass

    return str(obj)


def clean_payload(obj: Any) -> Any:
    """Public alias used by every route before returning."""
    return _clean(obj)


# --------------------------------------------------------------------------- #
# Market data helpers
# --------------------------------------------------------------------------- #
def _panel_for(tickers: list[str], period: str = "1y", vol_window: int = 20):
    prices = load_prices(tickers, period=period, allow_synthetic=True, cache_root=".volcast/cache")
    return prices, build_panel(prices, vol_window=vol_window)


def forecast_payload(tickers: list[str], horizon: int = 5, window: int = 20,
                     engine: str = "drift", period: str = "1y") -> dict:
    prices, panel = _panel_for(tickers, period=period, vol_window=window)
    fc = forecast_panel(
        panel.returns, horizon=horizon, window=window, engine=engine, prices=panel.prices
    )
    rows = fc.to_dict(orient="records")
    return _clean(
        {
            "as_of": panel.prices.index[-1].isoformat(),
            "engine": engine,
            "horizon": horizon,
            "rows": rows,
        }
    )


def screen_payload(tickers: list[str], top: int = 10, vol_window: int = 20,
                   period: str = "1y") -> dict:
    prices, panel = _panel_for(tickers, period=period, vol_window=vol_window)
    feats = feature_table(panel.prices, vol_window=vol_window)
    feats = feats.assign(
        score=-feats["vol_20"].rank(pct=True) + feats["mom_63"].rank(pct=True)
    ).sort_values("score", ascending=False)
    rows = feats.head(top).reset_index().to_dict(orient="records")
    return _clean({"as_of": panel.prices.index[-1].isoformat(), "rows": rows})


def weights_payload(tickers: list[str], horizon: int = 5, window: int = 20,
                    target_vol: float = 0.15, max_weight: float = 0.25,
                    period: str = "1y") -> dict:
    prices, panel = _panel_for(tickers, period=period, vol_window=window)
    feats = panel.last_features()
    if "vol" not in feats or "momentum" not in feats:
        return _clean({"as_of": panel.prices.index[-1].isoformat(),
                       "net": 0.0, "gross": 0.0, "rows": []})

    score = (-feats["vol"].rank(pct=True) + 0.5) + (feats["momentum"].rank(pct=True) - 0.5)
    raw = cross_sectional_weights(score * 2.0, max_weight=max_weight)

    fc = forecast_panel(panel.returns, horizon=horizon, window=window, engine="drift")
    sigma_h = fc.set_index("ticker")["sigma_h"]
    sized = target_positions(
        raw, sigma_h, target_vol=target_vol, horizon=horizon,
        max_weight=max_weight, dollar_neutral=True,
    )

    rows = [
        {"ticker": t, "signal_w": float(raw.get(t, 0.0)), "target_w": float(sized.get(t, 0.0))}
        for t in sized.sort_values(ascending=False).index
    ]
    return _clean(
        {
            "as_of": panel.prices.index[-1].isoformat(),
            "net": float(sized.sum()),
            "gross": float(sized.abs().sum()),
            "rows": rows,
        }
    )


def backtest_payload(tickers: list[str], cost_bps: float = 5.0, test_days: int = 126,
                     rebalance: int = 5, target_vol: float = 0.15,
                     max_weight: float = 0.25, horizon: int = 5,
                     period: str = "2y") -> dict:
    from ..backtest import run_backtest

    prices, panel = _panel_for(tickers, period=period)
    res = run_backtest(
        panel, target_vol=target_vol, max_weight=max_weight, cost_bps=cost_bps,
        horizon=horizon, rebalance=rebalance, engine="drift", test_days=test_days,
    )
    m = res.stats
    sh, se = m.get("sharpe"), m.get("sharpe_se")
    significant = None
    if sh is not None and se is not None and not (
        isinstance(sh, float) and math.isnan(sh)
    ) and not (isinstance(se, float) and math.isnan(se)):
        significant = bool(abs(sh) > 2 * se)

    eq = res.equity
    return _clean(
        {
            "engine": m.get("engine", "drift"),
            "days": m.get("days", 0),
            "sharpe": sh,
            "sharpe_se": se,
            "gross_sharpe": m.get("gross_sharpe"),
            "ann_return": m.get("ann_return"),
            "ann_vol": m.get("ann_vol"),
            "max_drawdown": m.get("max_drawdown"),
            "total_cost": m.get("total_cost"),
            "avg_turnover": m.get("avg_turnover"),
            "significant": significant,
            "equity": [float(v) for v in eq.to_numpy()],
            "dates": [d.date().isoformat() for d in eq.index],
        }
    )


def quotes_payload(tickers: list[str], period: str = "1mo") -> dict:
    prices, _ = _panel_for(tickers, period=period)
    rets = prices.pct_change()
    rows = []
    for t in prices.columns:
        last = prices[t].iloc[-1]
        chg = rets[t].iloc[-1] if len(rets) > 1 else float("nan")
        rows.append({"ticker": t, "last": float(last), "change_1d": float(chg)})
    return _clean({"as_of": prices.index[-1].isoformat(), "rows": rows})


# --------------------------------------------------------------------------- #
# Penny / sub-penny board
# --------------------------------------------------------------------------- #
PENNY_UNIVERSE = [
    "SNDL", "NAK", "LCID", "PLUG", "AMC", "KOSS", "QS", "OPEN", "MVIS",
]


def penny_board_payload(tickers: list[str] | None = None, rvol_min: float = 1.5) -> dict:
    """Card data for the dashboard: price, move, RVOL, breakout state, risk flags.

    Everything here is daily-bar based, so it describes end-of-day posture rather
    than an intraday trigger. ``signal`` is deliberately conservative: a card can
    be flagged as a breakout only when volume AND trend agree.
    """
    import numpy as _np

    from ..subpenny import moving_average_state, relative_volume

    syms = [t.upper() for t in (tickers or PENNY_UNIVERSE)]
    try:
        prices = load_prices(syms, period="6mo", allow_synthetic=False,
                             cache_root=".volcast/cache")
    except Exception as exc:
        return _clean({"as_of": None, "cards": [], "error": f"{type(exc).__name__}: {exc}"})

    if prices is None or prices.empty:
        return _clean({"as_of": None, "cards": [], "error": "no price data"})

    volumes = None
    try:
        import yfinance as yf

        raw = yf.download(syms, period="6mo", progress=False, threads=False,
                          auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex) and "Volume" in raw.columns.get_level_values(0):
            volumes = raw["Volume"].reindex(columns=prices.columns)
    except Exception:
        volumes = None

    ma = moving_average_state(prices)
    rvol = relative_volume(volumes) if volumes is not None else None

    cards = []
    for t in prices.columns:
        s = prices[t].dropna()
        if len(s) < 25:
            continue

        last = float(s.iloc[-1])
        prev = float(s.iloc[-2])
        chg = (last / prev - 1.0) if prev > 0 else None

        hi = float(s.tail(60).max())
        from_hi = (last / hi - 1.0) if hi > 0 else None

        r = None
        if rvol is not None and t in rvol.index and _np.isfinite(rvol.get(t, _np.nan)):
            r = float(rvol[t])

        ma20 = ma.loc[t, "ma20_pos"] if t in ma.index else None
        slope = ma.loc[t, "ma20_slope"] if t in ma.index else None
        cross = bool(ma.loc[t, "ma_cross"]) if t in ma.index else False

        # Conservative signal: needs volume AND trend agreement.
        hot_vol = r is not None and r >= rvol_min
        trending = ma20 is not None and ma20 > 0 and slope is not None and slope > 0
        if hot_vol and trending:
            signal = "breakout"
        elif trending:
            signal = "trending"
        elif hot_vol:
            signal = "volume_only"
        else:
            signal = "quiet"

        flags = []
        if last < 1.0:
            flags.append("penny")
        if last < 0.001:
            flags.append("sub_penny")
        if len(s) < 120:
            flags.append("short_history")
        if from_hi is not None and from_hi < -0.5:
            flags.append("down_50_from_high")

        # 0-100 composite for the card's headline number.
        score = 50.0
        if r is not None:
            score += min(max((r - 1.0) * 12.0, -25.0), 30.0)
        if ma20 is not None:
            score += min(max(ma20 * 0.30, -20.0), 20.0)
        if cross:
            score += 8.0
        if "down_50_from_high" in flags:
            score -= 10.0
        score = float(min(max(score, 0.0), 100.0))

        cards.append(
            {
                "ticker": t,
                "last": last,
                "change_1d": chg,
                "rvol": r,
                "ma20_pos": float(ma20) if ma20 is not None else None,
                "ma20_slope": float(slope) if slope is not None else None,
                "cross": cross,
                "from_high": from_hi,
                "signal": signal,
                "score": score,
                "flags": flags,
                "spark": [float(v) for v in s.tail(40).to_numpy()],
            }
        )

    cards.sort(key=lambda c: c["score"], reverse=True)
    return _clean(
        {
            "as_of": prices.index[-1].isoformat(),
            "rvol_min": rvol_min,
            "cards": cards,
            "note": (
                "Daily-bar posture, not an intraday trigger. 'breakout' requires "
                "unusual volume AND a rising MA20; it is a shortlist, not a signal "
                "to buy. Measured follow-through on this pattern was negative."
            ),
        }
    )
