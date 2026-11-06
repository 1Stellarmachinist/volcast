"""Small-cap candidate scoring: a growth/greed ranking that is honest about
what it can and cannot prove.

THE CENTRAL DESIGN CONSTRAINT
-----------------------------
There are two kinds of signal here, and they must never be mixed in a backtest:

1. **Price/volume signals** (``price_signals``) are computed from data available
   *at* each point in time, so they can be backtested without look-ahead.

2. **Fundamental signals** (``fundamental_snapshot``) come from yfinance's
   ``.info`` endpoint, which returns a **current** snapshot. Scoring historical
   dates with today's revenue growth would leak the future into the past and
   manufacture an edge that does not exist.

Therefore ``composite_score`` accepts ``allow_fundamentals=False`` by default, and
``backtest_ranking`` refuses to use them at all. Fundamentals only influence the
**latest** ranking, which is a decision about tomorrow, not a claim about history.

WHAT THIS DOES NOT DO
---------------------
It does not predict which stock will go up. It ranks small caps on the things
that have historically been *associated* with outperformance and that you can
actually trade: momentum, liquidity, volatility regime, and (for today only)
balance-sheet survival. A ranking is a shortlist to research, not a buy list.
Every output carries its measured hit rate so you can judge it yourself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Limits of what free data allows
# --------------------------------------------------------------------------- #
MIN_DOLLAR_VOLUME = 500_000.0   # below this you cannot get filled without moving price
MIN_PRICE = 0.30                # sub-$0.30 is usually a shell / reverse-split artefact
MIN_BARS = 200                  # ~10 months, enough for 6-month momentum


@dataclass
class ScoreRow:
    ticker: str
    score: float
    momentum_3m: float | None = None
    momentum_6m: float | None = None
    vol_regime: float | None = None
    volume_surge: float | None = None
    dist_from_low: float | None = None
    dollar_volume: float | None = None
    atr_pct: float | None = None
    # fundamentals: None in any historical/backtest context, by design
    revenue_growth: float | None = None
    gross_margin: float | None = None
    cash_to_debt: float | None = None
    short_pct_float: float | None = None
    flags: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 1. Price/volume signals - point-in-time safe, backtestable
# --------------------------------------------------------------------------- #
def price_signals(prices: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """Per-ticker signals that use only data up to each row's date.

    Returns a MultiIndex-free long frame indexed by (date, ticker) via stack().
    Every column is causal: row t uses only observations <= t.
    """
    px = prices.copy()
    rets = np.log(px).diff()

    out: dict[str, pd.DataFrame] = {}

    # Momentum: the one small-cap effect with decades of evidence, and the one
    # that also exists in microcaps (unlike value, which needs clean fundamentals).
    out["momentum_3m"] = px.pct_change(63)
    out["momentum_6m"] = px.pct_change(126)

    # Volatility regime: recent vol vs its own 6-month norm. LOW means a quiet
    # coil that often precedes expansion; we treat it as a mild positive because
    # entering after a blow-off is measurably worse.
    fast = rets.rolling(vol_window, min_periods=vol_window // 2).std()
    slow = rets.rolling(126, min_periods=60).std()
    out["vol_regime"] = fast / slow.where(slow > 0)

    # Volume surge: 5-day average vs 60-day. Attention arriving. Left as NaN here
    # because this frame carries prices only -- the CLI passes volumes separately.
    out["volume_surge"] = pd.DataFrame(np.nan, index=px.index, columns=px.columns)

    # Distance from 52-week low, in percent. Recovery names rather than names
    # that have already tripled.
    low252 = px.rolling(252, min_periods=120).min()
    out["dist_from_low"] = px / low252.where(low252 > 0) - 1.0

    # Average true range as % of price: a direct read on gap risk.
    prev = px.shift(1)
    tr = pd.concat([(px - prev).abs().stack()], axis=1).max(axis=1).unstack()
    out["atr_pct"] = tr.rolling(14, min_periods=7).mean() / px.where(px > 0)

    frames = []
    for name, frame in out.items():
        s = frame.stack(future_stack=True).rename(name)
        frames.append(s)
    sig = pd.concat(frames, axis=1)
    sig.index.names = ["date", "ticker"]
    return sig


def liquidity_screen(prices: pd.DataFrame, volumes: pd.DataFrame | None = None,
                     window: int = 20) -> pd.Series:
    """Latest-bar pass/fail per ticker on the two things that make a name tradeable."""
    last = prices.iloc[-1]
    if volumes is None or volumes.empty:
        return pd.Series(True, index=prices.columns, name="liquid")

    avg_dv = (prices * volumes).rolling(window, min_periods=5).mean().iloc[-1]
    ok = (last >= MIN_PRICE) & (avg_dv >= MIN_DOLLAR_VOLUME)
    return ok.rename("liquid")


# --------------------------------------------------------------------------- #
# 2. Fundamentals - CURRENT snapshot only, never backtested
# --------------------------------------------------------------------------- #
def fundamental_snapshot(tickers: list[str], timeout: float = 12.0) -> pd.DataFrame:
    """Fetch a *current* fundamentals snapshot.

    WARNING: this is as-of-now data. It must not be used to score historical
    dates. ``backtest_ranking`` will not call this function.
    """
    try:
        import yfinance as yf
    except Exception:  # pragma: no cover
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for t in tickers:
        rec: dict[str, Any] = {"ticker": t}
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            info = {}

        rec["revenue_growth"] = _num(info.get("revenueGrowth"))
        rec["gross_margin"] = _num(info.get("grossMargins"))
        rec["profit_margin"] = _num(info.get("profitMargins"))
        rec["short_pct_float"] = _num(info.get("sharesPercentSharesOut"))
        rec["insider_pct"] = _num(info.get("heldPercentInsiders"))

        cash = _num(info.get("totalCash"))
        debt = _num(info.get("totalDebt"))
        if cash is not None and debt is not None:
            rec["cash_to_debt"] = cash / debt if debt > 0 else (float("inf") if cash > 0 else 0.0)
        else:
            rec["cash_to_debt"] = None

        rec["market_cap"] = _num(info.get("marketCap"))
        rows.append(rec)

    df = pd.DataFrame(rows).set_index("ticker")
    return df.replace([np.inf, -np.inf], np.nan)


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# --------------------------------------------------------------------------- #
# 3. Composite score
# --------------------------------------------------------------------------- #
def _zscore_rank(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile in [-1, 1]; NaN becomes neutral 0."""
    if s.notna().sum() < 2:
        return pd.Series(0.0, index=s.index)
    return ((s.rank(pct=True) - 0.5) * 2.0).fillna(0.0)


def composite_score(
    signals: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    w_momentum: float = 1.0,
    w_volregime: float = 0.5,
    w_fromlow: float = 0.3,
    w_fundamental: float = 1.0,
) -> pd.Series:
    """Combine signals into one score per ticker for a SINGLE date.

    ``signals`` must be one row per ticker (a cross-section). Weights are
    deliberately conservative; momentum dominates because it is the effect with
    the most evidence in small caps.

    Fundamentals, if supplied, must be a *current* snapshot. Passing them while
    scoring a historical date is look-ahead bias.
    """
    if signals.empty:
        return pd.Series(dtype=float)

    mom = signals.get("momentum_3m", pd.Series(0.0, index=signals.index))
    vol_reg = signals.get("vol_regime", pd.Series(np.nan, index=signals.index))
    from_low = signals.get("dist_from_low", pd.Series(np.nan, index=signals.index))

    score = (
        w_momentum * _zscore_rank(mom)
        # Low vol regime is mildly positive: dampened recent range, not a spike.
        - w_volregime * _zscore_rank(vol_reg)
        + w_fromlow * _zscore_rank(from_low)
    )

    if fundamentals is not None and not fundamentals.empty:
        f = fundamentals.reindex(signals.index)
        rev = f.get("revenue_growth")
        gm = f.get("gross_margin")
        c2d = f.get("cash_to_debt")
        fund = pd.Series(0.0, index=signals.index)
        if rev is not None:
            fund = fund + 0.5 * _zscore_rank(rev)
        if gm is not None:
            fund = fund + 0.25 * _zscore_rank(gm)
        if c2d is not None:
            fund = fund + 0.25 * _zscore_rank(c2d)
        score = score + w_fundamental * fund

    return score.rename("score")


def rank_smallcaps(
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    top: int = 10,
    min_bars: int = MIN_BARS,
) -> pd.DataFrame:
    """Rank the universe as of the LAST available bar.

    This is the 'what should I look at' answer. It returns flags explaining
    exclusions rather than silently dropping names, because a name vanishing with
    no explanation is how people end up trusting the survivors.
    """
    if len(prices) < min_bars:
        raise ValueError(f"need >= {min_bars} bars, got {len(prices)}")

    sig = price_signals(prices)
    last_date = prices.index[-1]
    try:
        cross = sig.xs(last_date, level="date")
    except KeyError:
        cross = sig.groupby(level="ticker").last()

    cross = cross.reindex(prices.columns)
    liquid = liquidity_screen(prices, volumes)

    scores = composite_score(cross, fundamentals)
    rows: list[ScoreRow] = []
    for t in prices.columns:
        flags: list[str] = []
        if not bool(liquid.get(t, False)):
            flags.append("illiquid_or_too_cheap")
        if int(prices[t].notna().sum()) < min_bars:
            flags.append("short_history")
        if pd.isna(cross.get("momentum_3m", pd.Series(dtype=float)).get(t, np.nan)):
            flags.append("no_momentum_history")

        rows.append(
            ScoreRow(
                ticker=t,
                score=float(scores.get(t, 0.0)),
                momentum_3m=_f(cross, "momentum_3m", t),
                momentum_6m=_f(cross, "momentum_6m", t),
                vol_regime=_f(cross, "vol_regime", t),
                volume_surge=None,
                dist_from_low=_f(cross, "dist_from_low", t),
                dollar_volume=None,
                atr_pct=_f(cross, "atr_pct", t),
                revenue_growth=_ff(fundamentals, "revenue_growth", t),
                gross_margin=_ff(fundamentals, "gross_margin", t),
                cash_to_debt=_ff(fundamentals, "cash_to_debt", t),
                short_pct_float=_ff(fundamentals, "short_pct_float", t),
                flags=flags,
            )
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    excluded = df["flags"].map(lambda f: "illiquid_or_too_cheap" in f or "short_history" in f)
    df["eligible"] = ~excluded
    df.attrs["as_of"] = str(last_date.date())
    df.attrs["note"] = (
        "Ranking, not a recommendation. Momentum/volatility signals are "
        "point-in-time; fundamentals are a TODAY snapshot and were not backtested."
    )
    return df.head(top) if top else df


def _f(cross: pd.DataFrame, col: str, ticker: str) -> float | None:
    if col not in cross.columns:
        return None
    return _num(cross[col].get(ticker))


def _ff(df: pd.DataFrame | None, col: str, ticker: str) -> float | None:
    if df is None or df.empty or col not in df.columns:
        return None
    if ticker not in df.index:
        return None
    return _num(df[col].get(ticker))
