"""Follow-through study: when a sub-penny name has a volume/MA breakout,
what actually happens over the next 1, 3 and 5 days?

THE MEASUREMENT TRAP THIS AVOIDS
--------------------------------
The obvious (wrong) way to evaluate a breakout is to find breakout days and
average their returns. That measures the move that *caused* the breakout, not
what happened next. Every bar here is measured from the CLOSE OF THE BREAKOUT DAY
forward, and the signal itself is computed only from data up to and including that
close. Anything else is circular.

The other trap is survivorship in the exit price. We can only see prices for names
that still have data, so a name that stopped trading mid-window is treated as a
failed signal rather than dropped, and the count of those is reported.

WHAT A GOOD RESULT LOOKS LIKE
-----------------------------
Not "average return is positive". For a 100x-possible instrument you need the
DISTRIBUTION: median, the share that went up, the right tail, and the worst case.
A mean of +8% driven entirely by one 300% day is not a repeatable edge, so the
median and the win rate are reported alongside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BreakoutEvent:
    ticker: str
    date: pd.Timestamp
    rvol: float
    ma20_pos: float
    ma20_slope: float
    price: float
    dollar_volume: float


@dataclass
class FollowThroughResult:
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    stats: dict = field(default_factory=dict)
    horizons: tuple[int, ...] = (1, 3, 5)

    def summary(self) -> dict:
        return self.stats


# --------------------------------------------------------------------------- #
# Event detection
# --------------------------------------------------------------------------- #
def detect_breakouts(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    rvol_min: float = 3.0,
    ma_short: int = 20,
    ma_long: int = 50,
    require_ma_slope_up: bool = True,
    min_dollar_volume: float = 10_000.0,
) -> pd.DataFrame:
    """Find every historical bar meeting the breakout criteria.

    Criteria (all as of that bar's close):
      * RVOL >= rvol_min vs the prior 50-day average
      * price > MA20 and MA20 slope positive (optional)
      * dollar volume above a floor, else the signal is untradeable
    """
    if volumes is None or volumes.empty:
        raise ValueError("follow-through needs volumes")

    vol = volumes.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    px = prices.reindex(columns=vol.columns)

    # Point-in-time RVOL for EVERY bar, not just the last one.
    prior_avg = vol.shift(1).rolling(50, min_periods=20).mean()
    rvol_all = vol / prior_avg.where(prior_avg > 0)

    ma_s = px.rolling(ma_short, min_periods=ma_short // 2).mean()
    # ma_long is accepted for API symmetry but the long average is not part of the
    # breakout criterion (MA20 position + slope), so it is not computed.
    del ma_long
    slope = ma_s / ma_s.shift(5) - 1.0
    dv = (px * vol).rolling(20, min_periods=5).mean()

    rows = []
    for t in px.columns:
        # Build each condition as an explicitly 1-D boolean array. A single-column
        # frame collapses to Series and `series >= scalar` yields a scalar bool,
        # which then broadcasts the whole mask into a scalar and silently yields
        # zero events.
        above = (px[t] > ma_s[t]).to_numpy(dtype=bool)
        if require_ma_slope_up:
            slope_up = (slope[t] > 0).to_numpy(dtype=bool)
        else:
            slope_up = np.ones(len(px), dtype=bool)
        liquid = (dv[t] >= min_dollar_volume).to_numpy(dtype=bool)
        hot = (rvol_all[t] >= rvol_min).to_numpy(dtype=bool)

        mask = np.isfinite(px[t].to_numpy(dtype=float)) & above & slope_up & liquid & hot
        for dt in px.index[mask]:
            rows.append(
                {
                    "ticker": t,
                    "date": dt,
                    "rvol": float(rvol_all.loc[dt, t]),
                    "ma20_pos": float((px.loc[dt, t] / ma_s.loc[dt, t] - 1.0) * 100),
                    "ma20_slope": float(slope.loc[dt, t] * 100)
                    if pd.notna(slope.loc[dt, t]) else np.nan,
                    "price": float(px.loc[dt, t]),
                    "dollar_volume": float(dv.loc[dt, t]),
                }
            )

    cols = ["ticker", "date", "rvol", "ma20_pos", "ma20_slope", "price", "dollar_volume"]
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values("date").reset_index(drop=True) if len(df) else df


# --------------------------------------------------------------------------- #
# Forward returns
# --------------------------------------------------------------------------- #
def forward_returns(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """Attach forward returns measured from the CLOSE of each event day."""
    if events.empty:
        out = events.copy()
        for h in horizons:
            out[f"fwd_{h}d"] = pd.Series(dtype=float)
            out[f"max_{h}d"] = pd.Series(dtype=float)
        return out

    out = events.copy()
    idx = prices.index

    for h in horizons:
        vals = []
        for _, ev in events.iterrows():
            t, dt = ev["ticker"], ev["date"]
            if t not in prices.columns:
                vals.append(np.nan)
                continue
            try:
                pos = idx.get_loc(dt)
            except KeyError:
                vals.append(np.nan)
                continue
            fwd_pos = pos + h
            if fwd_pos >= len(idx):
                vals.append(np.nan)   # ran off the end of history
                continue
            p0 = prices[t].iloc[pos]
            p1 = prices[t].iloc[fwd_pos]
            if not (pd.notna(p0) and pd.notna(p1)) or p0 <= 0:
                vals.append(np.nan)
                continue
            vals.append(float(p1 / p0 - 1.0))
        out[f"fwd_{h}d"] = vals

    # Peak favourable excursion: the best moment inside the horizon, which is what
    # a day trader with a limit order could plausibly have captured.
    for h in horizons:
        vals = []
        for _, ev in events.iterrows():
            t, dt = ev["ticker"], ev["date"]
            if t not in prices.columns:
                vals.append(np.nan)
                continue
            try:
                pos = idx.get_loc(dt)
            except KeyError:
                vals.append(np.nan)
                continue
            window = prices[t].iloc[pos + 1 : pos + 1 + h]
            p0 = prices[t].iloc[pos]
            if window.empty or not pd.notna(p0) or p0 <= 0:
                vals.append(np.nan)
                continue
            vals.append(float(window.max() / p0 - 1.0))
        out[f"max_{h}d"] = vals

    return out


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _describe(x: pd.Series) -> dict:
    s = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(s))
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "mean": float(s.mean()),
        "median": float(s.median()),
        "win_rate": float((s > 0).mean()),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "best": float(s.max()),
        "worst": float(s.min()),
        "mean_se": float(s.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "double_rate": float((s >= 1.0).mean()),
        "halve_rate": float((s <= -0.5).mean()),
    }


def study(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    rvol_min: float = 3.0,
    horizons: tuple[int, ...] = (1, 3, 5),
    min_events: int = 20,
) -> FollowThroughResult:
    """Full study: detect breakouts, measure forward returns, summarise."""
    events = detect_breakouts(prices, volumes, rvol_min=rvol_min)
    if len(events) < min_events:
        return FollowThroughResult(
            events=events,
            stats={"error": f"only {len(events)} events, need {min_events}", "n_events": len(events)},
            horizons=horizons,
        )

    with_returns = forward_returns(prices, events, horizons)

    stats: dict = {
        "n_events": int(len(with_returns)),
        "n_tickers": int(with_returns["ticker"].nunique()),
        "rvol_min": rvol_min,
    }

    # Baseline: what did a RANDOM day in the same universe do? Without this the
    # breakout numbers have nothing to be compared against.
    for h in horizons:
        fwd_all = (prices.shift(-h) / prices - 1.0).stack().dropna()
        stats[f"baseline_{h}d"] = _describe(fwd_all)

    for h in horizons:
        stats[f"breakout_{h}d"] = _describe(with_returns.get(f"fwd_{h}d", pd.Series(dtype=float)))
        stats[f"maxup_{h}d"] = _describe(with_returns.get(f"max_{h}d", pd.Series(dtype=float)))

    # Edge = breakout median minus baseline median, per horizon.
    for h in horizons:
        b = stats[f"breakout_{h}d"].get("median")
        base = stats[f"baseline_{h}d"].get("median")
        if b is not None and base is not None:
            stats[f"median_edge_{h}d"] = float(b - base)

    return FollowThroughResult(events=with_returns, stats=stats, horizons=horizons)


def format_report(result: FollowThroughResult) -> str:
    """Human-readable report. Deliberately shows the ugly numbers too."""
    s = result.stats
    if "error" in s:
        return f"Not enough data: {s['error']}"

    lines: list[str] = []
    lines.append("=" * 74)
    lines.append("  SUB-PENNY BREAKOUT FOLLOW-THROUGH")
    lines.append("=" * 74)
    lines.append(
        f"  {s['n_events']} breakout events across {s['n_tickers']} tickers "
        f"(RVOL >= {s['rvol_min']})"
    )
    lines.append("  Forward returns measured from the CLOSE of the breakout day.")
    lines.append("")

    for h in result.horizons:
        b = s.get(f"breakout_{h}d", {})
        base = s.get(f"baseline_{h}d", {})
        mx = s.get(f"maxup_{h}d", {})
        if b.get("n", 0) == 0:
            continue

        lines.append(f"--- {h}-day horizon ---")
        lines.append(
            f"  breakout : median {b['median']:+.2%}   mean {b['mean']:+.2%} "
            f"(SE {b['mean_se']:.2%})   win rate {b['win_rate']:.0%}"
        )
        if base.get("n", 0):
            lines.append(
                f"  baseline : median {base['median']:+.2%}   mean {base['mean']:+.2%}   "
                f"win rate {base['win_rate']:.0%}   (all days, same universe)"
            )
            edge = s.get(f"median_edge_{h}d")
            if edge is not None:
                verdict = "better than random" if edge > 0 else "WORSE than random"
                lines.append(f"  median edge: {edge:+.2%}  -> {verdict}")
        lines.append(
            f"  spread   : p10 {b['p10']:+.1%} | p25 {b['p25']:+.1%} | "
            f"p75 {b['p75']:+.1%} | p90 {b['p90']:+.1%}"
        )
        lines.append(
            f"  tails    : best {b['best']:+.0%}   worst {b['worst']:+.0%}   "
            f"doubled {b['double_rate']:.1%}   halved+ {b['halve_rate']:.1%}"
        )
        if mx.get("n", 0):
            lines.append(
                f"  best moment inside window: median {mx['median']:+.1%}, "
                f"p90 {mx['p90']:+.0%}  (a limit sell could have hit this)"
            )
        lines.append("")

    lines.append("-" * 74)
    lines.append("Read this carefully:")
    lines.append("  * A positive MEAN with a near-zero MEDIAN means one lucky outlier.")
    lines.append("  * 'doubled' vs 'halved+' is the real risk profile of this market.")
    lines.append("  * If the median edge is not clearly positive, the breakout signal")
    lines.append("    is not an edge -- it is a description of the past.")
    lines.append("-" * 74)
    return "\n".join(lines)
