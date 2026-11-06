"""Does the small-cap ranking actually predict anything?

This module exists to answer that question with numbers instead of hope. It
deliberately uses ONLY the point-in-time price/volume signals, because
fundamentals from yfinance are a today-snapshot and would leak the future.

The output you should care about is ``hit_rate``: of the N names the ranking put
in its top bucket, how many went up over the following H days, versus the base
rate for the whole universe. If the lift is not clearly positive, the ranking is
decoration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .growth import MIN_BARS, composite_score, price_signals


@dataclass
class RankingBacktest:
    buckets: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "summary": self.summary,
            "buckets": self.buckets.to_dict(orient="records"),
        }


def _forward_return(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forward H-day log return, aligned so row t looks forward from t."""
    return returns.rolling(horizon).sum().shift(-horizon)


def backtest_ranking(
    prices: pd.DataFrame,
    horizon: int = 5,
    rebalance: int = 5,
    top_n: int = 3,
    min_bars: int = MIN_BARS,
    train_frac: float = 0.5,
) -> RankingBacktest:
    """Walk forward through history ranking names, then measure what happened.

    Deliberately simple and conservative:
      * signals recomputed from data up to each rebalance date only
      * a name must have full history at that point
      * returns are equal-weighted, top-N vs universe average
      * no compounding of position sizing, so this isolates skill in the ranking
    """
    if len(prices) < min_bars + horizon + 20:
        raise ValueError("not enough history to evaluate a ranking")

    rets = np.log(prices).diff()
    fwd = _forward_return(rets, horizon)
    sig = price_signals(prices)

    dates = prices.index
    start = min_bars
    end = len(dates) - horizon
    test_dates = list(dates[start:end:max(rebalance, 1)])

    split = int(len(test_dates) * train_frac)
    in_sample = set(test_dates[:split])

    rows = []
    for dt in test_dates:
        try:
            cross = sig.xs(dt, level="date")
        except KeyError:
            continue

        available = [
            t for t in prices.columns
            if prices[t].iloc[: dates.get_loc(dt) + 1].notna().sum() >= min_bars
        ]
        if len(available) < top_n + 2:
            continue

        cross = cross.reindex(available)
        scores = composite_score(cross).sort_values(ascending=False)

        future = fwd.loc[dt, available].dropna()
        if future.empty:
            continue

        picked = [t for t in scores.index[:top_n] if t in future.index]
        if not picked:
            continue

        univ_mean = float(future.mean())
        top_mean = float(future.loc[picked].mean())
        rows.append(
            {
                "date": dt,
                "n_available": len(available),
                "picks": ",".join(picked),
                "top_mean": top_mean,
                "universe_mean": univ_mean,
                "excess": top_mean - univ_mean,
                "top_up": bool(top_mean > 0),
                "universe_up": bool(univ_mean > 0),
                "phase": "in_sample" if dt in in_sample else "out_sample",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return RankingBacktest(summary={"error": "no evaluable rebalances"})

    summary: dict = {"rebalances": int(len(df)), "horizon_days": horizon, "top_n": top_n}

    for phase in ("in_sample", "out_sample", "all"):
        sub = df if phase == "all" else df[df["phase"] == phase]
        if sub.empty:
            continue
        excess = sub["excess"]
        mean_excess = float(excess.mean())
        sd = float(excess.std(ddof=1)) if len(excess) > 1 else float("nan")
        se = sd / math.sqrt(len(excess)) if len(excess) > 1 and sd > 0 else float("nan")
        hit = float((excess > 0).mean())
        summary[phase] = {
            "n": int(len(sub)),
            "mean_excess_return": mean_excess,
            "std": sd,
            "se": se,
            "t_stat": mean_excess / se if se and math.isfinite(se) and se > 0 else None,
            "hit_rate_beat_universe": hit,
            "top_abs_up_rate": float(sub["top_up"].mean()),
            "universe_up_rate": float(sub["universe_up"].mean()),
            "mean_top_return": float(sub["top_mean"].mean()),
            "mean_universe_return": float(sub["universe_mean"].mean()),
        }

    # Bucket view: does score monotonicity hold? Split the cross-section into
    # thirds at each rebalance and average forward return per third.
    bucket_rows = []
    for dt in test_dates:
        try:
            cross = sig.xs(dt, level="date")
        except KeyError:
            continue
        loc = dates.get_loc(dt)
        available = [t for t in prices.columns
                     if prices[t].iloc[: loc + 1].notna().sum() >= min_bars]
        if len(available) < 6:
            continue
        cross = cross.reindex(available)
        scores = composite_score(cross)
        future = fwd.loc[dt, available].dropna()
        common = scores.index.intersection(future.index)
        if len(common) < 6:
            continue
        s = scores.loc[common].sort_values(ascending=False)
        n = len(s)
        third = max(n // 3, 1)
        bucket_rows.append(
            {
                "date": dt,
                "top_third": float(future.loc[s.index[:third]].mean()),
                "mid_third": float(future.loc[s.index[third : 2 * third]].mean()),
                "bottom_third": float(future.loc[s.index[2 * third :]].mean()),
            }
        )

    buckets = pd.DataFrame(bucket_rows)
    if not buckets.empty:
        buckets = pd.DataFrame(
            [
                {
                    "bucket": "top_third",
                    "mean_fwd_return": float(buckets["top_third"].mean()),
                    "n": int(len(buckets)),
                },
                {
                    "bucket": "mid_third",
                    "mean_fwd_return": float(buckets["mid_third"].mean()),
                    "n": int(len(buckets)),
                },
                {
                    "bucket": "bottom_third",
                    "mean_fwd_return": float(buckets["bottom_third"].mean()),
                    "n": int(len(buckets)),
                },
            ]
        )
        summary["monotonic"] = bool(
            buckets.iloc[0]["mean_fwd_return"] > buckets.iloc[1]["mean_fwd_return"] > buckets.iloc[2]["mean_fwd_return"]
        )

    return RankingBacktest(buckets=buckets, summary=summary)
