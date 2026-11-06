"""Breakout scanner for sub-penny and low-priced OTC names.

This is a different tool from ``growth.py``, because it targets a different market
with different mechanics.

WHY THE ARITHMETIC IS DIFFERENT BELOW $1
----------------------------------------
Reg NMS Rule 612 sets the minimum pricing increment at **$0.0001 for stocks priced
under $1.00** (the 1-cent tick only applies at or above $1.00). So the earlier
"1 cent = 172 bps" analysis does not describe this market at all. At $0.0001 a
one-tick move is 100% -- quoting a spread in bps is meaningless. What matters
instead is:

  * **Can you get out?** Not the spread. The ABSENCE of a bid.
  * **Is the volume real?** OTC volume is routinely manufactured.
  * **Does the company survive?** Most sub-penny issuers dilute to zero.

The scanner therefore scores on the three things a volume/MA breakout trader
actually watches -- relative volume, MA position, and MA slope -- and reports
liquidity and dilution risk as hard filters rather than burying them.

HONEST LIMITS
-------------
A volume breakout in a sub-penny name is not a statistical edge; it is a
crowd-behaviour pattern that works until it does not. These instruments have a
structural negative drift because issuance is the business model for many
issuers. Nothing here predicts which breakouts follow through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Sub-penny markets need their own liquidity floor. A name with no volume cannot
# be exited at any price, so it is excluded no matter how good the chart looks.
MIN_DOLLAR_VOLUME = 50_000.0
MIN_AVG_VOLUME = 100_000


@dataclass
class ScanHit:
    ticker: str
    last: float
    rvol: float | None = None            # today's volume vs 50-day average
    ma20_pos: float | None = None        # price / MA20 - 1
    ma50_pos: float | None = None        # price / MA50 - 1
    ma20_slope: float | None = None      # MA20 slope over 5 bars
    ma_cross: bool = False               # price just crossed above MA20
    golden_cross: bool = False           # MA20 crossed above MA50
    dollar_volume: float | None = None
    avg_volume: float | None = None
    pct_off_low: float | None = None     # how far above the 60-day low
    pct_off_high: float | None = None    # how far below the 60-day high
    score: float = 0.0
    warnings: list[str] = field(default_factory=list)
    eligible: bool = True


def _safe(v: float | None) -> float | None:
    if v is None:
        return None
    return float(v) if math.isfinite(float(v)) else None


def relative_volume(volumes: pd.DataFrame, window: int = 50) -> pd.Series:
    """Latest volume divided by its own trailing average (RVOL).

    The single most-watched number in penny-stock scanning. A reading above ~3
    means something unusual is happening today.

    The baseline deliberately EXCLUDES today: comparing a spike against an average
    that already contains the spike understates it (a 5x day reads as 4.6x on a
    50-day window), and understating the one number this scanner exists to find
    would be a quiet, systematic error.
    """
    prior = volumes.shift(1).rolling(window, min_periods=10).mean()
    last_avg = prior.iloc[-1]
    last_vol = volumes.iloc[-1]
    return (last_vol / last_avg.where(last_avg > 0)).rename("rvol")


def moving_average_state(prices: pd.DataFrame, short: int = 20,
                         long: int = 50) -> pd.DataFrame:
    """Per-ticker MA position, slope, and crossover state on the latest bar."""
    ma_s = prices.rolling(short, min_periods=max(5, short // 2)).mean()
    ma_l = prices.rolling(long, min_periods=max(10, long // 2)).mean()  # noqa: F841 -- used below via ma_l[t]

    last = prices.iloc[-1]
    rows = {}
    for t in prices.columns:
        s_now = ma_s[t].iloc[-1]
        l_now = ma_l[t].iloc[-1]
        s_prev = ma_s[t].iloc[-2] if len(ma_s) > 1 else np.nan
        s_5ago = ma_s[t].iloc[-6] if len(ma_s) > 5 else np.nan

        px_now = last.get(t, np.nan)
        px_prev = prices[t].iloc[-2] if len(prices) > 1 else np.nan

        rows[t] = {
            "ma20_pos": _safe((px_now / s_now - 1.0) * 100) if s_now and s_now > 0 else None,
            "ma50_pos": _safe((px_now / l_now - 1.0) * 100) if l_now and l_now > 0 else None,
            "ma20_slope": _safe((s_now / s_5ago - 1.0) * 100) if s_5ago and s_5ago > 0 else None,
            # Crossed above MA20 today but was below yesterday.
            "ma_cross": bool(
                pd.notna(s_now) and pd.notna(px_now) and pd.notna(px_prev)
                and pd.notna(s_prev)
                and px_now > s_now and px_prev <= s_prev
            ),
            # MA20 above MA50 today, was below yesterday.
            "golden_cross": bool(
                pd.notna(s_now) and pd.notna(l_now)
                and pd.notna(s_prev) and pd.notna(ma_l[t].iloc[-2])
                and s_now > l_now and s_prev <= ma_l[t].iloc[-2]
            ),
        }
    return pd.DataFrame(rows).T


def scan(
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None = None,
    min_dollar_volume: float = MIN_DOLLAR_VOLUME,
    min_avg_volume: float = MIN_AVG_VOLUME,
    rvol_min: float = 1.5,
    require_above_ma20: bool = False,
    top: int | None = 20,
) -> pd.DataFrame:
    """Rank names by a volume/MA breakout score, flagging what to avoid."""
    if len(prices) < 25:
        raise ValueError("need at least 25 bars to compute moving averages")

    ma_state = moving_average_state(prices)
    last = prices.iloc[-1]

    if volumes is not None and not volumes.empty:
        # Align on the price index and use volume 0 (not forward-filled) where a
        # name did not trade. Forward-filling volume would carry a stale print
        # across many days and, worse, the two frames have different NaN patterns:
        # prices are ffilled by load_prices but volumes are not, so multiplying
        # them raw silently produces NaN and a near-zero dollar-volume average.
        vol = volumes.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
        rvol = relative_volume(vol)
        avg_vol = vol.rolling(50, min_periods=10).mean().iloc[-1]
        dv = (prices * vol).rolling(20, min_periods=5).mean().iloc[-1]
    else:
        rvol = pd.Series(np.nan, index=prices.columns)
        avg_vol = pd.Series(np.nan, index=prices.columns)
        dv = pd.Series(np.nan, index=prices.columns)

    low60 = prices.rolling(60, min_periods=20).min().iloc[-1]
    high60 = prices.rolling(60, min_periods=20).max().iloc[-1]

    hits: list[ScanHit] = []
    for t in prices.columns:
        px = _safe(last.get(t))
        if px is None:
            continue

        r = _safe(rvol.get(t))
        av = _safe(avg_vol.get(t))
        dvol = _safe(dv.get(t))
        lo = _safe(low60.get(t))
        hi = _safe(high60.get(t))

        warnings: list[str] = []
        eligible = True

        if dvol is None or dvol < min_dollar_volume:
            warnings.append("thin dollar volume - may be impossible to exit")
            eligible = False
        if av is None or av < min_avg_volume:
            warnings.append("very low share volume")
            eligible = False
        if px is not None and px < 0.0001:
            warnings.append("at or below the minimum tick - untradeable")

        hit = ScanHit(
            ticker=t,
            last=px,
            rvol=r,
            ma20_pos=_safe(ma_state.loc[t, "ma20_pos"]) if t in ma_state.index else None,
            ma50_pos=_safe(ma_state.loc[t, "ma50_pos"]) if t in ma_state.index else None,
            ma20_slope=_safe(ma_state.loc[t, "ma20_slope"]) if t in ma_state.index else None,
            ma_cross=bool(ma_state.loc[t, "ma_cross"]) if t in ma_state.index else False,
            golden_cross=bool(ma_state.loc[t, "golden_cross"]) if t in ma_state.index else False,
            dollar_volume=dvol,
            avg_volume=av,
            pct_off_low=_safe((px / lo - 1.0) * 100) if lo and lo > 0 else None,
            pct_off_high=_safe((px / hi - 1.0) * 100) if hi and hi > 0 else None,
            warnings=warnings,
            eligible=eligible,
        )
        hit.score = _score(hit, rvol_min=rvol_min, require_above_ma20=require_above_ma20)
        hits.append(hit)

    df = pd.DataFrame([h.__dict__ for h in hits])
    if df.empty:
        return df

    # Hard filter when requested: breakout traders do not fight the trend.
    if require_above_ma20 and "ma20_pos" in df:
        df = df[(df["ma20_pos"].fillna(-999) > 0) | (df["ma_cross"])]

    df = df.sort_values(["eligible", "score"], ascending=[False, False]).reset_index(drop=True)
    df.attrs["as_of"] = str(prices.index[-1].date())
    df.attrs["disclaimer"] = (
        "Volume/MA breakout scan, not a prediction. Sub-penny OTC names are "
        "frequently diluted to zero; most of these will not survive a year."
    )
    return df.head(top) if top else df


def _score(hit: ScanHit, rvol_min: float, require_above_ma20: bool) -> float:
    """Weighted breakout score. Deliberately simple and inspectable."""
    score = 0.0

    # Volume surge is the primary trigger.
    if hit.rvol is not None:
        if hit.rvol >= rvol_min:
            score += min(hit.rvol, 10.0) * 0.5

    # Price above rising moving averages.
    if hit.ma20_pos is not None and hit.ma20_pos > 0:
        score += 1.0
        if hit.ma20_slope is not None and hit.ma20_slope > 0:
            score += 1.0
    if hit.ma50_pos is not None and hit.ma50_pos > 0:
        score += 0.5

    if hit.ma_cross:
        score += 2.0
    if hit.golden_cross:
        score += 1.5

    # Coming off the low, but not already parabolic.
    if hit.pct_off_low is not None and 20 <= hit.pct_off_low <= 300:
        score += 1.0

    if require_above_ma20 and (hit.ma20_pos is None or hit.ma20_pos <= 0) and not hit.ma_cross:
        score -= 5.0

    return round(float(score), 3)


def dilution_risk_flags(prices: pd.DataFrame, volumes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cheap heuristics for the failure modes that matter in this market.

    These are red flags, not verdicts. A reverse split shows up as a sudden
    upward jump in raw (unadjusted) price; persistent downtrend with heavy volume
    is the classic dilution signature.
    """
    rows = {}
    for t in prices.columns:
        s = prices[t].dropna()
        warns = []
        if len(s) < 60:
            rows[t] = {"warnings": ["short history"], "decline_60d": None, "vol_spike": None}
            continue

        decline = (s.iloc[-1] / s.iloc[-60] - 1.0) * 100 if s.iloc[-60] > 0 else None

        spike = None
        if volumes is not None and t in volumes.columns:
            v = volumes[t].dropna()
            if len(v) >= 50:
                spike = float(v.iloc[-1] / v.tail(50).mean()) if v.tail(50).mean() > 0 else None

        if decline is not None and decline < -50:
            warns.append("down >50% in 60 days - dilution or collapse")
        if spike is not None and spike > 5:
            warns.append("volume spike >5x with weak price - possible distribution")
        if s.iloc[-1] < 0.001:
            warns.append("below $0.001 - one tick is >10% of price")

        rows[t] = {"warnings": warns, "decline_60d": decline, "vol_spike": spike}
    return pd.DataFrame(rows).T
