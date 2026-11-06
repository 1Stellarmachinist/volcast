from __future__ import annotations

import pandas as pd

from ..data import load_prices
from ..growth import fundamental_snapshot, rank_smallcaps
from ..plot import table
from ..ranking import backtest_ranking
from ._args import resolve_config


def run(args) -> int:
    cfg = resolve_config(args)
    tickers = cfg.universe

    prices = load_prices(tickers, period="2y", allow_synthetic=False,
                         cache_root=cfg.cache_dir)

    # Volume is needed for the liquidity screen. Fetch it alongside prices rather
    # than inventing a placeholder column that would silently pass every name.
    volumes = None
    try:
        import yfinance as yf

        vol_raw = yf.download(tickers, period="1y", progress=False, threads=False,
                              auto_adjust=True)
        if isinstance(vol_raw.columns, pd.MultiIndex) and "Volume" in vol_raw.columns.get_level_values(0):
            volumes = vol_raw["Volume"]
    except Exception:
        volumes = None

    fundamentals = None
    if not getattr(args, "no_fundamentals", False):
        print("fetching fundamentals (current snapshot, today only)...")
        fundamentals = fundamental_snapshot(tickers)

    df = rank_smallcaps(prices, volumes=volumes, fundamentals=fundamentals,
                        top=getattr(args, "top", None) or cfg.screen_top)

    print(f"\n=== small-cap ranking as of {df.attrs.get('as_of')} ===")
    cols = ["ticker", "score", "momentum_3m", "vol_regime", "dist_from_low",
            "atr_pct", "revenue_growth", "cash_to_debt", "eligible"]
    cols = [c for c in cols if c in df.columns]
    show = df[cols].copy()
    for c in ("momentum_3m", "vol_regime", "dist_from_low", "atr_pct", "revenue_growth"):
        if c in show:
            show[c] = show[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
    if "cash_to_debt" in show:
        show["cash_to_debt"] = show["cash_to_debt"].map(
            lambda v: f"{v:.2f}" if pd.notna(v) else "-"
        )
    if "score" in show:
        show["score"] = show["score"].map(lambda v: f"{v:+.3f}")
    print(show.to_string(index=False))

    flagged = df[df.get("flags", pd.Series(dtype=object)).map(bool)]
    if len(flagged):
        print("\nexcluded / flagged:")
        for _, r in flagged.iterrows():
            print(f"  {r['ticker']:6s} {', '.join(r['flags'])}")

    print("\n" + "-" * 70)
    print("This is a RANKING, not a recommendation. Fundamentals are a today-snapshot")
    print("and were NOT used in any backtest (that would leak the future).")
    print("-" * 70)

    if getattr(args, "evaluate", False):
        print("\n=== does the ranking predict anything? (price signals only) ===")
        res = backtest_ranking(prices, horizon=cfg.horizon, rebalance=cfg.rebalance)
        s = res.summary
        for phase in ("in_sample", "out_sample", "all"):
            if phase not in s:
                continue
            d = s[phase]
            tstat = d.get("t_stat")
            tstr = f"{tstat:+.2f}" if isinstance(tstat, (int, float)) else "n/a"
            print(f"  {phase:11s} n={d['n']:3d}  excess={d['mean_excess_return']:+.3%}  "
                  f"SE={d['se']:.3%}  t={tstr}  beat-universe={d['hit_rate_beat_universe']:.0%}")
        if not res.buckets.empty:
            print("\n  monotonicity check (forward return by score third):")
            print(table(res.buckets, floatfmt=".4f"))
        print("\n  |t| < 2 means the ranking is not statistically distinguishable from noise.")
    return 0
