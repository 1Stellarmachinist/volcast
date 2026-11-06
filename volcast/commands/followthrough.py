from __future__ import annotations

import pandas as pd

from ..data import load_prices
from ..followthrough import format_report, study
from ..plot import table
from ._args import resolve_config
from .subpenny import DEFAULT_SUBPENNY


def run(args) -> int:
    cfg = resolve_config(args)
    tickers = getattr(args, "tickers", None) or DEFAULT_SUBPENNY
    tickers = [t.upper() for t in tickers]
    rvol_min = getattr(args, "rvol", None) or 3.0
    period = getattr(args, "period", None) or "2y"

    print(f"studying {len(tickers)} sub-penny names over {period}...")
    print("looking for breakout days, then measuring what happened NEXT.\n")

    try:
        prices = load_prices(tickers, period=period, use_cache=False,
                             allow_synthetic=False, cache_root=cfg.cache_dir)
    except Exception as exc:
        print(f"ERROR fetching prices: {type(exc).__name__}: {exc}")
        return 2

    if prices.empty:
        print("no price data")
        return 3

    volumes = None
    try:
        import yfinance as yf

        raw = yf.download(tickers, period=period, progress=False, threads=False,
                          auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex) and "Volume" in raw.columns.get_level_values(0):
            volumes = raw["Volume"].reindex(columns=prices.columns)
    except Exception as exc:
        print(f"ERROR: volumes unavailable ({type(exc).__name__}); cannot study breakouts")
        return 4

    if volumes is None:
        print("ERROR: volumes unavailable; the study is impossible without them")
        return 4

    result = study(
        prices,
        volumes,
        rvol_min=rvol_min,
        horizons=(1, 3, 5),
        min_events=getattr(args, "min_events", None) or 20,
    )
    print(format_report(result))

    ev = result.events
    if len(ev):
        print("\n--- most recent 12 events ---")
        show = ev.tail(12).copy()
        show["date"] = show["date"].dt.date
        show["price"] = show["price"].map(lambda v: f"{v:.4f}")
        show["rvol"] = show["rvol"].map(lambda v: f"{v:.1f}")
        for c in ("ma20_pos", "ma20_slope"):
            if c in show:
                show[c] = show[c].map(lambda v: f"{v:+.1f}" if pd.notna(v) else "-")
        for c in ("fwd_1d", "fwd_3d", "fwd_5d", "max_5d"):
            if c in show:
                show[c] = show[c].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "-")
        cols = [c for c in ("ticker", "date", "rvol", "price", "fwd_1d", "fwd_3d",
                            "fwd_5d", "max_5d") if c in show]
        print(table(show[cols], floatfmt=".2f"))

        per_ticker = ev.groupby("ticker").agg(
            events=("date", "count"),
            median_5d=("fwd_5d", "median"),
            win_rate_5d=("fwd_5d", lambda s: (s > 0).mean()),
        ).sort_values("events", ascending=False)
        print("\n--- by ticker ---")
        print(per_ticker.head(12).to_string(float_format=lambda v: f"{v:.2f}"))

    return 0
