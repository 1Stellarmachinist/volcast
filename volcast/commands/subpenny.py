from __future__ import annotations

import pandas as pd

from ..data import load_prices
from ..plot import table
from ..subpenny import dilution_risk_flags, scan
from ._args import resolve_config

# A wider watchlist than configs/penny.toml: these are genuinely sub-$1 OTC names,
# the market the scanner targets. Symbols are illustrative of the universe type,
# not recommendations.
DEFAULT_SUBPENNY = [
    "ILUS", "TPTW", "AABB", "CYBL", "DPLS", "INND", "VPLM", "LTNC",
    "MJNA", "SNPW", "BSSP", "GEGI", "RBNW", "NSAV", "ONCI", "PHIL",
]


def run(args) -> int:
    cfg = resolve_config(args)
    tickers = getattr(args, "tickers", None) or DEFAULT_SUBPENNY
    tickers = [t.upper() for t in tickers]

    print(f"scanning {len(tickers)} sub-$1 OTC names for volume/MA breakouts...")

    try:
        prices = load_prices(tickers, period="6mo", use_cache=False,
                             allow_synthetic=False, cache_root=cfg.cache_dir)
    except Exception as exc:
        print(f"ERROR fetching prices: {type(exc).__name__}: {exc}")
        return 2

    if prices.empty:
        print("no price data returned")
        return 3

    # Volumes are essential here; without them the scanner cannot compute RVOL,
    # which is the whole point. Fail loudly rather than silently degating.
    volumes = None
    try:
        import yfinance as yf

        raw = yf.download(tickers, period="6mo", progress=False, threads=False,
                          auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex) and "Volume" in raw.columns.get_level_values(0):
            volumes = raw["Volume"].reindex(columns=prices.columns)
    except Exception as exc:
        print(f"  warning: volume fetch failed ({type(exc).__name__}); RVOL unavailable")

    if volumes is None:
        print("  WARNING: scoring without RVOL. The ranking is much weaker without volume.")

    try:
        df = scan(
            prices,
            volumes=volumes,
            rvol_min=getattr(args, "rvol", 1.5) or 1.5,
            require_above_ma20=getattr(args, "above_ma20", False),
            top=getattr(args, "top", None) or 20,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 4

    if df.empty:
        print("nothing matched")
        return 0

    print(f"\n=== breakout scan as of {df.attrs.get('as_of')} ===")

    show = df[["ticker", "last", "score", "rvol", "ma20_pos", "ma50_pos",
               "ma20_slope", "dollar_volume", "eligible"]].copy()
    show["last"] = show["last"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "-")
    show["score"] = show["score"].map(lambda v: f"{v:+.1f}")
    for c in ("rvol", "ma20_pos", "ma50_pos", "ma20_slope"):
        if c in show:
            show[c] = show[c].map(lambda v: f"{v:+.1f}" if pd.notna(v) else "-")
    if "dollar_volume" in show:
        show["dollar_volume"] = show["dollar_volume"].map(
            lambda v: f"{v/1000:,.0f}k" if pd.notna(v) else "-"
        )
    print(table(show, floatfmt=".2f"))

    flagged = df[df["warnings"].map(bool)]
    if len(flagged):
        print("\n--- risk flags ---")
        for _, r in flagged.iterrows():
            print(f"  {r['ticker']:6s} {', '.join(r['warnings'])}")

    # Dilution heuristics on the top names only, to keep the output readable.
    top_tickers = list(df["ticker"].head(8))
    if top_tickers:
        print("\n--- dilution / collapse check (top names) ---")
        risk = dilution_risk_flags(prices[top_tickers], volumes)
        for t in top_tickers:
            w = risk.loc[t, "warnings"] if t in risk.index else []
            d = risk.loc[t, "decline_60d"] if t in risk.index else None
            dstr = f"{d:+.0f}% over 60d" if d is not None and pd.notna(d) else "n/a"
            note = "; ".join(w) if len(w) else "no flags"
            print(f"  {t:6s} {dstr:>18s}  {note}")

    print("\n" + "-" * 70)
    print(df.attrs["disclaimer"])
    print("RVOL compares today's volume to the prior 50-day average, excluding today.")
    print("A high score means 'unusual activity + uptrend', NOT 'will go up'.")
    print("-" * 70)
    return 0
