from __future__ import annotations

import datetime as dt

import pandas as pd

from ..alpaca import (
    AlpacaError,
    Credentials,
    get_bars,
    get_quotes_range,
    get_volumes,
    list_tradable_assets,
    spread_stats,
    suggest_cost_bps,
)
from ..growth import rank_smallcaps
from ..plot import table
from ._args import resolve_config


def run(args) -> int:
    try:
        creds = Credentials.from_env()
    except AlpacaError as exc:
        print(f"ERROR: {exc}")
        return 2

    cfg = resolve_config(args)
    lookback_days = getattr(args, "days", 400) or 400
    tickers = list(cfg.universe)

    print("checking credentials...")
    try:
        assets = list_tradable_assets(creds)
        print(f"  ok - {len(assets):,} tradable US equities visible")
    except AlpacaError as exc:
        print(f"ERROR: {exc}")
        return 3

    if getattr(args, "universe", False):
        tradable = set(assets["symbol"])
        missing = [t for t in tickers if t not in tradable]
        if missing:
            print(f"  warning: not in Alpaca's tradable list: {missing}")

    end = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=20)  # stay inside SIP embargo
    # Calendar days -> trading days is roughly 5/7, so ask for 2.6x to be safe.
    # The old 2x factor returned 85 bars for a "60 day" request and then the
    # ranking stage rejected them for having too little history.
    start = end - dt.timedelta(days=int(lookback_days * 2.6) + 30)

    print(f"\nfetching {lookback_days}d of daily bars...")
    try:
        prices, feed_used = get_bars(tickers, "1Day", start, end, feed="sip", creds=creds)
    except AlpacaError as exc:
        print(f"ERROR: {exc}")
        return 4

    if prices.empty:
        print("no bars returned")
        return 5
    print(f"  {prices.shape[0]} bars x {prices.shape[1]} tickers (feed={feed_used})")

    try:
        volumes = get_volumes(tickers, start=start, end=end, feed="sip", creds=creds)
    except AlpacaError:
        volumes = None

    # ------------------------------------------------------------------ #
    # Measure real spreads. This is the point of the exercise: replace an
    # assumed cost with a measured one.
    # ------------------------------------------------------------------ #
    q_start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=5)
    q_end = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=20)

    print("\nmeasuring spreads from historical SIP quotes...")
    quotes_by_symbol: dict[str, pd.DataFrame] = {}
    rows = []
    for t in tickers:
        try:
            q = get_quotes_range(t, q_start, q_end, feed="sip", limit=5000, creds=creds)
        except AlpacaError as exc:
            rows.append({"ticker": t, "n": 0, "median_bps": None, "note": str(exc)[:40]})
            continue
        quotes_by_symbol[t] = q
        st = spread_stats(q)
        rows.append(
            {
                "ticker": t,
                "n": st.get("n", 0),
                "median_bps": st.get("median_spread_bps"),
                "p90_bps": st.get("p90_spread_bps"),
                "half_bps": st.get("median_half_spread_bps"),
                "note": "" if st.get("n", 0) >= 10 else "too few quotes",
            }
        )

    spreads = pd.DataFrame(rows)
    print(table(spreads, floatfmt=".1f"))

    measured = suggest_cost_bps(quotes_by_symbol)
    if measured == measured:  # not NaN
        print(f"\nMEASURED round-trip cost estimate: {measured:.0f} bps "
              f"(median half-spread x 1.5 safety)")
        print(f"  your config currently assumes: {cfg.cost_bps:.0f} bps")
        if abs(measured - cfg.cost_bps) > 20:
            print(f"  -> configs/penny.toml should use cost_bps = {measured:.0f}")
    else:
        print("\ncould not measure spreads (too few quotes); keeping configured cost")

    # ------------------------------------------------------------------ #
    # Rank with real bars
    # ------------------------------------------------------------------ #
    print("\nranking on Alpaca bars...")
    try:
        ranked = rank_smallcaps(prices, volumes=volumes, fundamentals=None,
                                top=getattr(args, "top", None) or cfg.screen_top)
        cols = [c for c in ("ticker", "score", "momentum_3m", "vol_regime",
                            "atr_pct", "eligible") if c in ranked.columns]
        print(table(ranked[cols], floatfmt=".4f"))
    except Exception as exc:
        print(f"  ranking failed: {type(exc).__name__}: {exc}")

    return 0
