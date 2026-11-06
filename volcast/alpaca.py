"""Alpaca market data client: bars, quotes, and spread measurement.

Keys come from the environment (never from source):
    ALPACA_API_KEY_ID
    ALPACA_API_SECRET_KEY

FEED RULES, which matter more than the API shape
-------------------------------------------------
Alpaca's free "Basic" plan gives:

  * ``feed=iex``  -- real time, but IEX is ONE exchange and carries roughly 2.5%
    of US consolidated volume. For microcaps it frequently has no quote at all,
    and when it does the spread is not the NBBO you would actually trade at.
  * ``feed=sip``  -- the consolidated tape, but only for data **older than 15
    minutes**. Query something newer and you get HTTP 403 with
    "subscription does not permit querying recent SIP data".

Consequence for this project: use SIP for *spread statistics* (historical), which
is what feeds the cost model, and never treat an IEX quote as "the" spread on a
penny stock. ``effective_feed`` encodes that policy rather than leaving it to the
caller to remember.
"""

from __future__ import annotations

import datetime as dt
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DATA_HOST = "https://data.alpaca.markets"
PAPER_HOST = "https://paper-api.alpaca.markets"

# SIP data is only queryable once it is this old on the free plan.
SIP_EMBARGO = dt.timedelta(minutes=15)


class AlpacaError(RuntimeError):
    """Raised for auth/plan/parameter problems, with the server message intact."""


@dataclass
class Credentials:
    key_id: str
    secret: str

    @classmethod
    def from_env(cls) -> Credentials:
        kid = os.environ.get("ALPACA_API_KEY_ID", "").strip()
        sec = os.environ.get("ALPACA_API_SECRET_KEY", "").strip()
        if not kid or not sec:
            raise AlpacaError(
                "ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY are not set.\n"
                "Create them at https://app.alpaca.markets/paper/dashboard/overview "
                "(the 'API Keys' box) and put them in .env, which is gitignored."
            )
        return cls(kid, sec)

    def headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret,
            "accept": "application/json",
        }


def effective_feed(feed: str, end: dt.datetime | None, now: dt.datetime | None = None) -> str:
    """Pick a feed the free plan is actually allowed to serve.

    Asking for ``sip`` with a recent ``end`` returns 403 on Basic, so we downgrade
    to IEX and let the caller know via ``feed_used`` on the returned frame.
    """
    if feed != "sip":
        return feed
    now = now or dt.datetime.now(dt.timezone.utc)
    if end is None:
        # "latest" with SIP is always refused on Basic.
        return "iex"
    end_utc = end if end.tzinfo else end.replace(tzinfo=dt.timezone.utc)
    return "sip" if (now - end_utc) >= SIP_EMBARGO else "iex"


def _request(path: str, params: dict[str, Any], creds: Credentials,
             timeout: float = 20.0, retries: int = 3) -> dict:
    """GET with modest retry on 429/5xx. Raises AlpacaError with the real message."""
    try:
        import httpx
    except Exception as exc:  # pragma: no cover
        raise AlpacaError("httpx is required for Alpaca access") from exc

    url = f"{DATA_HOST}{path}"
    clean = {k: v for k, v in params.items() if v is not None}
    last: Exception | None = None

    for attempt in range(retries):
        try:
            r = httpx.get(url, params=clean, headers=creds.headers(), timeout=timeout)
        except Exception as exc:
            last = exc
            time.sleep(0.4 * (attempt + 1))
            continue

        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            time.sleep(1.0 * (attempt + 1))
            last = AlpacaError("rate limited (429)")
            continue
        if r.status_code == 403:
            body = r.text[:300]
            raise AlpacaError(
                f"403 from Alpaca: {body}\n"
                "If it mentions SIP, you asked for consolidated data newer than "
                "15 minutes on the free plan. Use feed='iex' or an older `end`."
            )
        raise AlpacaError(f"HTTP {r.status_code} from {path}: {r.text[:300]}")

    raise AlpacaError(f"request to {path} failed after {retries} attempts: {last}")


# --------------------------------------------------------------------------- #
# Bars -> wide price frame
# --------------------------------------------------------------------------- #
def get_bars(
    symbols: Iterable[str],
    timeframe: str = "1Day",
    start: dt.date | dt.datetime | None = None,
    end: dt.date | dt.datetime | None = None,
    feed: str = "sip",
    limit: int = 10_000,
    creds: Credentials | None = None,
) -> tuple[pd.DataFrame, str]:
    """Fetch bars and return ``(wide_close_frame, feed_used)``.

    Paginates automatically; Alpaca returns a ``next_page_token`` per request.
    Free-tier daily bars older than 15 minutes are served from SIP, so daily
    history is complete even without a paid plan.
    """
    creds = creds or Credentials.from_env()
    syms = [s.upper() for s in symbols]
    if not syms:
        return pd.DataFrame(), feed

    use_feed = effective_feed(feed, end)

    def _fmt(d: dt.date | dt.datetime | None) -> str | None:
        if d is None:
            return None
        if isinstance(d, dt.datetime):
            return d.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return d.isoformat()

    rows: list[dict] = []
    page_token: str | None = None

    while True:
        payload = _request(
            "/v2/stocks/bars",
            {
                "symbols": ",".join(syms),
                "timeframe": timeframe,
                "start": _fmt(start),
                "end": _fmt(end),
                "feed": use_feed,
                "limit": limit,
                "adjustment": "all",
                "page_token": page_token,
            },
            creds,
        )
        bars = payload.get("bars") or {}
        for sym, series in bars.items():
            for b in series:
                rows.append({"timestamp": b["t"], "ticker": sym, "close": b["c"],
                             "open": b["o"], "high": b["h"], "low": b["l"],
                             "volume": b["v"]})
        page_token = payload.get("next_page_token")
        if not page_token:
            break

    if not rows:
        return pd.DataFrame(), use_feed

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    wide = df.pivot_table(index="timestamp", columns="ticker", values="close", aggfunc="last")
    wide = wide.sort_index().ffill()
    wide.attrs["feed_used"] = use_feed
    return wide, use_feed


def get_volumes(symbols: Iterable[str], **kwargs) -> pd.DataFrame:
    """Wide volume frame, for the liquidity screen."""
    creds = kwargs.pop("creds", None) or Credentials.from_env()
    syms = [s.upper() for s in symbols]
    end = kwargs.get("end")
    use_feed = effective_feed(kwargs.get("feed", "sip"), end)

    def _fmt(d):
        if d is None:
            return None
        if isinstance(d, dt.datetime):
            return d.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return d.isoformat()

    payload = _request(
        "/v2/stocks/bars",
        {
            "symbols": ",".join(syms),
            "timeframe": kwargs.get("timeframe", "1Day"),
            "start": _fmt(kwargs.get("start")),
            "end": _fmt(end),
            "feed": use_feed,
            "limit": 10_000,
            "adjustment": "all",
        },
        creds,
    )
    rows = []
    for sym, series in (payload.get("bars") or {}).items():
        for b in series:
            rows.append({"timestamp": b["t"], "ticker": sym, "volume": b["v"]})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    return df.pivot_table(index="timestamp", columns="ticker", values="volume", aggfunc="last").sort_index()


# --------------------------------------------------------------------------- #
# Quotes -> bid/ask and spread statistics
# --------------------------------------------------------------------------- #
def get_latest_quotes(symbols: Iterable[str], feed: str = "iex",
                      creds: Credentials | None = None) -> pd.DataFrame:
    """Latest NBBO per symbol.

    Defaults to IEX because the free plan refuses recent SIP. On a microcap this
    may return nothing, which is information: it means no IEX participant was
    quoting, not that the stock is untradeable.
    """
    creds = creds or Credentials.from_env()
    syms = [s.upper() for s in symbols]
    rows = []
    # The snapshot endpoint is 1 symbol per call on Basic; batch by repeating.
    for i in range(0, len(syms), 1):
        batch = syms[i : i + 1]
        payload = _request(
            "/v2/stocks/quotes/latest",
            {"symbols": ",".join(batch), "feed": feed},
            creds,
        )
        quotes = payload.get("quotes") or {}
        for sym, q in quotes.items():
            if not q:
                continue
            bid = q.get("bp")
            ask = q.get("ap")
            rows.append(
                {
                    "ticker": sym,
                    "bid": bid,
                    "ask": ask,
                    "bid_size": q.get("bs"),
                    "ask_size": q.get("as"),
                    "mid": ((bid + ask) / 2.0) if (bid and ask) else None,
                    "spread": (ask - bid) if (bid and ask) else None,
                    "spread_bps": (
                        (ask - bid) / ((ask + bid) / 2.0) * 10_000.0
                        if (bid and ask and (ask + bid) > 0)
                        else None
                    ),
                    "timestamp": q.get("t"),
                }
            )
    df = pd.DataFrame(rows)
    df.attrs["feed_used"] = feed
    return df


def get_quotes_range(
    symbol: str,
    start: dt.datetime,
    end: dt.datetime,
    feed: str = "sip",
    limit: int = 10_000,
    creds: Credentials | None = None,
) -> pd.DataFrame:
    """Historical quotes for one symbol, for measuring the typical spread.

    This is the valuable call on the free plan: historical SIP quotes older than
    15 minutes are permitted, and a spread distribution is what the cost model
    actually needs.
    """
    creds = creds or Credentials.from_env()
    use_feed = effective_feed(feed, end)

    def _fmt(d: dt.datetime) -> str:
        return d.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows: list[dict] = []
    token: str | None = None
    while True:
        payload = _request(
            f"/v2/stocks/{symbol.upper()}/quotes",
            {
                "start": _fmt(start),
                "end": _fmt(end),
                "feed": use_feed,
                "limit": limit,
                "page_token": token,
            },
            creds,
        )
        for q in payload.get("quotes") or []:
            bp, ap = q.get("bp"), q.get("ap")
            if not bp or not ap:
                continue
            mid = (bp + ap) / 2.0
            rows.append(
                {
                    "timestamp": q["t"],
                    "bid": bp,
                    "ask": ap,
                    "mid": mid,
                    "spread_bps": (ap - bp) / mid * 10_000.0 if mid > 0 else np.nan,
                }
            )
        token = payload.get("next_page_token")
        if not token:
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df.attrs["feed_used"] = use_feed
    return df


def spread_stats(quotes: pd.DataFrame) -> dict:
    """Summarise a quote series into numbers a cost model can use."""
    if quotes.empty or "spread_bps" not in quotes:
        return {"n": 0}

    s = quotes["spread_bps"].replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {"n": 0}

    # One-way cost of crossing the spread is ~half the quoted spread.
    half = s / 2.0
    return {
        "n": int(len(s)),
        "median_spread_bps": float(s.median()),
        "p75_spread_bps": float(s.quantile(0.75)),
        "p90_spread_bps": float(s.quantile(0.90)),
        "median_half_spread_bps": float(half.median()),
        "mean_spread_bps": float(s.mean()),
    }


def suggest_cost_bps(quotes_by_symbol: dict[str, pd.DataFrame],
                     safety_multiple: float = 1.5) -> float:
    """Turn measured spreads into a ``cost_bps`` for the backtester.

    Crossing the spread costs about half the quoted spread each way. A round trip
    is therefore ~1x the quoted spread in bps. The safety multiple covers
    slippage and the fact that you rarely get the midpoint.
    """
    medians = []
    for df in quotes_by_symbol.values():
        st = spread_stats(df)
        if st.get("n", 0) >= 10:
            medians.append(st["median_half_spread_bps"])

    if not medians:
        return float("nan")
    return float(np.median(medians) * safety_multiple)


def list_tradable_assets(creds: Credentials | None = None,
                         asset_class: str = "us_equity") -> pd.DataFrame:
    """All active, tradable US equities. Useful for building a broad universe."""
    creds = creds or Credentials.from_env()
    try:
        import httpx
    except Exception as exc:  # pragma: no cover
        raise AlpacaError("httpx is required") from exc

    r = httpx.get(
        f"{PAPER_HOST}/v2/assets",
        params={"status": "active", "asset_class": asset_class},
        headers=creds.headers(),
        timeout=30.0,
    )
    if r.status_code != 200:
        raise AlpacaError(f"assets endpoint HTTP {r.status_code}: {r.text[:200]}")

    df = pd.DataFrame(r.json())
    if df.empty:
        return df
    keep = [c for c in ("symbol", "name", "exchange", "tradable", "marginable",
                        "shortable", "fractionable", "easy_to_borrow") if c in df.columns]
    return df[keep]
