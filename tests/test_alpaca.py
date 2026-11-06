from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from volcast.alpaca import (
    SIP_EMBARGO,
    AlpacaError,
    Credentials,
    effective_feed,
    spread_stats,
    suggest_cost_bps,
)


# --------------------------------------------------------------------------- #
# Feed selection: the rule that prevents 403s on the free plan
# --------------------------------------------------------------------------- #
def test_recent_sip_is_downgraded_to_iex():
    """Free plan refuses SIP newer than 15 min; asking anyway gives a 403."""
    now = dt.datetime(2026, 11, 6, 18, 0, tzinfo=dt.timezone.utc)
    end = now - dt.timedelta(minutes=5)
    assert effective_feed("sip", end, now) == "iex"


def test_old_sip_is_allowed():
    now = dt.datetime(2026, 11, 6, 18, 0, tzinfo=dt.timezone.utc)
    end = now - dt.timedelta(minutes=30)
    assert effective_feed("sip", end, now) == "sip"


def test_exactly_at_embargo_boundary_is_allowed():
    now = dt.datetime(2026, 11, 6, 18, 0, tzinfo=dt.timezone.utc)
    end = now - SIP_EMBARGO
    assert effective_feed("sip", end, now) == "sip"


def test_latest_sip_request_falls_back_to_iex():
    """end=None means 'latest', which Basic can never serve from SIP."""
    assert effective_feed("sip", None) == "iex"


def test_iex_request_is_untouched():
    assert effective_feed("iex", None) == "iex"
    assert effective_feed("iex", dt.datetime(2026, 11, 6)) == "iex"


def test_naive_end_is_treated_as_utc():
    now = dt.datetime(2026, 11, 6, 18, 0, tzinfo=dt.timezone.utc)
    end = dt.datetime(2026, 11, 6, 17, 0)  # naive, 1h old
    assert effective_feed("sip", end, now) == "sip"


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
def test_missing_credentials_raise_with_helpful_message(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    with pytest.raises(AlpacaError, match="not set"):
        Credentials.from_env()


def test_credentials_from_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "PKTEST")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")
    c = Credentials.from_env()
    assert c.key_id == "PKTEST"
    assert c.headers()["APCA-API-KEY-ID"] == "PKTEST"
    assert c.headers()["APCA-API-SECRET-KEY"] == "secret"


def test_blank_credentials_are_rejected(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "   ")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "x")
    with pytest.raises(AlpacaError):
        Credentials.from_env()


# --------------------------------------------------------------------------- #
# Spread statistics: the numbers that replace a guessed cost model
# --------------------------------------------------------------------------- #
def _quotes(spreads_bps: list[float]) -> pd.DataFrame:
    mids = 2.0
    rows = []
    for s in spreads_bps:
        half = s / 10_000.0 * mids / 2.0
        rows.append({"bid": mids - half, "ask": mids + half, "mid": mids, "spread_bps": s})
    return pd.DataFrame(rows)


def test_spread_stats_median_and_p90():
    q = _quotes([100, 100, 100, 200, 1000])
    st = spread_stats(q)
    assert st["n"] == 5
    assert st["median_spread_bps"] == pytest.approx(100.0)
    assert st["p90_spread_bps"] >= 200.0


def test_half_spread_is_half_the_quoted_spread():
    st = spread_stats(_quotes([200, 200, 200]))
    assert st["median_half_spread_bps"] == pytest.approx(100.0)


def test_spread_stats_empty_is_safe():
    assert spread_stats(pd.DataFrame())["n"] == 0
    assert spread_stats(pd.DataFrame({"spread_bps": []}))["n"] == 0


def test_spread_stats_ignores_nan_and_inf():
    q = _quotes([100, 200, 300])
    q.loc[len(q)] = {"bid": 1.0, "ask": 2.0, "mid": 1.5, "spread_bps": np.nan}
    q.loc[len(q)] = {"bid": 1.0, "ask": 2.0, "mid": 1.5, "spread_bps": np.inf}
    st = spread_stats(q)
    assert st["n"] == 3
    assert np.isfinite(st["median_spread_bps"])


def test_suggest_cost_bps_reflects_measured_spread():
    """A 200bps quoted spread implies ~100bps to cross, x1.5 safety = 150."""
    q = _quotes([200] * 50)
    cost = suggest_cost_bps({"AAA": q})
    assert cost == pytest.approx(150.0)


def test_suggest_cost_bps_needs_enough_samples():
    thin = _quotes([200] * 5)   # fewer than the 10-quote minimum
    assert np.isnan(suggest_cost_bps({"AAA": thin}))


def test_suggest_cost_bps_takes_median_across_symbols():
    a = _quotes([100] * 20)     # half spread 50
    b = _quotes([300] * 20)     # half spread 150
    c = _quotes([200] * 20)     # half spread 100
    # median half-spread = 100, x1.5 = 150
    assert suggest_cost_bps({"A": a, "B": b, "C": c}) == pytest.approx(150.0)


def test_suggest_cost_bps_dwarfs_large_cap_assumptions():
    """A realistic microcap spread should be far above the 5bps large-cap default."""
    penny = _quotes([400] * 30)
    cost = suggest_cost_bps({"PNY": penny})
    assert cost > 100.0
