from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volcast.growth import (
    composite_score,
    liquidity_screen,
    price_signals,
    rank_smallcaps,
)
from volcast.ranking import backtest_ranking


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2023-01-02", periods=320)
    data = {}
    # One clear winner (steady uptrend), one clear loser, rest noise.
    for i, t in enumerate(["WIN", "MID", "LOSE", "NOISE"]):
        drift = {"WIN": 0.0015, "MID": 0.0002, "LOSE": -0.0018, "NOISE": 0.0}[t]
        vol = 0.02 + i * 0.004
        data[t] = 50 * np.exp(np.cumsum(rng.normal(drift, vol, len(idx))))
    return pd.DataFrame(data, index=idx)


def test_price_signals_are_point_in_time(prices):
    """A signal at date t must not change when future rows are appended."""
    full = price_signals(prices)
    cut = price_signals(prices.iloc[:-40])
    dt = prices.index[-60]

    row_full = full.xs(dt, level="date")
    row_cut = cut.xs(dt, level="date")
    # Momentum/vol at that date must be identical either way.
    for col in ("momentum_3m", "vol_regime"):
        a = row_full[col].reindex(prices.columns)
        b = row_cut[col].reindex(prices.columns)
        pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-12)


def test_momentum_ranks_winner_above_loser(prices):
    cross = price_signals(prices).xs(prices.index[-1], level="date")
    assert cross["momentum_3m"]["WIN"] > cross["momentum_3m"]["LOSE"]


def test_composite_score_prefers_momentum(prices):
    cross = price_signals(prices).xs(prices.index[-1], level="date")
    score = composite_score(cross)
    assert score["WIN"] > score["LOSE"]


def test_composite_score_is_bounded_and_finite(prices):
    cross = price_signals(prices).xs(prices.index[-1], level="date")
    score = composite_score(cross)
    assert np.isfinite(score.to_numpy()).all()
    # Rank-based, so it cannot explode regardless of raw magnitudes.
    assert score.abs().max() <= 10.0


def test_composite_score_with_nan_inputs_is_safe():
    sig = pd.DataFrame(
        {
            "momentum_3m": [0.1, np.nan, -0.2],
            "vol_regime": [1.0, np.nan, np.nan],
            "dist_from_low": [np.nan, 0.5, 0.1],
        },
        index=["A", "B", "C"],
    )
    score = composite_score(sig)
    assert np.isfinite(score.to_numpy()).all()


def test_liquidity_screen_flags_cheap_illiquid_names():
    idx = pd.bdate_range("2024-01-01", periods=30)
    px = pd.DataFrame({"OK": 5.0, "CHEAP": 0.05, "THIN": 2.0}, index=idx)
    vol = pd.DataFrame({"OK": 1e6, "CHEAP": 1e6, "THIN": 10}, index=idx)
    ok = liquidity_screen(px, vol)
    assert bool(ok["OK"]) is True
    assert bool(ok["CHEAP"]) is False   # below MIN_PRICE
    assert bool(ok["THIN"]) is False    # below MIN_DOLLAR_VOLUME


def test_rank_smallcaps_returns_sorted_with_metadata(prices):
    df = rank_smallcaps(prices, top=4)
    assert len(df) == 4
    assert df["score"].is_monotonic_decreasing
    assert df.attrs["as_of"] == str(prices.index[-1].date())
    assert "eligible" in df.columns


def test_rank_smallcaps_rejects_short_history():
    idx = pd.bdate_range("2024-01-01", periods=50)
    px = pd.DataFrame({"A": np.linspace(10, 12, 50)}, index=idx)
    with pytest.raises(ValueError, match="need >="):
        rank_smallcaps(px)


def test_backtest_ranking_separates_in_and_out_of_sample(prices):
    res = backtest_ranking(prices, horizon=5, rebalance=10, top_n=2)
    assert "in_sample" in res.summary
    assert "out_sample" in res.summary
    assert res.summary["in_sample"]["n"] > 0
    assert res.summary["out_sample"]["n"] > 0


def test_backtest_ranking_reports_se_and_tstat(prices):
    res = backtest_ranking(prices, horizon=5, rebalance=10, top_n=2)
    allsum = res.summary["all"]
    assert allsum["n"] > 1
    assert np.isfinite(allsum["se"])
    assert allsum["hit_rate_beat_universe"] is not None


def test_backtest_ranking_has_no_fundamentals_leak(prices):
    """The evaluation path must never consult the current-snapshot fundamentals."""
    import inspect

    from volcast import ranking

    src = inspect.getsource(ranking)
    assert "fundamental_snapshot" not in src, \
        "ranking.py imports fundamentals, which would leak future data into a backtest"


def test_backtest_ranking_on_trend_is_monotonic():
    """With a clear uptrend in one name, the top bucket should beat the bottom.

    Needs >= 6 tickers: the bucket view splits the cross-section into thirds, and
    with fewer names there is nothing to split.
    """
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2023-01-02", periods=320)
    data = {}
    for i in range(8):
        drift = 0.0018 if i == 0 else -0.0008 + i * 0.0002
        data[f"T{i}"] = 50 * np.exp(np.cumsum(rng.normal(drift, 0.02, len(idx))))
    wide = pd.DataFrame(data, index=idx)

    res = backtest_ranking(wide, horizon=5, rebalance=10, top_n=2)
    b = res.buckets
    assert len(b) == 3
    top = float(b.loc[b.bucket == "top_third", "mean_fwd_return"].iloc[0])
    bottom = float(b.loc[b.bucket == "bottom_third", "mean_fwd_return"].iloc[0])
    assert top > bottom


def test_backtest_ranking_raises_on_insufficient_history():
    idx = pd.bdate_range("2024-01-01", periods=60)
    px = pd.DataFrame({"A": np.linspace(10, 12, 60)}, index=idx)
    with pytest.raises(ValueError):
        backtest_ranking(px)
