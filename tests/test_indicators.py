from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volcast import indicators


@pytest.fixture
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2023-01-02", periods=300)
    data = {}
    for i, t in enumerate(["AAA", "BBB", "CCC"]):
        data[t] = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01 + i * 0.004, len(idx))))
    return pd.DataFrame(data, index=idx)


def test_log_returns_length(prices):
    r = indicators.log_returns(prices)
    assert len(r) == len(prices) - 1
    assert list(r.columns) == ["AAA", "BBB", "CCC"]


def test_realised_vol_matches_manual(prices):
    r = indicators.log_returns(prices)
    rv = indicators.realised_vol(r, 20)
    manual = r["AAA"].iloc[-20:].std(ddof=1)
    assert rv["AAA"].iloc[-1] == pytest.approx(manual, rel=1e-10)


def test_realised_vol_is_finite_after_warmup(prices):
    r = indicators.log_returns(prices)
    rv = indicators.realised_vol(r, 20)
    tail = rv.iloc[40:]
    assert np.isfinite(tail.to_numpy()).all()
    assert (tail > 0).all().all()


def test_rsi_bounds(prices):
    r = indicators.rsi(prices, 14).dropna()
    assert (r.to_numpy() >= 0).all()
    assert (r.to_numpy() <= 100).all()


def test_ewma_vol_positive(prices):
    r = indicators.log_returns(prices)
    v = indicators.ewma_vol(r).dropna()
    assert (v > 0).all().all()


def test_vol_ratio_has_no_inf(prices):
    r = indicators.log_returns(prices)
    ratio = indicators.vol_ratio(r, 10, 60)
    assert not np.isinf(ratio.to_numpy()).any()


def test_feature_table_shape(prices):
    ft = indicators.feature_table(prices, vol_window=20)
    assert list(ft.index) == ["AAA", "BBB", "CCC"]
    assert {"last", "vol_20", "mom_63", "z_60"} <= set(ft.columns)
