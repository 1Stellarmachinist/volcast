from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from volcast.backtest import run_backtest
from volcast.data import synthetic_prices
from volcast.evaluate import coverage, mean_pinball, metrics, pinball, sharpe_se
from volcast.panel import build_panel


@pytest.fixture(scope="module")
def panel():
    px = synthetic_prices(["AAA", "BBB", "CCC", "DDD"], periods=320, seed=11)
    return build_panel(px, vol_window=20)


# --------------------------------------------------------------------------- #
# Bug #9: NaN Sharpe on zero variance
# --------------------------------------------------------------------------- #
def test_metrics_zero_variance_sharpe_is_nan():
    """BUG #9: 0/0 is undefined. It must be NaN, not a fake large number."""
    m = metrics(pd.Series([0.0] * 100))
    assert math.isnan(m["sharpe"]), f"expected NaN sharpe, got {m['sharpe']}"
    assert math.isnan(m["sharpe_se"])
    assert m["n"] == 100


def test_metrics_constant_positive_returns_is_nan():
    m = metrics(pd.Series([0.001] * 50))
    assert math.isnan(m["sharpe"])


def test_metrics_normal_returns_are_finite():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0004, 0.01, 500))
    m = metrics(r)
    assert np.isfinite(m["sharpe"])
    assert np.isfinite(m["sharpe_se"])
    assert m["sharpe_se"] > 0


def test_sharpe_se_shrinks_with_sample_size():
    assert sharpe_se(1.0, 1000) < sharpe_se(1.0, 100)


# --------------------------------------------------------------------------- #
# Bug #10: pinball loss minimised at the claimed quantile
# --------------------------------------------------------------------------- #
def test_pinball_zero_for_perfect_forecast():
    assert pinball(0.01, 0.01, 0.9) == 0.0


def test_pinball_underforecast_priced_at_quantile():
    assert pinball(0.03, 0.01, 0.9) == pytest.approx(0.9 * 0.02)


def test_pinball_overforecast_priced_at_complement():
    assert pinball(0.01, 0.03, 0.9) == pytest.approx(0.1 * 0.02)


def test_pinball_is_minimised_at_the_claimed_quantile():
    """Empirical check: the argmin of mean loss equals the sample quantile."""
    rng = np.random.default_rng(3)
    y = rng.normal(0.0, 0.02, 4000)
    grid = np.linspace(-0.05, 0.05, 201)
    for q in (0.1, 0.5, 0.9):
        losses = [mean_pinball(y, np.full_like(y, f), q) for f in grid]
        best = grid[int(np.argmin(losses))]
        assert best == pytest.approx(float(np.quantile(y, q)), abs=2e-3)


def test_coverage_close_to_claimed_quantile():
    rng = np.random.default_rng(4)
    y = rng.normal(0.0, 0.02, 5000)
    q = 0.9
    f = np.full_like(y, float(np.quantile(y, q)))
    assert coverage(y, f, q) == pytest.approx(q, abs=0.03)


# --------------------------------------------------------------------------- #
# Backtest integration
# --------------------------------------------------------------------------- #
def test_backtest_runs_and_reports(panel):
    res = run_backtest(panel, target_vol=0.15, cost_bps=5.0, test_days=60, rebalance=5)
    assert res.stats["days"] > 0
    assert len(res.equity) == res.stats["days"]
    assert np.isfinite(res.returns.to_numpy()).all()
    assert res.stats["total_cost"] >= 0


def test_backtest_equity_matches_returns(panel):
    res = run_backtest(panel, test_days=40, rebalance=5)
    manual = (1.0 + res.returns).cumprod()
    assert np.allclose(res.equity.to_numpy(), manual.to_numpy())


def test_zero_cost_backtest_has_zero_cost(panel):
    res = run_backtest(panel, cost_bps=0.0, test_days=40, rebalance=5)
    assert res.stats["total_cost"] == pytest.approx(0.0)


def test_backtest_costs_reduce_returns(panel):
    cheap = run_backtest(panel, cost_bps=0.0, test_days=60, rebalance=5)
    dear = run_backtest(panel, cost_bps=50.0, test_days=60, rebalance=5)
    assert dear.returns.sum() <= cheap.returns.sum()
