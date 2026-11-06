from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volcast.sizing import (
    apply_turnover_penalty,
    kelly_fraction,
    target_positions,
    turnover,
)
from volcast.utils import safe_sigma


def test_safe_sigma_never_returns_nan_or_zero():
    for bad in (float("nan"), float("inf"), -1.0, 0.0, None, "x"):
        v = safe_sigma(bad)
        assert np.isfinite(v)
        assert v > 0


def test_safe_sigma_preserves_valid_value():
    assert safe_sigma(0.02) == pytest.approx(0.02)


def test_vol_target_scales_to_target():
    w = pd.Series({"A": 1.0})
    # sigma_h for h=5: ann = sigma*sqrt(5)*sqrt(252/5) = sigma*sqrt(252)
    sigma_h = 0.10 / np.sqrt(252) * np.sqrt(5)
    out = target_positions(w, pd.Series({"A": sigma_h}), target_vol=0.10, horizon=5,
                           max_weight=10.0)
    assert out["A"] == pytest.approx(1.0, rel=1e-6)


def test_turnover_counts_both_legs():
    old = pd.Series({"A": 0.5, "B": -0.5})
    new = pd.Series({"A": 0.0, "B": 0.0})
    assert turnover(old, new) == pytest.approx(1.0)


def test_turnover_penalty_blends_toward_old():
    old = pd.Series({"A": 0.5, "B": -0.5})
    new = pd.Series({"A": -0.5, "B": 0.5})
    blended = apply_turnover_penalty(old, new, max_turnover=0.5)
    assert blended.abs().sum() < new.abs().sum()


def test_kelly_is_capped():
    assert kelly_fraction(edge=100.0, variance=0.01, cap=0.5) == pytest.approx(0.5)
    assert kelly_fraction(edge=-100.0, variance=0.01, cap=0.5) == pytest.approx(-0.5)


def test_vol_scaling_preserves_dollar_neutrality():
    """Per-name vol scaling must not tilt a neutral book net-long or net-short.

    This was a live bug: signal weights summed to zero, but dividing each by its
    own volatility left net=+0.13 because the quiet names grew more than the loud
    ones shrank.
    """
    weights = pd.Series({"A": 0.18, "B": 0.13, "C": 0.10, "D": -0.16, "E": -0.21})
    # Deliberately very different volatilities.
    sigma_h = pd.Series({"A": 0.028, "B": 0.045, "C": 0.050, "D": 0.064, "E": 0.079})
    sized = target_positions(weights, sigma_h, dollar_neutral=True)
    assert sized.sum() == pytest.approx(0.0, abs=1e-10)
    assert sized.abs().max() <= 0.25 + 1e-12


def test_vol_scaling_without_neutral_flag_can_tilt():
    weights = pd.Series({"A": 0.18, "B": -0.18})
    sigma_h = pd.Series({"A": 0.02, "B": 0.08})
    sized = target_positions(weights, sigma_h, dollar_neutral=False, max_weight=10.0)
    assert abs(sized.sum()) > 1e-6  # documents why the flag exists
