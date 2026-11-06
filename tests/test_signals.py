from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volcast.signals import cross_sectional_weights, momentum_signal, rank_signal, vol_carry
from volcast.sizing import target_positions, turnover


def test_weights_are_dollar_neutral_after_cap():
    """BUG #11: demeaning must survive the cap."""
    score = pd.Series({"A": 1.0, "B": 0.8, "C": -0.4, "D": -0.7, "E": -0.7})
    w = cross_sectional_weights(score, max_weight=0.25)
    assert w.sum() == pytest.approx(0.0, abs=1e-12)
    assert w.abs().max() <= 0.25 + 1e-12


def test_weights_respect_cap_on_skewed_book():
    # One huge positive score would otherwise take the whole book.
    score = pd.Series({"A": 10.0, "B": 0.1, "C": -0.1, "D": -0.2})
    w = cross_sectional_weights(score, max_weight=0.25)
    assert w.abs().max() <= 0.25 + 1e-12
    assert w.sum() == pytest.approx(0.0, abs=1e-12)


def test_two_name_book_is_capped_and_neutral():
    w = cross_sectional_weights(pd.Series({"A": 1.0, "B": -1.0}), max_weight=0.25)
    assert w["A"] == pytest.approx(0.25)
    assert w["B"] == pytest.approx(-0.25)


def test_all_zero_score_gives_zero_weights():
    w = cross_sectional_weights(pd.Series({"A": 0.0, "B": 0.0}), max_weight=0.25)
    assert (w == 0.0).all()


def test_nan_scores_do_not_propagate():
    w = cross_sectional_weights(pd.Series({"A": 1.0, "B": np.nan, "C": -1.0}), max_weight=0.25)
    assert np.isfinite(w.to_numpy()).all()


def test_non_neutral_mode_keeps_net_direction():
    w = cross_sectional_weights(
        pd.Series({"A": 1.0, "B": 1.0}), max_weight=0.25, dollar_neutral=False
    )
    assert w.sum() > 0


def test_vol_carry_sign():
    assert vol_carry(0.10, 0.20) > 0     # short vol is cheap -> long carry
    assert vol_carry(0.20, 0.10) < 0


def test_rank_signal_range():
    r = rank_signal(pd.Series({"A": 1.0, "B": 2.0, "C": 3.0}))
    assert r.max() <= 1.0 + 1e-12
    assert r.min() >= -1.0 - 1e-12


def test_momentum_signal_zero_when_flat():
    m = momentum_signal(pd.Series([0.0] * 10))
    assert (m.abs() < 1e-12).all()


def test_target_positions_survive_nan_sigma():
    """BUG #1 consequence: a NaN sigma must not silently zero the book."""
    weights = pd.Series({"A": 0.5, "B": -0.5})
    sigma = pd.Series({"A": np.nan, "B": 0.01})
    out = target_positions(weights, sigma, target_vol=0.15, horizon=5, max_weight=0.25)
    assert np.isfinite(out.to_numpy()).all()
    assert out["A"] != 0.0


def test_target_positions_respect_max_weight():
    weights = pd.Series({"A": 1.0})
    sigma = pd.Series({"A": 1e-6})   # tiny vol -> huge naive size
    out = target_positions(weights, sigma, target_vol=0.15, horizon=5, max_weight=0.25)
    assert out["A"] <= 0.25


def test_turnover_identity_on_same_book():
    w = pd.Series({"A": 0.2, "B": -0.2})
    assert turnover(w, w) == pytest.approx(0.0)
