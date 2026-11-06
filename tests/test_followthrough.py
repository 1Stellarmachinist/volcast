from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volcast.followthrough import (
    detect_breakouts,
    forward_returns,
    study,
)


def _make(px_map: dict, vol_map: dict, start="2025-01-01"):
    n = len(next(iter(px_map.values())))
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(px_map, index=idx), pd.DataFrame(vol_map, index=idx)


def _realistic(n: int = 400, seed: int = 7, price: float = 0.005,
               drift: float = 0.002, vol_daily: float = 0.06,
               base_vol: float = 40_000_000.0, spike_p: float = 0.08):
    """A series that behaves like a real sub-penny name.

    Mild positive drift and ~6% daily vol keep the price (and therefore dollar
    volume) viable, and occasional 5-15x volume spikes create genuine breakout
    events. A pure 15%-vol random walk decays to $0.00001 and every name gets
    filtered as untradeable, which tests nothing.
    """
    rng = np.random.default_rng(seed)
    prices = price * np.exp(np.cumsum(rng.normal(drift, vol_daily, n)))
    vols = base_vol * np.exp(rng.normal(0, 0.4, n))
    spikes = rng.random(n) < spike_p
    vols[spikes] *= rng.uniform(5, 15, spikes.sum())
    idx = pd.bdate_range("2025-01-01", periods=n)
    return (pd.DataFrame({"T": prices}, index=idx),
            pd.DataFrame({"T": vols}, index=idx))
def _flat_with_spike(price=0.001, n=120, spike_at=100, spike_mult=10.0):
    """An uptrending series with a volume spike on one bar.

    A perfectly flat series can never satisfy `price > MA20`, so a flat fixture
    would silently test nothing.
    """
    px = {"T": [price * (1.004 ** i) for i in range(n)]}
    # 100M shares at ~$.001 is ~$100k dollar volume, i.e. above the $10k floor.
    # A 1M-share base would be filtered out as untradeable, which is correct but
    # makes the fixture test nothing.
    vol = {"T": [100_000_000.0] * n}
    vol["T"][spike_at] = 100_000_000.0 * spike_mult
    return _make(px, vol)


# --------------------------------------------------------------------------- #
# No look-ahead
# --------------------------------------------------------------------------- #
def test_signal_on_bar_t_does_not_use_bar_t_plus_1():
    """Truncating the future must not change the event on the truncation date."""
    px, vol = _flat_with_spike(n=120, spike_at=100)
    full = detect_breakouts(px, vol, rvol_min=3.0, require_ma_slope_up=False)
    cut = detect_breakouts(px.iloc[:101], vol.iloc[:101], rvol_min=3.0,
                           require_ma_slope_up=False)

    f_dates = set(full[full.ticker == "T"]["date"]) if len(full) else set()
    c_dates = set(cut[cut.ticker == "T"]["date"]) if len(cut) else set()
    # Any event found using only the first 101 bars must also exist in the full run.
    assert c_dates.issubset(f_dates)


def test_rvol_event_fires_on_the_spike_bar_not_before():
    px, vol = _flat_with_spike(n=120, spike_at=100)
    events = detect_breakouts(px, vol, rvol_min=3.0, require_ma_slope_up=False)
    assert len(events) >= 1
    first = events.iloc[0]
    assert first["date"] == px.index[100]


def test_forward_return_measured_from_the_next_bar():
    """A 10x move on the day AFTER the event must be captured in fwd_1d."""
    n = 120
    base = [0.005 * (1.004 ** i) for i in range(n)]
    px = {"T": list(base)}
    vol = {"T": [100_000_000.0] * n}
    px["T"][101] = base[101] * 10      # 10x the day after the spike
    vol["T"][100] = 100_000_000.0 * 20
    prices, volumes = _make(px, vol)

    events = detect_breakouts(prices, volumes, rvol_min=3.0, require_ma_slope_up=False)
    ev = events[events["date"] == prices.index[100]]
    assert len(ev) == 1, f"expected the spike bar to qualify, got {len(events)} events"

    with_ret = forward_returns(prices, ev, horizons=(1,))
    got = float(with_ret.iloc[0]["fwd_1d"])
    assert got == pytest.approx(10.0 - 1.0, rel=0.01)   # +900%


def test_forward_return_excludes_the_breakout_bar_itself():
    """If the price already jumped ON the event bar, fwd_1d must not count it."""
    n = 120
    base = [0.005 * (1.004 ** i) for i in range(n)]
    px = {"T": list(base)}
    vol = {"T": [100_000_000.0] * n}
    px["T"][100] = base[100] * 5       # spike bar itself is up 5x
    vol["T"][100] = 100_000_000.0 * 20
    prices, volumes = _make(px, vol)

    events = detect_breakouts(prices, volumes, rvol_min=3.0, require_ma_slope_up=False)
    ev = events[events["date"] == prices.index[100]]
    assert len(ev) == 1
    with_ret = forward_returns(prices, ev, horizons=(1,))
    # The next bar is the untouched trend value, so this is a big drop, NOT +400%.
    expected = base[101] / (base[100] * 5) - 1.0
    assert float(with_ret.iloc[0]["fwd_1d"]) == pytest.approx(expected)
    assert float(with_ret.iloc[0]["fwd_1d"]) < -0.5


def test_event_at_end_of_history_gets_nan_not_crash():
    px, vol = _flat_with_spike(n=60, spike_at=59)
    events = detect_breakouts(px, vol, rvol_min=3.0, require_ma_slope_up=False)
    if len(events):
        wr = forward_returns(px, events, horizons=(1, 3, 5))
        assert np.isnan(wr.iloc[0]["fwd_1d"])


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def test_low_rvol_produces_no_events():
    px, vol = _flat_with_spike(spike_mult=1.2)
    events = detect_breakouts(px, vol, rvol_min=3.0, require_ma_slope_up=False)
    assert len(events) == 0


def test_thin_dollar_volume_is_excluded():
    px = {"T": [0.0001] * 120}
    vol = {"T": [1_000.0] * 119 + [100_000.0]}
    prices, volumes = _make(px, vol)
    events = detect_breakouts(prices, volumes, rvol_min=3.0,
                              require_ma_slope_up=False, min_dollar_volume=10_000)
    # 0.0001 * 100k = $10, only $0.50/day average: far below the floor.
    assert len(events) == 0


def test_requires_volumes():
    px, _ = _flat_with_spike()
    with pytest.raises(ValueError, match="volumes"):
        detect_breakouts(px, pd.DataFrame())


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_study_reports_baseline_for_comparison():
    prices, volumes = _realistic(seed=7)
    res = study(prices, volumes, rvol_min=2.0, min_events=5)
    assert "error" not in res.stats, res.stats
    assert "baseline_1d" in res.stats
    assert res.stats["baseline_1d"]["n"] > 100


def test_study_reports_median_and_tails():
    prices, volumes = _realistic(seed=9)
    res = study(prices, volumes, rvol_min=2.0, min_events=5)
    assert "error" not in res.stats, res.stats
    d = res.stats["breakout_1d"]
    for key in ("median", "mean", "win_rate", "p10", "p90", "best", "worst",
                "double_rate", "halve_rate"):
        assert key in d, f"missing {key}"


def test_study_flags_insufficient_events():
    prices, volumes = _realistic(seed=3, spike_p=0.0)   # no spikes at all
    res = study(prices, volumes, rvol_min=3.0, min_events=20)
    assert "error" in res.stats


def test_median_edge_is_computed_when_available():
    prices, volumes = _realistic(seed=13)
    res = study(prices, volumes, rvol_min=2.0, min_events=5)
    assert "error" not in res.stats, res.stats
    assert "median_edge_1d" in res.stats
    assert np.isfinite(res.stats["median_edge_1d"])


def test_max_up_is_at_least_the_close_return():
    """Best moment inside the window cannot be worse than the endpoint."""
    prices, volumes = _realistic(n=300, seed=21, vol_daily=0.05, spike_p=0.15)
    res = study(prices, volumes, rvol_min=2.0, min_events=5)
    assert "error" not in res.stats, res.stats
    ev = res.events.dropna(subset=["fwd_5d", "max_5d"])
    assert len(ev) > 0, "fixture produced no evaluable events"
    assert (ev["max_5d"] >= ev["fwd_5d"] - 1e-12).all()
