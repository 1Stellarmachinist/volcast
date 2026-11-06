from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volcast.subpenny import (
    dilution_risk_flags,
    moving_average_state,
    relative_volume,
    scan,
)


def _frame(series: dict[str, list[float]], start="2025-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(next(iter(series.values()))))
    return pd.DataFrame(series, index=idx)


def _volume(value: float, n: int, start="2025-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"AAA": [value] * n}, index=idx)


# --------------------------------------------------------------------------- #
# Relative volume
# --------------------------------------------------------------------------- #
def test_rvol_reflects_volume_surge():
    idx = pd.bdate_range("2025-01-01", periods=60)
    v = pd.DataFrame({"A": [1000.0] * 59 + [5000.0]}, index=idx)
    rvol = relative_volume(v)
    assert float(rvol["A"]) == pytest.approx(5.0, rel=0.02)


def test_rvol_is_one_when_volume_is_flat():
    idx = pd.bdate_range("2025-01-01", periods=60)
    v = pd.DataFrame({"A": [2000.0] * 60}, index=idx)
    assert float(relative_volume(v)["A"]) == pytest.approx(1.0, rel=0.02)


def test_rvol_handles_zero_average():
    idx = pd.bdate_range("2025-01-01", periods=60)
    v = pd.DataFrame({"A": [0.0] * 59 + [100.0]}, index=idx)
    rvol = relative_volume(v)
    assert not np.isinf(rvol["A"])


# --------------------------------------------------------------------------- #
# Moving averages
# --------------------------------------------------------------------------- #
def test_ma_positions_positive_in_uptrend():
    px = _frame({"A": [0.001 * (1.02 ** i) for i in range(80)]})
    st = moving_average_state(px)
    assert float(st.loc["A", "ma20_pos"]) > 0
    assert float(st.loc["A", "ma50_pos"]) > 0
    assert float(st.loc["A", "ma20_slope"]) > 0


def test_ma_positions_negative_in_downtrend():
    px = _frame({"A": [1.0 * (0.98 ** i) for i in range(80)]})
    st = moving_average_state(px)
    assert float(st.loc["A", "ma20_pos"]) < 0
    assert float(st.loc["A", "ma20_slope"]) < 0


def test_crossover_detected_on_the_crossing_bar():
    # Flat at 1.00 then a jump: price crosses above its own trailing MA.
    vals = [1.00] * 40 + [1.50]
    px = _frame({"A": vals})
    st = moving_average_state(px)
    assert bool(st.loc["A", "ma_cross"]) is True


def test_no_crossover_when_already_above():
    px = _frame({"A": [1.0 * (1.01 ** i) for i in range(60)]})
    st = moving_average_state(px)
    assert bool(st.loc["A", "ma_cross"]) is False


# --------------------------------------------------------------------------- #
# Scanner
# --------------------------------------------------------------------------- #
def test_scan_flags_illiquid_names_ineligible():
    px = _frame({"THIN": [0.001] * 60, "LIQD": [0.001] * 60})
    vol = pd.DataFrame(
        {"THIN": [100.0] * 60, "LIQD": [50_000_000.0] * 60},
        index=px.index,
    )
    df = scan(px, vol, top=None)
    thin = df[df.ticker == "THIN"].iloc[0]
    liqd = df[df.ticker == "LIQD"].iloc[0]
    assert thin["eligible"] is False or thin["eligible"] == np.False_
    assert any("volume" in w for w in thin["warnings"])
    assert liqd["eligible"] in (True, np.True_)


def test_scan_ranks_volume_surge_higher():
    px = _frame({"SURGE": [0.002] * 60, "QUIET": [0.002] * 60})
    vol = pd.DataFrame(
        {
            "SURGE": [1_000_000.0] * 59 + [20_000_000.0],
            "QUIET": [1_000_000.0] * 60,
        },
        index=px.index,
    )
    df = scan(px, vol, top=None)
    surge = df[df.ticker == "SURGE"].iloc[0]
    quiet = df[df.ticker == "QUIET"].iloc[0]
    assert surge["rvol"] > 3.0
    assert surge["score"] > quiet["score"]


def test_scan_requires_min_bars():
    px = _frame({"A": [0.001] * 10})
    with pytest.raises(ValueError, match="25 bars"):
        scan(px)


def test_scan_without_volumes_still_works():
    px = _frame({"A": [0.001 * (1.01 ** i) for i in range(60)]})
    df = scan(px, volumes=None, top=None)
    assert len(df) == 1
    assert df.iloc[0]["score"] >= 0


def test_scan_respects_top_n():
    cols = {f"T{i}": [0.001 * (1.005 ** j) for j in range(60)] for i in range(8)}
    px = _frame(cols)
    vol = pd.DataFrame({c: [1e6] * 60 for c in cols}, index=px.index)
    df = scan(px, vol, top=3)
    assert len(df) == 3


def test_scan_above_ma20_filter_removes_downtrend():
    px = _frame(
        {
            "UP": [0.001 * (1.02 ** i) for i in range(60)],
            "DOWN": [1.0 * (0.98 ** i) for i in range(60)],
        }
    )
    vol = pd.DataFrame({"UP": [1e6] * 60, "DOWN": [1e6] * 60}, index=px.index)
    df = scan(px, vol, require_above_ma20=True, top=None)
    assert "DOWN" not in set(df["ticker"])


def test_scan_output_has_disclaimer_metadata():
    px = _frame({"A": [0.001 * (1.01 ** i) for i in range(60)]})
    df = scan(px, top=None)
    assert "not a prediction" in df.attrs["disclaimer"]
    assert df.attrs["as_of"] == str(px.index[-1].date())


def test_scan_never_returns_inf_scores():
    px = _frame({"A": [0.0] * 59 + [0.001]})
    df = scan(px, volumes=None, top=None)
    if len(df):
        assert np.isfinite(df["score"]).all()


# --------------------------------------------------------------------------- #
# Dilution flags
# --------------------------------------------------------------------------- #
def test_dilution_flags_detect_collapse():
    px = _frame({"BAD": [0.01 * (0.95 ** i) for i in range(80)]})
    flags = dilution_risk_flags(px)
    assert any("dilution" in w or "collapse" in w for w in flags.loc["BAD", "warnings"])


def test_scan_handles_ffilled_prices_with_sparse_volumes():
    """Regression: prices are ffilled but volumes are not.

    Multiplying the two raw produced NaN for most bars and a dollar-volume average
    ~20x too low, which silently marked every liquid name as illiquid.
    """
    idx = pd.bdate_range("2025-01-01", periods=60)
    px = pd.DataFrame({"A": [0.002] * 60}, index=idx)   # ffilled: no gaps
    vol = pd.DataFrame({"A": [1_000_000.0] * 60}, index=idx)
    vol.iloc[::3] = np.nan                              # sparse, like a real fetch

    df = scan(px, vol, top=None)
    row = df[df.ticker == "A"].iloc[0]
    assert row["dollar_volume"] is not None
    # 0.002 * 1,000,000 = $2,000 on days with volume; the 20-day mean of a series
    # that is NaN on 1/3 of bars (-> 0) is ~$1,330. The bug produced ~$70.
    assert row["dollar_volume"] > 1_000, f"got {row['dollar_volume']}"


def test_scan_volume_alignment_uses_zero_not_ffill():
    """A day with no volume must contribute 0, not repeat yesterday's volume."""
    idx = pd.bdate_range("2025-01-01", periods=60)
    px = pd.DataFrame({"A": [1.0] * 60}, index=idx)
    vol = pd.DataFrame({"A": [0.0] * 59 + [1_000_000.0]}, index=idx)

    dv = (px * vol.reindex(index=px.index, columns=px.columns).fillna(0.0)).rolling(
        20, min_periods=5
    ).mean().iloc[-1]
    # Only the final bar has volume, so the 20-day mean is 1e6/20.
    assert float(dv["A"]) == pytest.approx(50_000.0)


def test_dilution_flags_detect_sub_thousandth_price():
    px = _frame({"TINY": [0.0002] * 80})
    flags = dilution_risk_flags(px)
    assert any("0.001" in w for w in flags.loc["TINY", "warnings"])


def test_dilution_flags_quiet_on_healthy_name():
    px = _frame({"OK": [0.01 * (1.005 ** i) for i in range(80)]})
    flags = dilution_risk_flags(px)
    warns = flags.loc["OK", "warnings"]
    assert not any("collapse" in w for w in warns)
