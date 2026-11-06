"""Signal construction: vol-targeting, vol carry, and cross-sectional weights."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import safe_sigma, zscore


def vol_target_weight(
    sigma_h: float,
    target_vol: float,
    horizon: int,
    max_weight: float = 0.25,
    periods: int = 252,
) -> float:
    """Size a position so its expected annualised vol equals ``target_vol``.

    Weight = target_vol / annualised forecast vol, clipped to +/- max_weight.
    """
    ann = safe_sigma(sigma_h) * np.sqrt(periods / max(horizon, 1))
    w = float(target_vol) / safe_sigma(ann)
    return float(np.clip(w, -max_weight, max_weight))


def vol_carry(short_vol: float, long_vol: float) -> float:
    """Long vol-futures-carry proxy: short vol cheap / long vol rich, sign kept."""
    s = safe_sigma(short_vol)
    lo = safe_sigma(long_vol)
    return float(np.clip((lo - s) / lo, -1.0, 1.0))


def momentum_signal(mom: pd.Series, window: int = 63) -> pd.Series:
    return zscore(mom, window=None).fillna(0.0)


def rank_signal(features: pd.Series, invert: bool = False) -> pd.Series:
    """Cross-sectional percentile rank in [-1, 1]."""
    r = features.rank(pct=True) - 0.5
    r = r * 2.0
    if invert:
        r = -r
    return r.fillna(0.0)


def build_signals(panel, horizon: int = 5, vol_window: int = 20) -> pd.DataFrame:
    """Combine vol + momentum features into a cross-sectional score per date."""
    vol = panel.features["vol"]
    mom = panel.features["momentum"]

    # Low vol is attractive for a vol-targeted book; momentum adds direction.
    low_vol_score = -vol.rank(axis=1, pct=True) + 0.5
    mom_score = mom.rank(axis=1, pct=True) - 0.5
    score = (low_vol_score * 0.5 + mom_score * 0.5) * 2.0
    return score.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _cap(w: pd.Series, max_weight: float) -> pd.Series:
    """Pure clamp to the [-max_weight, +max_weight] box.

    Deliberately a pure projection (no renormalisation) so it composes with the
    demeaning projection in ``cross_sectional_weights``.
    """
    return w.clip(lower=-max_weight, upper=max_weight)


def cross_sectional_weights(
    score: pd.Series,
    max_weight: float = 0.25,
    dollar_neutral: bool = True,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> pd.Series:
    """Turn a cross-sectional score into portfolio weights.

    BUG #11 FIX: demeaning *before* ``_cap`` does not survive clipping - clipping
    a long-heavy book leaves it net-long, and centring *after* the cap can push a
    single name back over the cap. The two constraints (net == 0, |w_i| <= cap)
    are therefore satisfied jointly by alternating projections, which converge
    because both are convex sets with a non-empty intersection.
    """
    s = score.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if float(s.abs().sum()) <= 1e-12:
        return pd.Series(0.0, index=s.index)

    w = s / float(s.abs().sum())
    w = _cap(w, max_weight)

    if dollar_neutral:
        # Alternate: project onto {net == 0}, then back onto the cap box.
        for _ in range(max_iter):
            centred = w - w.mean()
            capped = _cap(centred, max_weight)
            if float((capped - w).abs().max()) < tol:
                w = capped
                break
            w = capped

    # Numerical hygiene: kill sub-epsilon dust so net is exactly representable.
    w = w.where(w.abs() > 1e-15, 0.0)
    if dollar_neutral and len(w):
        w = w - w.mean()

    return w


def portfolio_score_to_weights(
    scores: pd.DataFrame,
    max_weight: float = 0.25,
    dollar_neutral: bool = True,
) -> pd.DataFrame:
    rows = {}
    for dt, row in scores.iterrows():
        rows[dt] = cross_sectional_weights(
            row, max_weight=max_weight, dollar_neutral=dollar_neutral
        )
    return pd.DataFrame(rows).T.reindex(columns=scores.columns).fillna(0.0)
