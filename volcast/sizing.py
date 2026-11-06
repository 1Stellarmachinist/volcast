"""Position sizing and turnover-aware adjustment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import safe_sigma


def target_positions(
    weights: pd.Series,
    sigma_h: pd.Series,
    target_vol: float = 0.15,
    horizon: int = 5,
    max_weight: float = 0.25,
    periods: int = 252,
    dollar_neutral: bool = False,
) -> pd.Series:
    """Scale raw signal weights by vol-targeting.

    BUG #1 consequence: if ``sigma_h`` were NaN, every position would be 0. The
    guard below turns a non-finite forecast into the *unscaled* weight instead of
    a zero position, so a single bad ticker cannot silently flatten the book.

    ``dollar_neutral`` matters because the vol scaling is a *per-name* multiplier:
    a book that was neutral before scaling comes out net-long or net-short after,
    with the tilt determined purely by which names happen to be quieter. When
    requested, the scaled book is re-centred onto zero net exposure.
    """
    idx = weights.index
    sh = pd.to_numeric(sigma_h.reindex(idx), errors="coerce")
    ann = sh * np.sqrt(periods / max(horizon, 1))
    scale = float(target_vol) / ann.where(ann > 0)

    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out = (weights.astype(float) * scale).clip(-max_weight, max_weight)
    out = out.fillna(0.0)

    if dollar_neutral and len(out):
        out = out - out.mean()
        out = out.clip(-max_weight, max_weight)
        out = out - out.mean()

    return out


def turnover(old: pd.Series, new: pd.Series) -> float:
    idx = old.index.union(new.index)
    return float((new.reindex(idx).fillna(0.0) - old.reindex(idx).fillna(0.0)).abs().sum())


def apply_turnover_penalty(
    old: pd.Series,
    new: pd.Series,
    max_turnover: float = 1.0,
) -> pd.Series:
    """Blend toward the old book when a rebalance would churn too hard."""
    t = turnover(old, new)
    if t <= max_turnover or t <= 1e-12:
        return new
    blend = max_turnover / t
    idx = old.index.union(new.index)
    blended = old.reindex(idx).fillna(0.0) + blend * (
        new.reindex(idx).fillna(0.0) - old.reindex(idx).fillna(0.0)
    )
    return blended.reindex(new.index).fillna(0.0)


def kelly_fraction(edge: float, variance: float, cap: float = 0.5) -> float:
    v = safe_sigma(variance)
    return float(np.clip(float(edge) / (v * v), -cap, cap))
