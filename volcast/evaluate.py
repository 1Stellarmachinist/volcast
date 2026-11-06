"""Evaluation metrics: Sharpe with standard error, drawdown, pinball loss."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .utils import clean_float

PERIODS = 252


def sharpe_se(sharpe: float, n: int) -> float:
    """Standard error of an annualised Sharpe ratio (Lo 2002, iid approximation)."""
    if n < 3:
        return float("nan")
    sr_daily = float(sharpe) / math.sqrt(PERIODS)
    return float(math.sqrt((1.0 + 0.5 * sr_daily**2) / n) * math.sqrt(PERIODS))


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def ann_return(returns: pd.Series) -> float:
    if len(returns) == 0:
        return float("nan")
    total = float((1.0 + returns.fillna(0.0)).prod())
    years = len(returns) / PERIODS
    if years <= 0 or total <= 0:
        return float("nan")
    return float(total ** (1.0 / years) - 1.0)


def metrics(returns: pd.Series, periods: int = PERIODS) -> dict:
    """Compute the standard metric bundle.

    BUG #9 FIX: on zero-variance returns the Sharpe ratio is genuinely undefined
    (0/0), so we return NaN instead of manufacturing a huge or zero number.
    Callers must compare |sharpe| against ``sharpe_se`` rather than trusting a
    point estimate.
    """
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(r))
    if n < 2:
        return {
            "sharpe": float("nan"),
            "sharpe_se": float("nan"),
            "ann_return": float("nan"),
            "ann_vol": float("nan"),
            "max_drawdown": float("nan"),
            "n": n,
            "t_stat": float("nan"),
        }

    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    ann_mu = mu * periods
    ann_sd = sd * math.sqrt(periods)

    if not math.isfinite(sd) or sd <= 1e-15:
        sharpe = float("nan")       # zero-variance: undefined, not infinite
    else:
        sharpe = ann_mu / ann_sd

    se = sharpe_se(sharpe, n) if math.isfinite(sharpe) else float("nan")
    equity = (1.0 + r).cumprod()

    return {
        "sharpe": clean_float(sharpe) if clean_float(sharpe) is not None else float("nan"),
        "sharpe_se": clean_float(se) if clean_float(se) is not None else float("nan"),
        "t_stat": (sharpe / se) if (math.isfinite(sharpe) and math.isfinite(se) and se > 0) else float("nan"),
        "ann_return": clean_float(ann_return(r)) if clean_float(ann_return(r)) is not None else float("nan"),
        "ann_vol": clean_float(ann_sd) if clean_float(ann_sd) is not None else float("nan"),
        "max_drawdown": clean_float(max_drawdown(equity)) if clean_float(max_drawdown(equity)) is not None else float("nan"),
        "n": n,
    }


def pinball(actual: float, forecast: float, quantile: float = 0.5) -> float:
    """Quantile (pinball) loss, argument order ``(actual, forecast, quantile)``.

    L = q * (y - f)          if y >= f
      = (1 - q) * (f - y)    if y <  f

    BUG #10 FIX: the earlier implementation multiplied the shortfall branch by a
    duplicated factor, so the loss was *not* minimised where the forecast equals
    the claimed quantile. Two invariants now hold and are asserted in the tests:

      pinball(0.01, 0.01, 0.9) == 0            # perfect forecast -> zero loss
      pinball(0.03, 0.01, 0.9) == 0.9 * 0.02   # under-forecast priced at q
    """
    q = float(quantile)
    e = float(actual) - float(forecast)
    if e >= 0:
        return q * e
    return (1.0 - q) * (-e)


def mean_pinball(actual, forecast, quantile: float = 0.5) -> float:
    """Average pinball loss over aligned samples."""
    y_arr = np.asarray(actual, dtype=float)
    f_arr = np.asarray(forecast, dtype=float)
    if y_arr.size == 0:
        return float("nan")
    q = float(quantile)
    loss = np.where(y_arr >= f_arr, q * (y_arr - f_arr), (1.0 - q) * (f_arr - y_arr))
    return float(np.mean(loss))


def coverage(actual, forecast, quantile: float) -> float:
    """Empirical hit rate of a quantile forecast; should approach ``quantile``."""
    y_arr = np.asarray(actual, dtype=float)
    f_arr = np.asarray(forecast, dtype=float)
    if y_arr.size == 0:
        return float("nan")
    return float(np.mean(y_arr <= f_arr))


def report(returns: pd.Series) -> str:
    m = metrics(returns)
    s, se = m.get("sharpe"), m.get("sharpe_se")
    txt = (
        f"n={m['n']}  Sharpe={s:.3f}  SE={se:.3f}  "
        f"ann_ret={m['ann_return']:.2%}  ann_vol={m['ann_vol']:.2%}  maxDD={m['max_drawdown']:.2%}"
    )
    if math.isfinite(s) and math.isfinite(se) and abs(s) < 2 * se:
        txt += "\n  NOTE: |Sharpe| < 2*SE -> statistically indistinguishable from zero."
    return txt
