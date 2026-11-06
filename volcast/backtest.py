"""Event-driven-ish vectorised backtest with explicit transaction costs."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import signals, sizing
from .forecast import forecast_panel


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    stats: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return dict(self.stats)


def run_backtest(
    panel,
    target_vol: float = 0.15,
    max_weight: float = 0.25,
    cost_bps: float = 5.0,
    horizon: int = 5,
    rebalance: int = 5,
    engine: str = "drift",
    test_days: int = 126,
    vol_window: int = 20,
    dollar_neutral: bool = True,
) -> BacktestResult:
    """Backtest a vol-targeted cross-sectional book over the last ``test_days``."""
    rets = panel.returns
    if len(rets) < 30:
        raise ValueError("not enough history to backtest")

    oos = rets.iloc[-int(test_days):]
    scores = signals.build_signals(panel, horizon=horizon, vol_window=vol_window)
    sig_oos = scores.reindex(oos.index).ffill().fillna(0.0)

    cols = list(oos.columns)
    w = pd.Series(0.0, index=cols, dtype=float)

    weight_rows: dict[pd.Timestamp, pd.Series] = {}
    gross_parts: list[pd.Series] = []
    cost_parts: list[float] = []
    turn_parts: list[float] = []

    step = max(int(rebalance), 1)

    for i, dt in enumerate(oos.index):
        if i % step == 0:
            # Forecast using only data available at dt.
            hist = panel.returns.loc[panel.returns.index <= dt]
            fc = forecast_panel(hist, horizon=horizon, window=vol_window, engine=engine)
            sigma_h = pd.Series(dict(zip(fc["ticker"], fc["sigma_h"], strict=False)))

            raw = signals.cross_sectional_weights(
                sig_oos.loc[dt], max_weight=max_weight, dollar_neutral=dollar_neutral
            )
            new = sizing.target_positions(
                raw,
                sigma_h,
                target_vol=target_vol,
                horizon=horizon,
                max_weight=max_weight,
                dollar_neutral=dollar_neutral,
            )
            t = sizing.turnover(w, new)
            turn_parts.append((dt, t))
            cost_parts.append((dt, t * cost_bps / 10_000.0))
            w = new

        weight_rows[dt] = w.copy()
        day_ret = oos.loc[dt].reindex(cols).fillna(0.0)
        gross_parts.append((dt, float((w * day_ret).sum())))

    gross = pd.Series(dict(gross_parts)).sort_index()
    costs = pd.Series(dict(cost_parts)).reindex(gross.index).fillna(0.0)
    turns = pd.Series(dict(turn_parts)).reindex(gross.index).fillna(0.0)

    net = gross - costs
    equity = (1.0 + net).cumprod()

    from .evaluate import metrics  # local import avoids a cycle

    stats = metrics(net)
    stats.update(
        {
            "gross_sharpe": metrics(gross)["sharpe"],
            "total_cost": float(costs.sum()),
            "avg_turnover": float(turns[turns > 0].mean()) if (turns > 0).any() else 0.0,
            "rebalances": int((turns > 0).sum()),
            "days": int(len(net)),
            "engine": engine,
        }
    )

    return BacktestResult(
        equity=equity,
        returns=net,
        gross_returns=gross,
        weights=pd.DataFrame(weight_rows).T.reindex(columns=cols).fillna(0.0),
        turnover=turns,
        stats=stats,
    )
