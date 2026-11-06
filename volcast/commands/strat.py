from __future__ import annotations

from ..backtest import run_backtest
from ..evaluate import report
from ..plot import equity_chart
from ._args import load_panel, resolve_config


def run(args) -> int:
    cfg = resolve_config(args)
    prices, panel = load_panel(args, cfg)

    res = run_backtest(
        panel,
        target_vol=cfg.target_vol,
        max_weight=cfg.max_weight,
        cost_bps=cfg.cost_bps,
        horizon=cfg.horizon,
        rebalance=cfg.rebalance,
        engine=cfg.engine,
        test_days=cfg.test_days,
        vol_window=cfg.vol_window,
        dollar_neutral=not getattr(args, "no_neutral", False),
    )

    s = res.stats
    print(
        f"engine={s['engine']} days={s['days']} rebalances={s['rebalances']} "
        f"avg_turnover={s['avg_turnover']:.3f} total_cost={s['total_cost']:.4%}"
    )
    print(report(res.returns))
    print(f"gross Sharpe={s['gross_sharpe']:.3f}")
    print()
    print(equity_chart(res.equity))
    return 0
