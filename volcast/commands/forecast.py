from __future__ import annotations

from ..forecast import forecast_panel
from ..plot import table
from ._args import load_panel, resolve_config


def run(args) -> int:
    cfg = resolve_config(args)
    prices, panel = load_panel(args, cfg)

    fc = forecast_panel(
        panel.returns,
        horizon=cfg.horizon,
        window=cfg.vol_window,
        engine=cfg.engine,
        prices=panel.prices,
        context_len=cfg.context_len,
    )
    if getattr(args, "top", None):
        fc = fc.nlargest(args.top, "ann_vol")

    print(f"engine={cfg.engine} horizon={cfg.horizon} vol_window={cfg.vol_window} "
          f"bars={len(panel.returns)} tickers={len(panel.tickers)}")
    print(table(fc.set_index("ticker"), floatfmt=".6f"))
    return 0
