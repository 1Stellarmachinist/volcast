from __future__ import annotations

from ..indicators import feature_table
from ..plot import table
from ._args import load_panel, resolve_config


def run(args) -> int:
    cfg = resolve_config(args)
    prices, panel = load_panel(args, cfg)

    feats = feature_table(panel.prices, vol_window=cfg.vol_window).dropna(how="all")
    # Low vol + positive momentum ranks first.
    feats = feats.assign(
        score=-feats["vol_20"].rank(pct=True) + feats["mom_63"].rank(pct=True)
    ).sort_values("score", ascending=False)

    n = cfg.screen_top
    print(f"screened {len(feats)} tickers, showing top {min(n, len(feats))}")
    print(table(feats.head(n), floatfmt=".4f"))
    return 0
