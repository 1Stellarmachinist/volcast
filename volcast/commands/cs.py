from __future__ import annotations

from ..forecast import forecast_panel
from ..plot import table
from ..signals import cross_sectional_weights
from ..sizing import target_positions
from ._args import load_panel, resolve_config


def run(args) -> int:
    cfg = resolve_config(args)
    prices, panel = load_panel(args, cfg)

    feats = panel.last_features()
    vol = feats.get("vol")
    mom = feats.get("momentum")

    if vol is None or mom is None:
        print("insufficient history for cross-sectional weights")
        return 1

    score = (-vol.rank(pct=True) + 0.5) * 1.0 + (mom.rank(pct=True) - 0.5) * 1.0
    raw = cross_sectional_weights(score * 2.0, max_weight=cfg.max_weight)

    fc = forecast_panel(
        panel.returns, horizon=cfg.horizon, window=cfg.vol_window, engine=cfg.engine
    )
    sigma_h = fc.set_index("ticker")["sigma_h"]

    sized = target_positions(
        raw, sigma_h, target_vol=cfg.target_vol, horizon=cfg.horizon,
        max_weight=cfg.max_weight, dollar_neutral=True,
    )

    out = (
        feats.assign(score=score, signal_w=raw, target_w=sized)
        .sort_values("target_w", ascending=False)
    )
    print(f"as of {panel.prices.index[-1].date()}  "
          f"net={sized.sum():+.4f}  gross={sized.abs().sum():.4f}")
    print(table(out[["score", "signal_w", "target_w"]], floatfmt=".4f"))
    return 0
