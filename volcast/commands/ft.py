from __future__ import annotations

from ..finetune import FinetunePlan, check_environment, finetune
from ._args import load_panel, resolve_config


def run(args) -> int:
    cfg = resolve_config(args)
    ok, why = check_environment()
    print(f"environment: {'OK' if ok else 'NOT USABLE'} - {why}")

    prices, _panel = load_panel(args, cfg)
    plan = FinetunePlan(
        model_id=cfg.chronos_model,
        context_len=cfg.context_len,
        horizon=cfg.horizon,
    )
    result = finetune(plan, prices)
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0
