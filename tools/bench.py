#!/usr/bin/env python
"""Micro-benchmark: how expensive is one forecast / backtest on this machine?

Run this before enabling anything heavier than the drift engine.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from volcast.backtest import run_backtest  # noqa: E402
from volcast.config import load_config  # noqa: E402
from volcast.data import synthetic_prices  # noqa: E402
from volcast.forecast import drift_forecast, forecast_panel  # noqa: E402
from volcast.panel import build_panel  # noqa: E402


def timeit(fn, repeats: int = 5) -> dict:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "runs": repeats,
        "mean_ms": round(statistics.mean(times), 3),
        "median_ms": round(statistics.median(times), 3),
        "max_ms": round(max(times), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="benchmark volcast on this machine")
    ap.add_argument("--config", default="configs/lowspec.toml")
    ap.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = ap.parse_args()

    cfg_path = ROOT / args.config
    cfg = load_config(cfg_path) if cfg_path.is_file() else load_config(None)
    cfg.engine = "drift"  # the only engine that is sane here

    px = synthetic_prices(cfg.universe, periods=cfg.lookback + cfg.test_days, seed=5)
    t0 = time.perf_counter()
    panel = build_panel(px, vol_window=cfg.vol_window)
    panel_ms = (time.perf_counter() - t0) * 1000.0

    results = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cores": __import__("os").cpu_count(),
        },
        "config": {
            "engine": cfg.engine,
            "horizon": cfg.horizon,
            "universe": len(cfg.universe),
            "lookback": cfg.lookback,
            "test_days": cfg.test_days,
            "rebalance": cfg.rebalance,
        },
        "build_panel_ms": round(panel_ms, 2),
        "forecast_1bar": timeit(lambda: drift_forecast(panel.returns, horizon=cfg.horizon)),
        "forecast_table": timeit(
            lambda: forecast_panel(panel.returns, horizon=cfg.horizon, engine="drift")
        ),
        "backtest_full": timeit(
            lambda: run_backtest(
                panel,
                target_vol=cfg.target_vol,
                max_weight=cfg.max_weight,
                cost_bps=cfg.cost_bps,
                horizon=cfg.horizon,
                rebalance=cfg.rebalance,
                test_days=cfg.test_days,
            ),
            repeats=3,
        ),
    }

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print("=" * 66)
    print("  volcast benchmark")
    print("=" * 66)
    print(f"  machine        : {results['machine']['platform']}")
    print(f"  cores          : {results['machine']['cores']}")
    print(f"  python/numpy   : {results['machine']['python']} / {results['machine']['numpy']}")
    print(f"  build_panel    : {results['build_panel_ms']:.1f} ms "
          f"({results['config']['universe']} tickers x "
          f"{cfg.lookback + cfg.test_days} bars)")
    for key in ("forecast_1bar", "forecast_table", "backtest_full"):
        r = results[key]
        print(f"  {key:15s}: mean {r['mean_ms']:8.1f} ms  median {r['median_ms']:8.1f} ms  "
              f"max {r['max_ms']:8.1f} ms")
    print("=" * 66)
    print("  Chronos-2 is intentionally NOT benchmarked: it requires torch and would")
    print("  take minutes-to-hours per forecast on this CPU. Stay on engine=drift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
