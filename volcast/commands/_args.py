"""Shared argument wiring and the config/panel bootstrap for commands."""

from __future__ import annotations

import argparse

from ..config import Config, load_config
from ..data import load_prices
from ..panel import build_panel


def add_common(p: argparse.ArgumentParser) -> None:
    # Accept -c/--config on the subcommand too (``volcast strat -c x.toml`` reads
    # more naturally than the argparse-mandated global-before-subcommand form).
    # SUPPRESS keeps the parent parser's value when the flag is omitted here.
    p.add_argument("-c", "--config", default=argparse.SUPPRESS, help="path to a TOML config")
    p.add_argument("--tickers", nargs="*", default=None, help="override the universe")
    p.add_argument("--period", default="2y", help="yfinance lookback period")
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--vol-window", type=int, default=None)
    p.add_argument("--offline", action="store_true", help="use synthetic prices")
    p.add_argument("--no-cache", action="store_true")


def dispatch(args) -> int:  # pragma: no cover - replaced by cli.main properly
    raise SystemExit("dispatch should not be called directly")


def resolve_config(args) -> Config:
    path = getattr(args, "config", None)
    cfg = load_config(path) if path else Config()
    if getattr(args, "engine", None):
        cfg.engine = args.engine
    for attr, field in (
        ("tickers", "universe"),
        ("horizon", "horizon"),
        ("vol_window", "vol_window"),
        ("target_vol", "target_vol"),
        ("cost_bps", "cost_bps"),
        ("test_days", "test_days"),
        ("rebalance", "rebalance"),
        ("top", "screen_top"),
    ):
        v = getattr(args, attr, None)
        if v is not None:
            setattr(cfg, field, v)
    return cfg


def load_panel(args, cfg: Config):
    prices = load_prices(
        cfg.universe,
        period=getattr(args, "period", "2y"),
        use_cache=not getattr(args, "no_cache", False),
        allow_synthetic=True,
        cache_root=cfg.cache_dir,
    )
    return prices, build_panel(prices, vol_window=cfg.vol_window)
