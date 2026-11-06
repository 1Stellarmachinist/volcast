"""volcast command line interface."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .commands import _args


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="volcast",
        description="Short-horizon volatility forecasting and net-of-cost strategy research.",
    )
    p.add_argument("--version", action="version", version=f"volcast {__version__}")
    p.add_argument("-c", "--config", default=None, help="path to a TOML config")
    p.add_argument("--engine", default=None, choices=["drift", "chronos"], help="forecast engine")

    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("forecast", help="print a forecast table")
    _args.add_common(f)
    f.add_argument("--top", type=int, default=None)
    f.set_defaults(func=_args.dispatch, handler="forecast")

    s = sub.add_parser("screen", help="rank the universe by features")
    _args.add_common(s)
    s.add_argument("--top", type=int, default=None)
    s.set_defaults(func=_args.dispatch, handler="screen")

    b = sub.add_parser("strat", help="run a vol-targeted cross-sectional backtest")
    _args.add_common(b)
    b.add_argument("--target-vol", type=float, default=None)
    b.add_argument("--cost-bps", type=float, default=None)
    b.add_argument("--test-days", type=int, default=None)
    b.add_argument("--rebalance", type=int, default=None)
    b.add_argument("--no-neutral", action="store_true")
    b.set_defaults(func=_args.dispatch, handler="strat")

    c = sub.add_parser("cs", help="cross-sectional weights for the latest bar")
    _args.add_common(c)
    c.set_defaults(func=_args.dispatch, handler="cs")

    g = sub.add_parser("pick", help="rank small caps and say whether the ranking works")
    _args.add_common(g)
    g.add_argument("--top", type=int, default=None)
    g.add_argument("--no-fundamentals", action="store_true",
                   help="skip the current-snapshot fundamentals fetch")
    g.add_argument("--evaluate", action="store_true",
                   help="also measure whether the ranking predicted anything")
    g.set_defaults(func=_args.dispatch, handler="growth")

    sp = sub.add_parser("subpenny", help="volume/MA breakout scan for sub-$1 OTC names")
    _args.add_common(sp)
    sp.add_argument("--top", type=int, default=20)
    sp.add_argument("--rvol", type=float, default=1.5, help="min relative volume")
    sp.add_argument("--above-ma20", action="store_true", help="only names above MA20")
    sp.set_defaults(func=_args.dispatch, handler="subpenny")

    a = sub.add_parser("alpaca", help="measure real spreads via Alpaca (needs API keys)")
    _args.add_common(a)
    a.add_argument("--top", type=int, default=None)
    a.add_argument("--days", type=int, default=60, help="days of daily bars to fetch")
    a.add_argument("--universe", action="store_true", help="check symbols against Alpaca tradable list")
    a.set_defaults(func=_args.dispatch, handler="alpaca")

    t = sub.add_parser("ft", help="Chronos-2 fine-tuning (GPU only)")
    _args.add_common(t)
    t.set_defaults(func=_args.dispatch, handler="ft")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from .commands import alpaca as cmd_alpaca
    from .commands import cs as cmd_cs
    from .commands import forecast as cmd_forecast
    from .commands import ft as cmd_ft
    from .commands import growth as cmd_growth
    from .commands import screen as cmd_screen
    from .commands import strat as cmd_strat
    from .commands import subpenny as cmd_subpenny

    handlers = {
        "forecast": cmd_forecast.run,
        "screen": cmd_screen.run,
        "strat": cmd_strat.run,
        "cs": cmd_cs.run,
        "growth": cmd_growth.run,
        "alpaca": cmd_alpaca.run,
        "subpenny": cmd_subpenny.run,
        "ft": cmd_ft.run,
    }
    fn = handlers.get(args.handler)
    if fn is None:
        parser.error(f"unknown command {args.handler}")
        return 2
    return int(fn(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
