"""Configuration: tolerant TOML/env loading with sane low-spec defaults."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import tomllib

DEFAULT_UNIVERSE = [
    "SPY",
    "QQQ",
    "IWM",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
]


@dataclass
class Config:
    """Runtime configuration shared by the library, CLI and API."""

    engine: str = "drift"
    horizon: int = 5
    vol_window: int = 20

    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))

    target_vol: float = 0.15
    max_weight: float = 0.25
    cost_bps: float = 5.0
    lookback: int = 252
    test_days: int = 126
    rebalance: int = 5

    screen_top: int = 5

    # Chronos knobs (unused unless engine == "chronos")
    context_len: int = 256
    chronos_model: str = "amazon/chronos-2"

    # Caching
    cache_dir: str = ".volcast/cache"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce(dc: type, raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only known fields and coerce types to the dataclass annotations."""
    known = {f.name: f for f in fields(dc)}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in known:
            continue
        ann = str(known[key].type)
        try:
            if "list" in ann and isinstance(value, (list, tuple)):  # type: ignore[operator]
                out[key] = [str(v) for v in value]
            elif "int" in ann and not isinstance(value, bool):
                out[key] = int(value)
            elif "float" in ann:
                out[key] = float(value)
            elif "str" in ann and not isinstance(value, list):
                out[key] = str(value)
            else:
                out[key] = value
        except (TypeError, ValueError):
            continue
    return out


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load config from TOML, then overlay ``VOLCAST_*`` environment variables."""
    cfg = Config()

    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"config file not found: {p}")
        with p.open("rb") as fh:
            raw = tomllib.load(fh)
        cfg = Config(**_coerce(Config, raw))

    env = {}
    for name, f in ((f.name, f) for f in fields(Config)):
        key = f"VOLCAST_{name.upper()}"
        if key in os.environ:
            val: Any = os.environ[key]
            if "list" in str(f.type):
                env[name] = [s.strip() for s in val.split(",") if s.strip()]
            else:
                env[name] = val
    if env:
        merged = cfg.to_dict()
        merged.update(_coerce(Config, env))
        cfg = Config(**merged)

    return cfg
