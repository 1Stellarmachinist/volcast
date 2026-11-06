"""Dependency-free terminal plotting (no matplotlib on the dep list).

Everything here degrades to pure ASCII when the console codepage cannot encode
box-drawing characters. A Windows console defaults to cp1252, so printing "│"
raises UnicodeEncodeError and kills the command *after* the numbers were computed
-- the worst possible failure mode.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

_UTF8_BLOCKS = "▁▂▃▄▅▆▇█"
_ASCII_BLOCKS = ".:-=+*#@"


def _console_is_utf8() -> bool:
    """True if stdout can encode the box-drawing glyphs we would like to use."""
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "│└─●▁".encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def blocks() -> tuple[str, str]:
    """(block ramp, palette) adjusted for the current console."""
    return (_UTF8_BLOCKS, "utf8") if _console_is_utf8() else (_ASCII_BLOCKS, "ascii")


def sparkline(values, width: int | None = None) -> str:
    ramp, _ = blocks()
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return ""
    if width and arr.size > width:
        idx = np.linspace(0, arr.size - 1, width).astype(int)
        arr = arr[idx]
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return ramp[0] * arr.size
    scaled = (arr - lo) / (hi - lo) * (len(ramp) - 1)
    return "".join(ramp[int(round(s))] for s in scaled)


def equity_chart(equity: pd.Series, width: int = 62, height: int = 9) -> str:
    """ASCII/TUI equity curve with a labelled y axis."""
    eq = pd.Series(equity).dropna()
    if len(eq) == 0:
        return "(no data)"

    _, palette = blocks()
    if palette == "utf8":
        axis, corner, bar, dot = "│", "└", "─", "●"
    else:
        axis, corner, bar, dot = "|", "+", "-", "*"

    if len(eq) > width:
        idx = np.linspace(0, len(eq) - 1, width).astype(int)
        eq = eq.iloc[idx]

    lo, hi = float(eq.min()), float(eq.max())
    if hi - lo < 1e-12:
        hi = lo + 1e-9

    rows = [[" "] * len(eq) for _ in range(height)]
    for x, v in enumerate(eq.to_numpy()):
        y = int(round((v - lo) / (hi - lo) * (height - 1)))
        rows[height - 1 - y][x] = dot

    lines = []
    for r, row in enumerate(rows):
        val = hi - (hi - lo) * r / (height - 1)
        lines.append(f"{val:7.3f} {axis}" + "".join(row))
    lines.append("        " + corner + bar * len(eq))
    lines.append(f"         {eq.index[0].date()}  ->  {eq.index[-1].date()}")
    return "\n".join(lines)


def table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df is None or len(df) == 0:
        return "(empty)"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{v:{floatfmt}}" if pd.notna(v) else "nan")
    return d.to_string()
