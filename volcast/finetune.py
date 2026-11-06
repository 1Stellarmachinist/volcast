"""Optional Chronos-2 fine-tuning harness.

Nothing here imports torch at module load. On a CPU-only 2-core box this is
documented-but-unused; the guard raises a clear message rather than hanging.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FinetunePlan:
    model_id: str = "amazon/chronos-2"
    context_len: int = 256
    horizon: int = 5
    steps: int = 200
    batch_size: int = 8
    lr: float = 1e-5
    out_dir: str = ".volcast/finetune"


def build_windows(prices: pd.DataFrame, context_len: int, horizon: int, max_windows: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Cut (context, target) windows for fine-tuning without any DL dependency."""
    xs, ys = [], []
    for t in prices.columns:
        s = prices[t].dropna().to_numpy(dtype=float)
        need = context_len + horizon
        if len(s) < need:
            continue
        starts = range(0, len(s) - need + 1, max(horizon, 1))
        for i in starts:
            xs.append(s[i: i + context_len])
            ys.append(s[i + context_len: i + context_len + horizon])
            if len(xs) >= max_windows:
                break
        if len(xs) >= max_windows:
            break
    if not xs:
        return np.empty((0, context_len)), np.empty((0, horizon))
    return np.asarray(xs), np.asarray(ys)


def check_environment() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except Exception:
        return False, "torch is not installed; fine-tuning is disabled on this machine (CPU-only, 2 cores)."
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "no CUDA device; Chronos-2 fine-tuning on CPU would take hours. Use engine='drift'."
    except Exception as exc:  # pragma: no cover
        return False, f"torch probe failed: {exc}"
    return True, "GPU available"


def finetune(plan: FinetunePlan, prices: pd.DataFrame) -> dict:
    ok, why = check_environment()
    if not ok:
        return {"status": "skipped", "reason": why, "plan": plan.__dict__}

    xs, ys = build_windows(prices, plan.context_len, plan.horizon)  # pragma: no cover
    out = Path(plan.out_dir)  # pragma: no cover
    out.mkdir(parents=True, exist_ok=True)  # pragma: no cover
    return {
        "status": "ready",
        "windows": int(len(xs)),
        "out_dir": str(out),
        "note": "Hook your trainer here; this repo ships the data plumbing only.",
    }
