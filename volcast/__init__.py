"""volcast - short-horizon volatility forecasting and net-of-cost strategy research.

Engine defaults to the analytic `drift` baseline, which needs only numpy/pandas.
Chronos-2 is an optional extra and is *never* imported at module load time.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Config, load_config  # noqa: E402
from .forecast import drift_forecast, forecast_panel  # noqa: E402

__all__ = ["Config", "load_config", "drift_forecast", "forecast_panel", "__version__"]
