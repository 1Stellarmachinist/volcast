"""Market data routes (forecasts, screening, weights, backtests)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..config import DEFAULT_UNIVERSE
from .schemas import (
    BacktestResponse,
    ForecastResponse,
    QuoteResponse,
    ScreenResponse,
    WeightsResponse,
)
from .service import (
    backtest_payload,
    clean_payload,
    forecast_payload,
    quotes_payload,
    screen_payload,
    weights_payload,
)
from .settings import Settings, get_settings

router = APIRouter(prefix="/api/market", tags=["market"])


def _tickers(tickers: str | None) -> list[str]:
    if not tickers:
        return list(DEFAULT_UNIVERSE)
    out = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not out:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty ticker list")
    if len(out) > 30:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "at most 30 tickers per request")
    return out


@router.get("/forecast", response_model=ForecastResponse)
def forecast(
    tickers: str | None = Query(default=None, description="comma separated"),
    horizon: int = Query(default=5, ge=1, le=60),
    window: int = Query(default=20, ge=5, le=252),
    engine: str = Query(default="drift", pattern="^(drift|chronos)$"),
    cfg: Settings = Depends(get_settings),
):
    if engine == "chronos" and not cfg.allow_chronos:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "chronos engine is disabled on this host (set VOLCAST_ALLOW_CHRONOS=1 on a GPU machine)",
        )
    return clean_payload(
        forecast_payload(_tickers(tickers), horizon=horizon, window=window, engine=engine)
    )


@router.get("/screen", response_model=ScreenResponse)
def screen(
    tickers: str | None = None,
    top: int = Query(default=10, ge=1, le=30),
):
    return clean_payload(screen_payload(_tickers(tickers), top=top))


@router.get("/weights", response_model=WeightsResponse)
def weights(
    tickers: str | None = None,
    target_vol: float = Query(default=0.15, gt=0, le=2.0),
    max_weight: float = Query(default=0.25, gt=0, le=1.0),
    horizon: int = Query(default=5, ge=1, le=60),
):
    return clean_payload(
        weights_payload(
            _tickers(tickers), horizon=horizon, target_vol=target_vol, max_weight=max_weight
        )
    )


@router.get("/backtest", response_model=BacktestResponse)
def backtest(
    tickers: str | None = None,
    cost_bps: float = Query(default=5.0, ge=0, le=200),
    test_days: int = Query(default=126, ge=21, le=504),
    rebalance: int = Query(default=5, ge=1, le=60),
    target_vol: float = Query(default=0.15, gt=0, le=2.0),
):
    return clean_payload(
        backtest_payload(
            _tickers(tickers),
            cost_bps=cost_bps,
            test_days=test_days,
            rebalance=rebalance,
            target_vol=target_vol,
        )
    )


@router.get("/quotes", response_model=QuoteResponse)
def quotes(tickers: str | None = None):
    return clean_payload(quotes_payload(_tickers(tickers)))


@router.get("/universe")
def universe():
    return clean_payload({"default": list(DEFAULT_UNIVERSE), "count": len(DEFAULT_UNIVERSE)})


@router.get("/penny")
def penny(
    tickers: str | None = None,
    rvol_min: float = Query(default=1.5, ge=1.0, le=20.0),
):
    """Card feed for the penny-stock dashboard."""
    from .service import penny_board_payload

    return clean_payload(
        penny_board_payload(_tickers_optional(tickers), rvol_min=rvol_min)
    )


def _tickers_optional(tickers: str | None) -> list[str] | None:
    if not tickers:
        return None
    out = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    return out or None
