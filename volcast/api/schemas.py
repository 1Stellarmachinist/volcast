"""Pydantic response/request schemas.

Every response goes through ``_clean`` (see ``service.py``) before serialisation,
because ``float('nan')`` is not valid JSON and FastAPI would emit the literal
token ``NaN``, which ``JSON.parse`` rejects on the client.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
class RegisterBeginRequest(BaseModel):
    username: str | None = Field(default=None, description="Omitted for usernameless")
    display_name: str | None = None
    add_to_existing: bool = Field(
        default=False,
        description="Attach the new credential to the already-signed-in user instead of forking the account",
    )


class RegisterBeginResponse(BaseModel):
    challenge_id: str
    public_key: dict[str, Any]
    user_id: str


class RegisterFinishRequest(BaseModel):
    challenge_id: str
    credential: dict[str, Any]
    label: str | None = None


class LoginBeginRequest(BaseModel):
    username: str | None = None


class LoginBeginResponse(BaseModel):
    challenge_id: str
    public_key: dict[str, Any]


class LoginFinishRequest(BaseModel):
    challenge_id: str
    credential: dict[str, Any]


class TokenResponse(BaseModel):
    access_token: str | None = None
    token_type: str = "bearer"
    expires_in: int = 900
    subject: str | None = None
    display_name: str | None = None


class RefreshRequest(BaseModel):
    """Body is optional: the refresh token normally rides in an HttpOnly cookie."""

    refresh_token: str | None = None


class MeResponse(BaseModel):
    subject: str
    display_name: str | None = None
    credentials: list[dict[str, Any]] = Field(default_factory=list)
    device: bool = False


class PasskeyInfo(BaseModel):
    id: str
    label: str | None = None
    created_at: float | None = None
    last_used_at: float | None = None
    backed_up: bool = False
    transports: list[str] = Field(default_factory=list)


class PasskeyListResponse(BaseModel):
    credentials: list[PasskeyInfo]
    count: int = 0


class PasskeyRenameRequest(BaseModel):
    label: str


# --------------------------------------------------------------------------- #
# Market
# --------------------------------------------------------------------------- #
class ForecastRow(BaseModel):
    ticker: str
    sigma: float
    sigma_h: float
    mu: float
    ann_vol: float
    engine: str = "drift"
    horizon: int = 5


class ForecastResponse(BaseModel):
    as_of: str
    engine: str
    horizon: int
    rows: list[ForecastRow]


class ScreenRow(BaseModel):
    ticker: str
    last: float | None = None
    vol_20: float | None = None
    mom_63: float | None = None
    z_60: float | None = None
    score: float | None = None


class ScreenResponse(BaseModel):
    as_of: str
    rows: list[ScreenRow]


class WeightRow(BaseModel):
    ticker: str
    signal_w: float
    target_w: float


class WeightsResponse(BaseModel):
    as_of: str
    net: float
    gross: float
    rows: list[WeightRow]


class BacktestResponse(BaseModel):
    engine: str
    days: int
    sharpe: float | None = None
    sharpe_se: float | None = None
    gross_sharpe: float | None = None
    ann_return: float | None = None
    ann_vol: float | None = None
    max_drawdown: float | None = None
    total_cost: float | None = None
    avg_turnover: float | None = None
    significant: bool | None = None
    equity: list[float] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)


class QuoteRow(BaseModel):
    ticker: str
    last: float | None = None
    change_1d: float | None = None


class QuoteResponse(BaseModel):
    as_of: str
    rows: list[QuoteRow]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    version: str = "0.1.0"
    engine: str = "drift"
    chronos_available: bool = False
