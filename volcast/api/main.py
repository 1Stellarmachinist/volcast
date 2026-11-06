"""FastAPI application factory."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from . import db as dbmod
from .routes_auth import router as auth_router
from .routes_market import router as market_router
from .routes_passkey import router as passkey_router
from .service import clean_payload
from .settings import get_settings


class SafeJSONResponse(JSONResponse):
    """JSONResponse that can never serialise NaN/Infinity.

    ``json.dumps`` permits NaN by default, which produces ``NaN`` literals the
    browser's JSON.parse rejects. Between ``_clean`` in the service layer and this
    response class there is no path for a non-finite float to reach the wire.
    """

    def render(self, content) -> bytes:
        return json.dumps(
            clean_payload(content),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")


def create_app() -> FastAPI:
    cfg = get_settings()
    dbmod.init_db(cfg.db_path)
    Path(cfg.db_path).parent.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="volcast api",
        version=__version__,
        default_response_class=SafeJSONResponse,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.origins,
        allow_credentials=True,       # required for the HttpOnly refresh cookie
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth_router)
    app.include_router(passkey_router)
    app.include_router(market_router)

    @app.get("/api/health")
    def health():
        chronos = False
        try:  # never import torch on a request unless it is genuinely present
            import importlib.util

            chronos = importlib.util.find_spec("chronos") is not None
        except Exception:
            chronos = False
        return clean_payload(
            {
                "status": "ok",
                "version": __version__,
                "engine": cfg.engine,
                "chronos_available": bool(chronos),
                "env": cfg.env,
                "secure_cookies": cfg.secure_cookies,
            }
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):  # pragma: no cover
        return SafeJSONResponse(
            status_code=500,
            content={"detail": "internal error", "error": type(exc).__name__},
        )

    # Serve the built PWA if it exists (web/dist). Harmless when absent.
    dist = Path(__file__).resolve().parents[2] / "web" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="web")

    return app


app = create_app()
