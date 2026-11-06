"""FastAPI dependencies: per-request DB connection and auth claims."""

from __future__ import annotations

from fastapi import Depends, Request

from . import db as dbmod
from . import security
from .settings import Settings, get_settings


def db_conn(cfg: Settings = Depends(get_settings)):
    """Yield a short-lived SQLite connection (WAL keeps concurrent reads safe)."""
    conn = dbmod.connect(cfg.db_path)
    try:
        yield conn
    finally:
        conn.close()


def current_claims(request: Request, cfg: Settings = Depends(get_settings)) -> dict:
    return security.read_access_token(cfg, request)
