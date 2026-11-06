"""JWT access/refresh tokens, cookies, and the passkey-aware subject check.

Bugs addressed here:
  #3  ``decode_access_claims(cfg, token) -> dict`` exists; ``read_access_token``
      wraps it instead of re-implementing (and re-failing) the decode.
  #4  Refresh must accept passkey subjects -> ``_known_subject`` looks in
      ``passkey_users`` *and* ``devices``; previously only devices were checked so
      a passkey-derived refresh token was rejected with 401 on every attempt.
  #8  ``cookie_secure`` comes from settings (derived from ``is_prod``).
"""

from __future__ import annotations

import secrets
import time
import uuid
from typing import Any

import jwt
from fastapi import HTTPException, Request, Response, status

from . import db as dbmod
from .settings import Settings

ACCESS = "access"
REFRESH = "refresh"


# --------------------------------------------------------------------------- #
# Token creation / decoding
# --------------------------------------------------------------------------- #
def _now() -> int:
    return int(time.time())


def create_access_token(cfg: Settings, subject: str, **claims: Any) -> str:
    payload = {
        "sub": str(subject),
        "typ": ACCESS,
        "iat": _now(),
        "exp": _now() + int(cfg.access_ttl_seconds),
        "jti": uuid.uuid4().hex,
        **claims,
    }
    return jwt.encode(payload, cfg.jwt_secret, algorithm=cfg.jwt_alg)


def create_refresh_token(
    cfg: Settings, subject: str, family: str | None = None
) -> tuple[str, dict[str, Any]]:
    fam = family or uuid.uuid4().hex
    jti = uuid.uuid4().hex
    payload = {
        "sub": str(subject),
        "typ": REFRESH,
        "iat": _now(),
        "exp": _now() + int(cfg.refresh_ttl_seconds),
        "jti": jti,
        "fam": fam,
    }
    token = jwt.encode(payload, cfg.jwt_secret, algorithm=cfg.jwt_alg)
    return token, payload


def decode_access_claims(cfg: Settings, token: str) -> dict:
    """Decode and validate an access token, returning its claims.

    BUG #3: this function is the single source of truth. It is deliberately a
    plain function returning a dict (not raising HTTPException for a *malformed*
    token vs an *expired* one) so it can be unit-tested without a request.
    """
    if not token or not isinstance(token, str):
        raise ValueError("missing token")
    try:
        claims = jwt.decode(
            token,
            cfg.jwt_secret,
            algorithms=[cfg.jwt_alg],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise ValueError("token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise ValueError(f"invalid token: {exc}") from exc

    if claims.get("typ") != ACCESS:
        raise ValueError("wrong token type")
    return dict(claims)


def decode_refresh_claims(cfg: Settings, token: str) -> dict:
    if not token:
        raise ValueError("missing token")
    try:
        claims = jwt.decode(token, cfg.jwt_secret, algorithms=[cfg.jwt_alg])
    except jwt.ExpiredSignatureError as exc:
        raise ValueError("refresh token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise ValueError(f"invalid refresh token: {exc}") from exc
    if claims.get("typ") != REFRESH:
        raise ValueError("wrong token type")
    return dict(claims)


def read_access_token(cfg: Settings, request: Request) -> dict:
    """Pull the access token off the request and return validated claims.

    Reads ``Authorization: Bearer`` first, then the access cookie. Raises 401 on
    any failure so route handlers never see an unvalidated claims dict.
    """
    raw = ""
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        raw = header[7:].strip()
    if not raw:
        raw = request.cookies.get(cfg.access_cookie, "")

    try:
        return decode_access_claims(cfg, raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def optional_access_claims(cfg: Settings, request: Request) -> dict | None:
    try:
        return read_access_token(cfg, request)
    except HTTPException:
        return None


# --------------------------------------------------------------------------- #
# Refresh-token persistence + rotation
# --------------------------------------------------------------------------- #
def store_refresh(conn, cfg: Settings, payload: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO refresh_tokens (id, subject, family, expires_at, revoked, created_at)"
        " VALUES (?,?,?,?,0,?)",
        (payload["jti"], payload["sub"], payload.get("fam", ""), payload["exp"], dbmod.now()),
    )


def revoke_family(conn, family: str) -> int:
    cur = conn.execute("UPDATE refresh_tokens SET revoked=1 WHERE family=?", (family,))
    return cur.rowcount or 0


def is_refresh_active(conn, jti: str) -> bool:
    row = dbmod.get(conn, "SELECT revoked FROM refresh_tokens WHERE id=?", (jti,))
    return bool(row) and not int(row["revoked"])


def _known_subject(conn, subject: str) -> bool:
    """BUG #4 FIX: a subject is valid if it is a passkey user OR a device.

    The original only consulted ``devices``, so users who signed up with a
    discoverable passkey (no device row) were told their refresh token was
    invalid, forcing a re-login every 15 minutes.
    """
    if not subject:
        return False
    if dbmod.get(conn, "SELECT 1 FROM passkey_users WHERE id=?", (subject,)):
        return True
    if dbmod.get(conn, "SELECT 1 FROM devices WHERE id=?", (subject,)):
        return True
    return False


def rotate_refresh(conn, cfg: Settings, claims: dict) -> tuple[str, dict]:
    """Rotate a refresh token, reusing detection: revoked family -> revoke all."""
    jti = claims.get("jti", "")
    family = claims.get("fam", "")
    subject = claims.get("sub", "")

    if not _known_subject(conn, subject):
        raise ValueError("unknown subject for refresh token")

    if not is_refresh_active(conn, jti):
        # Token reuse: the family is compromised, kill it.
        revoke_family(conn, family)
        raise ValueError("refresh token already used")

    conn.execute("UPDATE refresh_tokens SET revoked=1 WHERE id=?", (jti,))
    token, payload = create_refresh_token(cfg, subject, family=family)
    store_refresh(conn, cfg, payload)
    # Commit here: the caller hands this connection straight back to the
    # dependency teardown, and an uncommitted rotation is silently rolled back on
    # close -- which presents as "refresh token already used" on the very next
    # request, because the new jti never made it into the table.
    conn.commit()
    return token, payload


# --------------------------------------------------------------------------- #
# Cookie helpers
# --------------------------------------------------------------------------- #
def set_auth_cookies(cfg: Settings, response: Response, access: str, refresh: str) -> None:
    common = {
        "httponly": True,
        # Bug #8: never force Secure on http://localhost.
        "secure": cfg.secure_cookies,
        "samesite": cfg.cookie_samesite,
        "path": "/",
    }
    if cfg.cookie_domain:
        common["domain"] = cfg.cookie_domain

    response.set_cookie(cfg.access_cookie, access, max_age=cfg.access_ttl_seconds, **common)
    response.set_cookie(
        cfg.refresh_cookie,
        refresh,
        max_age=cfg.refresh_ttl_seconds,
        **{**common, "path": "/api/auth"},
    )


def clear_auth_cookies(cfg: Settings, response: Response) -> None:
    response.delete_cookie(cfg.access_cookie, path="/")
    response.delete_cookie(cfg.refresh_cookie, path="/api/auth")


def new_device_id() -> str:
    return "dev_" + secrets.token_urlsafe(12)
