"""Auth routes: passkey login, registration, refresh, logout, me.

BUG #5 lives in this file: ``refresh`` takes ``(body, request, response)`` and
``refresh_cookie`` reads the ``volcast_rt`` cookie off the request and forwards it.
The old one-argument version could not see the cookie at all, so the browser's
refresh path always 401'd while the mobile body-based path appeared to work.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from .. import __version__
from . import db as dbmod
from . import passkeys, security
from .deps import current_claims, db_conn
from .schemas import (
    LoginBeginRequest,
    LoginBeginResponse,
    LoginFinishRequest,
    MeResponse,
    RefreshRequest,
    RegisterBeginRequest,
    RegisterBeginResponse,
    RegisterFinishRequest,
)
from .service import clean_payload
from .settings import Settings, get_settings

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _issue(cfg: Settings, conn, subject: str, display_name: str | None) -> tuple[str, str]:
    access = security.create_access_token(cfg, subject)
    refresh, payload = security.create_refresh_token(cfg, subject)
    security.store_refresh(conn, cfg, payload)
    conn.commit()
    return access, refresh


@router.post("/register/begin", response_model=RegisterBeginResponse)
def register_begin(
    body: RegisterBeginRequest,
    request: Request,
    cfg: Settings = Depends(get_settings),
    conn=Depends(db_conn),
):
    add_to = None
    if body.add_to_existing:
        claims = security.optional_access_claims(cfg, request)
        add_to = claims.get("sub") if claims else None
    out = passkeys.begin_registration(cfg, conn, body.username, body.display_name, add_to)
    return clean_payload(out)


@router.post("/register/finish")
def register_finish(
    body: RegisterFinishRequest,
    response: Response,
    cfg: Settings = Depends(get_settings),
    conn=Depends(db_conn),
):
    try:
        out = passkeys.finish_registration(
            cfg, conn, body.challenge_id, body.credential, body.label
        )
    except passkeys.PasskeyError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    access, refresh = _issue(cfg, conn, out["subject"], out["display_name"])
    security.set_auth_cookies(cfg, response, access, refresh)
    payload = dict(out)
    payload.update(
        {
            "access_token": access,
            "token_type": "bearer",
            "expires_in": cfg.access_ttl_seconds,
        }
    )
    return clean_payload(payload)


@router.post("/login/begin", response_model=LoginBeginResponse)
def login_begin(
    body: LoginBeginRequest,
    cfg: Settings = Depends(get_settings),
    conn=Depends(db_conn),
):
    try:
        out = passkeys.begin_authentication(cfg, conn, body.username)
    except passkeys.PasskeyError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return clean_payload(out)


@router.post("/login/finish")
def login_finish(
    body: LoginFinishRequest,
    response: Response,
    cfg: Settings = Depends(get_settings),
    conn=Depends(db_conn),
):
    try:
        out = passkeys.finish_authentication(cfg, conn, body.challenge_id, body.credential)
    except passkeys.PasskeyError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc

    access, refresh = _issue(cfg, conn, out["subject"], out["display_name"])
    security.set_auth_cookies(cfg, response, access, refresh)
    payload = dict(out)
    payload.update(
        {
            "access_token": access,
            "token_type": "bearer",
            "expires_in": cfg.access_ttl_seconds,
        }
    )
    return clean_payload(payload)


@router.post("/refresh")
def refresh(body: RefreshRequest, request: Request, response: Response):
    """Exchange a refresh token for a new access token.

    BUG #5: exactly three parameters ``(body, request, response)``. Returning a
    ``Response`` object also lets us rewrite the refresh cookie in place.

    The token may arrive in the body (native client) or in the HttpOnly cookie
    (web); the cookie is read off ``request``.
    """
    cfg = get_settings()
    conn = dbmod.connect(cfg.db_path)
    try:
        return _do_refresh(body.refresh_token, request, response, cfg, conn)
    finally:
        conn.close()


@router.post("/refresh/cookie")
def refresh_cookie(
    request: Request,
    response: Response,
    cfg: Settings = Depends(get_settings),
    conn=Depends(db_conn),
):
    """Cookie-only refresh used by the web app.

    BUG #5 FIX: read the ``volcast_rt`` cookie and forward it into the shared
    implementation instead of assuming a JSON body.
    """
    token = request.cookies.get(cfg.refresh_cookie, "")
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing refresh cookie")
    return _do_refresh(token, request, response, cfg, conn)


def _do_refresh(token: str | None, request: Request, response: Response,
                cfg: Settings, conn) -> dict:
    if not token:
        token = request.cookies.get(cfg.refresh_cookie, "")
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "no refresh token supplied")

    try:
        claims = security.decode_refresh_claims(cfg, token)
    except ValueError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc

    try:
        new_refresh, payload = security.rotate_refresh(conn, cfg, claims)
    except ValueError as exc:
        security.clear_auth_cookies(cfg, response)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc

    subject = claims.get("sub", "")
    access = security.create_access_token(cfg, subject)
    security.set_auth_cookies(cfg, response, access, new_refresh)
    return clean_payload(
        {
            "access_token": access,
            "token_type": "bearer",
            "expires_in": cfg.access_ttl_seconds,
            "subject": subject,
        }
    )


@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    cfg: Settings = Depends(get_settings),
    conn=Depends(db_conn),
):
    token = request.cookies.get(cfg.refresh_cookie, "")
    if token:
        try:
            claims = security.decode_refresh_claims(cfg, token)
            security.revoke_family(conn, claims.get("fam", ""))
            conn.commit()
        except ValueError:
            pass
    security.clear_auth_cookies(cfg, response)
    return clean_payload({"ok": True})


@router.get("/me", response_model=MeResponse)
def me(request: Request, cfg: Settings = Depends(get_settings), conn=Depends(db_conn)):
    claims = current_claims(request, cfg)
    subject = claims["sub"]
    creds = passkeys.list_credentials(conn, subject)
    row = dbmod.get(conn, "SELECT display_name FROM passkey_users WHERE id=?", (subject,))
    return clean_payload(
        {
            "subject": subject,
            "display_name": (row["display_name"] if row else None),
            "credentials": creds,
            "device": bool(dbmod.get(conn, "SELECT 1 FROM devices WHERE id=?", (subject,))),
        }
    )


@router.post("/demo")
def demo_login(
    response: Response,
    request: Request,
    cfg: Settings = Depends(get_settings),
    conn=Depends(db_conn),
):
    """Dev-only: sign in without a passkey so the dashboard is visible immediately.

    Refused outright when ``env=prod``. This exists because a single-user local
    install should not require a hardware ceremony before it will tell you
    anything -- being unable to see the product was the real reason the UI was
    incomprehensible.
    """
    if not cfg.demo_enabled:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "demo login is disabled (env=prod); register a passkey instead",
        )

    subject = "usr_demo_local"
    conn.execute(
        "INSERT OR IGNORE INTO passkey_users (id, username, display_name, created_at)"
        " VALUES (?,?,?,?)",
        (subject, "demo", "Demo (local)", dbmod.now()),
    )
    conn.commit()

    access, refresh = _issue(cfg, conn, subject, "Demo (local)")
    security.set_auth_cookies(cfg, response, access, refresh)
    return clean_payload(
        {
            "access_token": access,
            "token_type": "bearer",
            "expires_in": cfg.access_ttl_seconds,
            "subject": subject,
            "display_name": "Demo (local)",
            "demo": True,
        }
    )


@router.get("/config")
def auth_config(cfg: Settings = Depends(get_settings)):
    """Public bits the client needs to build its WebAuthn options."""
    return clean_payload(
        {
            "rp_id": cfg.rp_id,
            "rp_name": cfg.rp_name,
            "origins": cfg.origins,
            "user_verification": "required",
            "resident_key": "required",
            "conditional_ui": True,
            "demo_login": cfg.demo_enabled,
            "version": __version__,
        }
    )


@router.post("/device")
def register_device(cfg: Settings = Depends(get_settings), conn=Depends(db_conn)):
    """Anonymous device identity (used only for non-passkey access)."""
    did = security.new_device_id()
    conn.execute(
        "INSERT INTO devices (id, label, created_at, last_seen_at) VALUES (?,?,?,?)",
        (did, "anonymous", dbmod.now(), dbmod.now()),
    )
    conn.commit()
    return clean_payload({"subject": did})
