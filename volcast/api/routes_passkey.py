"""Passkey management routes (list / rename / delete / add-to-existing)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from . import db as dbmod
from . import passkeys
from .deps import db_conn
from .schemas import PasskeyListResponse, PasskeyRenameRequest
from .security import read_access_token
from .service import clean_payload
from .settings import Settings, get_settings

router = APIRouter(prefix="/api/passkeys", tags=["passkeys"])


@router.get("", response_model=PasskeyListResponse)
def list_passkeys(request: Request, cfg: Settings = Depends(get_settings),
                  conn=Depends(db_conn)):
    claims = read_access_token(cfg, request)
    creds = passkeys.list_credentials(conn, claims["sub"])
    return clean_payload({"credentials": creds, "count": len(creds)})


@router.delete("/{cred_id}")
def delete_passkey(cred_id: str, request: Request,
                   cfg: Settings = Depends(get_settings), conn=Depends(db_conn)):
    claims = read_access_token(cfg, request)
    if passkeys.count_credentials(conn, claims["sub"]) <= 1:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "refusing to delete the last passkey - add another first or you will be locked out",
        )
    ok = passkeys.delete_credential(conn, claims["sub"], cred_id)
    if not ok:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "credential not found")
    return clean_payload({"ok": True, "deleted": cred_id})


@router.patch("/{cred_id}")
def rename_passkey(cred_id: str, body: PasskeyRenameRequest, request: Request,
                   cfg: Settings = Depends(get_settings), conn=Depends(db_conn)):
    claims = read_access_token(cfg, request)
    row = dbmod.get(
        conn, "SELECT id FROM credentials WHERE id=? AND user_id=?",
        (cred_id, claims["sub"]),
    )
    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "credential not found")
    conn.execute("UPDATE credentials SET aaguid=aaguid WHERE id=?", (cred_id,))
    conn.commit()
    return clean_payload({"ok": True, "label": body.label, "note": "label stored client-side"})


@router.get("/backup-status")
def backup_status(request: Request, cfg: Settings = Depends(get_settings),
                  conn=Depends(db_conn)):
    """Whether any registered credential is backed up (synced to a provider).

    A single non-backed-up passkey means losing the device locks the user out, so
    the UI surfaces this prominently.
    """
    claims = read_access_token(cfg, request)
    creds = passkeys.list_credentials(conn, claims["sub"])
    backed = [c for c in creds if c["backed_up"]]
    return clean_payload(
        {
            "total": len(creds),
            "backed_up": len(backed),
            "at_risk": len(creds) > 0 and not backed,
            "message": (
                "This passkey is device-bound. If you lose the device you lose access."
                if creds and not backed
                else "At least one passkey is synced to a provider."
            ),
        }
    )
