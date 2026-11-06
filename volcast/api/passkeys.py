"""WebAuthn / passkey ceremonies.

Design constraints (these are requirements, not incidental choices):
  * residentKey = REQUIRED, userVerification = REQUIRED
  * usernameless (discoverable) login: ``allowCredentials = []``
  * ``add_to_existing``: registering a second passkey must NOT create a new user
  * ``credential_backed_up`` (BE/BS flags) is surfaced to the UI
  * tolerant of py_webauthn builds that lack ``exclude_credentials``

BUG #12 lives here: older py_webauthn releases reject an ``exclude_credentials``
keyword at construction time, which made registration 500 on those versions.
"""

from __future__ import annotations

import base64
import os
import secrets
import uuid
from typing import Any

import webauthn
from webauthn.helpers import base64url_to_bytes, bytes_to_base64url
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    PublicKeyCredentialDescriptor,
    ResidentKeyRequirement,
    UserVerificationRequirement,
)

from . import db as dbmod
from .settings import Settings


class PasskeyError(Exception):
    """Domain error -> mapped to a 400 by the route layer."""


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _as_str(value: Any, default: str) -> str:
    """Enum or str -> plain str ("required", "none", ...)."""
    if value is None:
        return default
    raw = getattr(value, "value", value)
    text = str(raw)
    return text.split(".")[-1].lower() if text else default


def unb64url(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _new_challenge() -> bytes:
    return os.urandom(32)


# --------------------------------------------------------------------------- #
# Challenge persistence (options must round-trip; the client only echoes an id)
# --------------------------------------------------------------------------- #
def store_challenge(conn, cfg: Settings, kind: str, data: dict, user_id: str | None) -> str:
    cid = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO challenges (id, user_id, kind, data, created_at) VALUES (?,?,?,?,?)",
        (cid, user_id, kind, dbmod.dumps(data), dbmod.now()),
    )
    # Challenges are single-use and short-lived; prune aggressively.
    conn.execute("DELETE FROM challenges WHERE created_at < ?", (dbmod.now() - 300,))
    return cid


def pop_challenge(conn, cid: str, kind: str) -> dict:
    row = dbmod.get(conn, "SELECT * FROM challenges WHERE id=?", (cid,))
    if not row:
        raise PasskeyError("unknown or expired challenge")
    conn.execute("DELETE FROM challenges WHERE id=?", (cid,))
    if row["kind"] != kind:
        raise PasskeyError("challenge type mismatch")
    if dbmod.now() - float(row["created_at"]) > 300:
        raise PasskeyError("challenge expired")
    data = dbmod.loads(row["data"]) or {}
    data["_user_id"] = row["user_id"]
    return data


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def begin_registration(cfg: Settings, conn, username: str | None,
                       display_name: str | None, add_to_existing: str | None) -> dict:
    """Build PublicKeyCredentialCreationOptions.

    ``add_to_existing`` is the id of an already-authenticated user; when present we
    reuse it so a second passkey does not fork the account (this was a real bug:
    every registration minted a fresh user id, so "add another device" silently
    produced an orphan account with no credentials of its own).
    """
    if add_to_existing:
        row = dbmod.get(conn, "SELECT * FROM passkey_users WHERE id=?", (add_to_existing,))
        if not row:
            raise PasskeyError("cannot add a passkey to an unknown user")
        user_id = row["id"]
        uname = row["username"]
        dname = display_name or row["display_name"] or uname
    else:
        if username:
            existing = dbmod.get(
                conn, "SELECT * FROM passkey_users WHERE username=?", (username,)
            )
            if existing:
                # Existing username: attach to that account rather than duplicating.
                user_id = existing["id"]
                uname = existing["username"]
                dname = existing["display_name"] or username
            else:
                user_id = "usr_" + uuid.uuid4().hex
                uname = username
                dname = display_name or username
        else:
            # Usernameless: mint an opaque, non-guessable handle.
            user_id = "usr_" + uuid.uuid4().hex
            uname = "user-" + secrets.token_hex(4)
            if conn.execute(
                "SELECT 1 FROM passkey_users WHERE username=?", (uname,)
            ).fetchone():
                uname = "user-" + secrets.token_hex(8)
            dname = display_name or uname

        conn.execute(
            "INSERT OR IGNORE INTO passkey_users (id, username, display_name, created_at)"
            " VALUES (?,?,?,?)",
            (user_id, uname, dname, dbmod.now()),
        )

    challenge = _new_challenge()

    exclude = [
        PublicKeyCredentialDescriptor(id=unb64url(r["id"]))
        for r in dbmod.all_rows(
            conn, "SELECT id FROM credentials WHERE user_id=?", (user_id,)
        )
    ]

    kwargs: dict[str, Any] = {
        "rp_id": cfg.rp_id,
        "rp_name": cfg.rp_name,
        "user_id": user_id.encode("utf-8"),
        "user_name": uname,
        "user_display_name": dname,
        "timeout": 120_000,
        "challenge": challenge,
        "attestation": "none",
        "authenticator_selection": AuthenticatorSelectionCriteria(
            resident_key=ResidentKeyRequirement.REQUIRED,
            user_verification=UserVerificationRequirement.REQUIRED,
            authenticator_attachment=None,   # platform + roaming both allowed
            require_resident_key=True,
        ),
    }

    # BUG #12 FIX: py_webauthn < 2.x (and some vendor builds) have no
    # ``exclude_credentials`` parameter and raise TypeError on the kwarg.
    try:
        options = webauthn.generate_registration_options(
            exclude_credentials=exclude, **kwargs
        )
    except TypeError:
        options = webauthn.generate_registration_options(**kwargs)

    # The browser API only speaks camelCase. py_webauthn serialises snake_case
    # dataclasses, so the options are rebuilt explicitly here instead of dumping
    # ``__dict__`` (doing the latter produced ``authenticator_selection`` and
    # ``pub_key_cred_params`` keys that ``navigator.credentials.create`` ignores,
    # silently falling back to a non-resident, uv=preferred credential).
    public_key: dict[str, Any] = {
        "challenge": bytes_to_base64url(challenge),
        "rp": {"id": cfg.rp_id, "name": cfg.rp_name},
        "user": {
            "id": bytes_to_base64url(user_id.encode("utf-8")),
            "name": uname,
            "displayName": dname,
        },
        "pubKeyCredParams": [
            {"type": "public-key", "alg": int(p.alg)} for p in options.pub_key_cred_params
        ],
        "timeout": int(options.timeout or 120_000),
        "attestation": _as_str(options.attestation, "none"),
        "authenticatorSelection": {
            "residentKey": "required",
            "requireResidentKey": True,
            "userVerification": "required",
        },
        "excludeCredentials": [
            {"id": bytes_to_base64url(c.id), "type": "public-key"} for c in exclude
        ],
    }

    cid = store_challenge(
        conn,
        cfg,
        "register",
        {
            "challenge": bytes_to_base64url(challenge),
            "user_id": user_id,
            "exclude": [bytes_to_base64url(unb64url(r["id"])) for r in dbmod.all_rows(
                conn, "SELECT id FROM credentials WHERE user_id=?", (user_id,))],
        },
        user_id,
    )
    conn.commit()

    return {"challenge_id": cid, "public_key": public_key, "user_id": user_id}


def finish_registration(cfg: Settings, conn, challenge_id: str,
                        credential: dict, label: str | None) -> dict:
    state = pop_challenge(conn, challenge_id, "register")
    expected = state["challenge"]

    try:
        verification = webauthn.verify_registration_response(
            credential=credential,
            expected_challenge=base64url_to_bytes(expected),
            expected_rp_id=cfg.rp_id,
            expected_origin=cfg.origins,
            require_user_verification=True,
        )
    except Exception as exc:
        conn.commit()
        raise PasskeyError(f"registration verification failed: {exc}") from exc

    user_id = state.get("_user_id") or state.get("user_id")
    if not user_id:
        raise PasskeyError("missing user for registration")

    cred_id = bytes_to_base64url(verification.credential_id)
    transports = credential.get("response", {}).get("transports") or credential.get("transports") or []
    backed_up = bool(getattr(verification, "credential_backed_up", False))

    conn.execute(
        "INSERT OR REPLACE INTO credentials"
        " (id, user_id, public_key, sign_count, transports, backed_up, aaguid, created_at, last_used_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (
            cred_id,
            user_id,
            verification.credential_public_key,
            int(verification.sign_count or 0),
            ",".join(str(t) for t in transports),
            1 if backed_up else 0,
            str(getattr(verification, "aaguid", "") or ""),
            dbmod.now(),
            dbmod.now(),
        ),
    )
    conn.commit()

    row = dbmod.get(conn, "SELECT * FROM passkey_users WHERE id=?", (user_id,))
    return {
        "subject": user_id,
        "display_name": row["display_name"] if row else None,
        "credential_id": cred_id,
        "backed_up": backed_up,
        "transports": list(transports),
        "label": label,
    }


# --------------------------------------------------------------------------- #
# Authentication (usernameless by default)
# --------------------------------------------------------------------------- #
def begin_authentication(cfg: Settings, conn, username: str | None) -> dict:
    """Build PublicKeyCredentialRequestOptions.

    Usernameless login sends ``allowCredentials = []`` so the authenticator
    (and the browser's conditional-UI autofill) can offer any discoverable
    credential for this rp_id.
    """
    allow: list[PublicKeyCredentialDescriptor] = []
    if username:
        row = dbmod.get(conn, "SELECT id FROM passkey_users WHERE username=?", (username,))
        if not row:
            raise PasskeyError("no passkey registered for that username")
        allow = [
            PublicKeyCredentialDescriptor(id=unb64url(r["id"]))
            for r in dbmod.all_rows(
                conn, "SELECT id FROM credentials WHERE user_id=?", (row["id"],)
            )
        ]

    challenge = _new_challenge()
    options = webauthn.generate_authentication_options(
        rp_id=cfg.rp_id,
        challenge=challenge,
        allow_credentials=allow,
        user_verification=UserVerificationRequirement.REQUIRED,
        timeout=120_000,
    )

    public_key = {
        "challenge": bytes_to_base64url(challenge),
        "rpId": cfg.rp_id,
        "timeout": int(options.timeout or 120_000),
        "userVerification": "required",
        # Usernameless: an empty list lets the authenticator offer any
        # discoverable credential for this rp_id, and is what conditional UI needs.
        "allowCredentials": [
            {"id": bytes_to_base64url(c.id), "type": "public-key"}
            for c in (options.allow_credentials or [])
        ],
    }

    cid = store_challenge(
        conn,
        cfg,
        "authenticate",
        {"challenge": bytes_to_base64url(challenge), "username": username},
        None,
    )
    conn.commit()
    return {"challenge_id": cid, "public_key": public_key}


def finish_authentication(cfg: Settings, conn, challenge_id: str, credential: dict) -> dict:
    state = pop_challenge(conn, challenge_id, "authenticate")
    expected = state["challenge"]

    raw_id = credential.get("rawId") or credential.get("id")
    if not raw_id:
        raise PasskeyError("credential is missing an id")

    row = dbmod.get(conn, "SELECT * FROM credentials WHERE id=?", (raw_id,))
    if not row:
        conn.commit()
        raise PasskeyError("credential is not registered")

    try:
        verification = webauthn.verify_authentication_response(
            credential=credential,
            expected_challenge=base64url_to_bytes(expected),
            expected_rp_id=cfg.rp_id,
            expected_origin=cfg.origins,
            credential_public_key=row["public_key"],
            credential_current_sign_count=int(row["sign_count"] or 0),
            require_user_verification=True,
        )
    except Exception as exc:
        conn.commit()
        raise PasskeyError(f"authentication verification failed: {exc}") from exc

    conn.execute(
        "UPDATE credentials SET sign_count=?, last_used_at=?, backed_up=? WHERE id=?",
        (
            int(verification.new_sign_count or 0),
            dbmod.now(),
            1 if bool(getattr(verification, "credential_backed_up", False)) else 0,
            row["id"],
        ),
    )
    conn.commit()

    user = dbmod.get(conn, "SELECT * FROM passkey_users WHERE id=?", (row["user_id"],))
    return {
        "subject": row["user_id"],
        "display_name": user["display_name"] if user else None,
        "credential_id": row["id"],
        "backed_up": bool(getattr(verification, "credential_backed_up", False)),
    }


def list_credentials(conn, user_id: str) -> list[dict]:
    rows = dbmod.all_rows(
        conn, "SELECT * FROM credentials WHERE user_id=? ORDER BY created_at", (user_id,)
    )
    out = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "label": None,
                "created_at": float(r["created_at"]),
                "last_used_at": float(r["last_used_at"]) if r["last_used_at"] else None,
                "backed_up": bool(r["backed_up"]),
                "transports": [t for t in (r["transports"] or "").split(",") if t],
            }
        )
    return out


def delete_credential(conn, user_id: str, cred_id: str) -> bool:
    cur = conn.execute(
        "DELETE FROM credentials WHERE id=? AND user_id=?", (cred_id, user_id)
    )
    conn.commit()
    return bool(cur.rowcount)


def count_credentials(conn, user_id: str) -> int:
    row = dbmod.get(
        conn, "SELECT COUNT(*) AS n FROM credentials WHERE user_id=?", (user_id,)
    )
    return int(row["n"]) if row else 0
