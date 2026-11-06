"""SQLite persistence for devices, passkey users, credentials and refresh tokens."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    id            TEXT PRIMARY KEY,
    label         TEXT,
    created_at    REAL NOT NULL,
    last_seen_at  REAL
);

CREATE TABLE IF NOT EXISTS passkey_users (
    id            TEXT PRIMARY KEY,
    username      TEXT UNIQUE,
    display_name  TEXT,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS credentials (
    id              TEXT PRIMARY KEY,          -- base64url credential id
    user_id         TEXT NOT NULL,
    public_key      BLOB NOT NULL,
    sign_count      INTEGER NOT NULL DEFAULT 0,
    transports      TEXT,
    backed_up       INTEGER NOT NULL DEFAULT 0,
    aaguid          TEXT,
    created_at      REAL NOT NULL,
    last_used_at    REAL
);
CREATE INDEX IF NOT EXISTS idx_cred_user ON credentials(user_id);

CREATE TABLE IF NOT EXISTS challenges (
    id          TEXT PRIMARY KEY,               -- opaque challenge handle
    user_id     TEXT,
    kind        TEXT NOT NULL,                  -- register | authenticate
    data        TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS refresh_tokens (
    id          TEXT PRIMARY KEY,               -- jti
    subject     TEXT NOT NULL,
    family      TEXT NOT NULL,
    expires_at  REAL NOT NULL,
    revoked     INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rt_subject ON refresh_tokens(subject);

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    payload     TEXT,
    result      TEXT,
    error       TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
"""


def _db_path(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path(path)), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(path: str | Path) -> None:
    with closing_conn(path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@contextmanager
def closing_conn(path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()


def now() -> float:
    return time.time()


def get(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> sqlite3.Row | None:
    return conn.execute(sql, params).fetchone()


def all_rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    return list(conn.execute(sql, params).fetchall())


def dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str)


def loads(raw: str | None) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None
