from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from volcast.api import db as dbmod
from volcast.api import passkeys
from volcast.api.settings import Settings


@pytest.fixture
def conn():
    td = tempfile.mkdtemp()
    path = Path(td) / "t.db"
    dbmod.init_db(path)
    c = dbmod.connect(path)
    yield c
    c.close()


def _cfg() -> Settings:
    return Settings(rp_id="localhost", expected_origins="http://localhost:5173")


# --------------------------------------------------------------------------- #
# Regression: begin_registration must NOT create the user row
# --------------------------------------------------------------------------- #
def test_begin_registration_does_not_create_user(conn):
    """Abandoned registrations used to leave credential-less accounts behind."""
    before = conn.execute("SELECT COUNT(*) FROM passkey_users").fetchone()[0]
    out = passkeys.begin_registration(_cfg(), conn, "newperson", "New Person", None)
    after = conn.execute("SELECT COUNT(*) FROM passkey_users").fetchone()[0]

    assert out["user_id"].startswith("usr_")
    assert after == before, f"begin_registration created {after - before} orphan row(s)"


def test_begin_registration_still_returns_usable_options(conn):
    out = passkeys.begin_registration(_cfg(), conn, "someone", "Someone", None)
    pk = out["public_key"]
    assert pk["challenge"]
    assert pk["user"]["name"] == "someone"
    assert pk["authenticatorSelection"]["residentKey"] == "required"
    assert pk["authenticatorSelection"]["userVerification"] == "required"


def test_begin_registration_marks_new_user_pending(conn):
    out = passkeys.begin_registration(_cfg(), conn, "pendingcheck", "P", None)
    row = dbmod.get(conn, "SELECT data FROM challenges WHERE id=?", (out["challenge_id"],))
    import json

    state = json.loads(row["data"])
    assert state["pending"] is True
    assert state["username"] == "pendingcheck"


def test_begin_registration_existing_user_is_not_pending(conn):
    conn.execute(
        "INSERT INTO passkey_users (id, username, display_name, created_at) VALUES (?,?,?,?)",
        ("usr_existing", "existing", "Existing", dbmod.now()),
    )
    conn.commit()
    out = passkeys.begin_registration(_cfg(), conn, "existing", None, None)
    row = dbmod.get(conn, "SELECT data FROM challenges WHERE id=?", (out["challenge_id"],))
    import json

    state = json.loads(row["data"])
    assert state["pending"] is False
    assert out["user_id"] == "usr_existing"


def test_add_to_existing_is_not_pending(conn):
    conn.execute(
        "INSERT INTO passkey_users (id, username, display_name, created_at) VALUES (?,?,?,?)",
        ("usr_owner", "owner", "Owner", dbmod.now()),
    )
    conn.commit()
    out = passkeys.begin_registration(_cfg(), conn, None, None, "usr_owner")
    assert out["user_id"] == "usr_owner"


# --------------------------------------------------------------------------- #
# RP ID must match the page origin
# --------------------------------------------------------------------------- #
class _FakeURL:
    scheme = "http"


class _FakeReq:
    def __init__(self, host: str):
        self.headers = {"host": host}
        self.url = _FakeURL()


def test_resolve_rp_maps_loopback_ip_to_localhost():
    """IP literals are not valid rp.id values; loopback must become 'localhost'."""
    rp_id, _, origins = passkeys.resolve_rp(_cfg(), _FakeReq("127.0.0.1:8000"))
    assert rp_id == "localhost", "an IP rp.id is rejected by browsers as an invalid domain"
    # The origin that is actually allowed must be the localhost one, because that
    # is the address the user is told to open.
    assert "http://localhost:8000" in origins


def test_resolve_rp_keeps_real_domain():
    rp_id, _, _ = passkeys.resolve_rp(_cfg(), _FakeReq("volcast.example.com"))
    assert rp_id == "volcast.example.com"


def test_resolve_rp_keeps_explicit_non_placeholder_config():
    cfg = Settings(rp_id="example.com", expected_origins="https://example.com")
    rp_id, _, _ = passkeys.resolve_rp(cfg, _FakeReq("example.com"))
    assert rp_id == "example.com"


def test_resolve_rp_without_request_falls_back():
    rp_id, _, _ = passkeys.resolve_rp(_cfg(), None)
    assert rp_id == "localhost"


def test_register_options_rp_is_never_an_ip(conn):
    out = passkeys.begin_registration(
        _cfg(), conn, "hostcheck", None, None, _FakeReq("127.0.0.1:8000")
    )
    rp_id = out["public_key"]["rp"]["id"]
    assert rp_id == "localhost"
    assert not rp_id.replace(".", "").isdigit(), "rp.id must be a domain, not an IP"


# --------------------------------------------------------------------------- #
# Orphan pruning
# --------------------------------------------------------------------------- #
def test_prune_selector_finds_only_credentialless_users(tmp_path):
    dbp = tmp_path / "p.db"
    dbmod.init_db(dbp)
    c = sqlite3.connect(str(dbp))
    c.execute("INSERT INTO passkey_users VALUES ('usr_orphan','orphan','O',0)")
    c.execute("INSERT INTO passkey_users VALUES ('usr_real','real','R',0)")
    c.executemany(
        "INSERT INTO credentials (id,user_id,public_key,sign_count,transports,backed_up,aaguid,"
        "created_at,last_used_at) VALUES (?,?,?,?,?,?,?,?,?)",
        [("cred1", "usr_real", b"k", 0, "", 0, "", 0, 0)],
    )
    c.commit()
    c.close()

    rows = c2 = None
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "prune", Path(__file__).resolve().parents[1] / "tools" / "prune_orphans.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    c2 = sqlite3.connect(str(dbp))
    found = mod.orphans(c2)
    c2.close()
    assert rows is None
    names = [n for _, n, _ in found]
    assert "orphan" in names
    assert "real" not in names
