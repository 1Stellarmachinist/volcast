#!/usr/bin/env python
"""Self-verification harness for the volcast project.

Checks each of the 14 previously-identified bugs against the *live* codebase
rather than trusting comments. Exits non-zero if any check fails.

Usage:
    .venv\\Scripts\\python tools\\verify.py
    .venv\\Scripts\\python tools\\verify.py --verbose
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"


class Report:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, fn) -> bool:
        try:
            detail = fn()
            ok = True
        except AssertionError as exc:
            ok, detail = False, f"{exc}"
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        self.results.append((name, ok, str(detail or "")))
        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        line = f"  [{mark}] {name}"
        if detail and (not ok or ARGS.verbose):
            line += f"\n         {DIM}{detail}{RESET}"
        print(line)
        return ok

    def summary(self) -> int:
        total = len(self.results)
        passed = sum(1 for _, ok, _ in self.results if ok)
        failed = total - passed
        print()
        print("=" * 74)
        print(f"{BOLD}BUG CHECK SUMMARY{RESET}: {passed}/{total} passed"
              + (f", {RED}{failed} failed{RESET}" if failed else f", {GREEN}0 failed{RESET}"))
        if failed:
            for name, ok, detail in self.results:
                if not ok:
                    print(f"  {RED}x{RESET} {name} -> {detail}")
        print("=" * 74)
        return 0 if failed == 0 else 1


REPORT = Report()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def src(rel: str) -> str:
    p = ROOT / rel
    assert p.is_file(), f"missing file: {rel}"
    return p.read_text(encoding="utf-8")


def func_src(module: str, name: str) -> str:
    mod = importlib.import_module(module)
    fn = getattr(mod, name)
    return inspect.getsource(fn)


def strip_noise(text: str) -> list[str]:
    """Executable code lines only (comments/docstrings removed)."""
    out: list[str] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
    for node in ast.walk(tree):
        for sub in (ast.Expr,):
            if isinstance(node, sub) and isinstance(getattr(node, "value", None), ast.Constant):
                node.value.value = ""
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(line)
    return out


def expect_import(module: str, label: str | None = None) -> str:
    spec = importlib.util.find_spec(module)
    assert spec is not None, f"{module} is not installed"
    return label or module


def http_get(app, path: str, **kwargs):
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        return client.get(path, **kwargs)


# --------------------------------------------------------------------------- #
# Bug 1: drift_forecast returns finite positive sigma_h
# --------------------------------------------------------------------------- #
def check_bug_1() -> str:
    import numpy as np
    import pandas as pd

    from volcast.forecast import drift_forecast, forecast_panel

    # Empty frame
    fc = drift_forecast(pd.DataFrame(), horizon=5)
    assert fc == [], "empty input should yield no forecasts"

    # Single-column frame with fewer than 2 observations -> std is undefined
    tiny = pd.DataFrame({"X": [0.01]})
    f = drift_forecast(tiny, horizon=5)[0]
    assert math.isfinite(f.sigma_h), f"sigma_h NaN for 1 obs: {f.sigma_h}"
    assert f.sigma_h > 0.0, f"sigma_h not positive: {f.sigma_h}"

    # All-NaN column
    nan_col = pd.DataFrame({"Y": [np.nan] * 40})
    g = drift_forecast(nan_col, horizon=3)[0]
    assert math.isfinite(g.sigma_h) and g.sigma_h > 0.0, f"sigma_h bad: {g.sigma_h}"

    # Normal case: sigma_h == sigma * sqrt(h)
    rng = np.random.default_rng(2)
    rets = pd.DataFrame({"A": rng.normal(0, 0.01, 100), "B": rng.normal(0, 0.02, 100)})
    table = forecast_panel(rets, horizon=5, window=20, engine="drift")
    assert np.isfinite(table[["sigma", "sigma_h"]].to_numpy()).all()
    assert (table["sigma_h"] > 0).all(), "non-positive sigma_h in forecast table"
    row = table[table.ticker == "A"].iloc[0]
    assert row["sigma_h"] == pytest_approx(row["sigma"] * math.sqrt(5), 1e-12)

    # And the sizer must not flatten the book
    from volcast.signals import cross_sectional_weights
    from volcast.sizing import target_positions

    w = cross_sectional_weights(pd.Series({"A": 1.0, "B": -1.0}))
    sized = target_positions(
        w, table.set_index("ticker")["sigma_h"], target_vol=0.15, horizon=5
    )
    assert sized.abs().sum() > 0, "sizer produced a zero book from valid forecasts"
    return f"sigma_h finite & >0 for 1-obs/NaN inputs; book gross={sized.abs().sum():.4f}"


def pytest_approx(v, tol):
    return _Approx(v, tol)


class _Approx:
    def __init__(self, v, tol):
        self.v, self.tol = v, tol

    def __eq__(self, other):
        return abs(float(other) - self.v) <= self.tol

    def __repr__(self):
        return f"approx({self.v})"


# --------------------------------------------------------------------------- #
# Bug 2: _clean sanitiser applied to every payload
# --------------------------------------------------------------------------- #
def check_bug_2() -> str:
    import numpy as np

    from volcast.api.service import _clean

    payload = {
        "a": float("nan"),
        "b": float("inf"),
        "c": float("-inf"),
        "d": [1.0, float("nan"), {"nested": float("inf")}],
        "e": np.float64("nan"),
        "f": np.array([1.0, np.nan]),
        "g": {"deep": {"deeper": [float("-inf"), 2.0]}},
        "h": 3.5,
        "i": None,
        "j": True,
    }
    out = _clean(payload)
    text = json.dumps(out, allow_nan=False)   # raises if anything non-finite survives
    assert "NaN" not in text and "Infinity" not in text, "sanitiser leaked a non-finite literal"
    assert out["a"] is None and out["b"] is None and out["c"] is None
    assert out["d"][1] is None and out["d"][2]["nested"] is None
    assert out["f"] == [1.0, None]
    assert out["h"] == 3.5

    # Every market payload must be NaN-free on the wire.
    from fastapi.testclient import TestClient

    from volcast.api.main import create_app

    app = create_app()
    with TestClient(app) as c:
        for path in ("/api/market/forecast?tickers=AAA,BBB&horizon=3",
                     "/api/market/weights?tickers=AAA,BBB",
                     "/api/market/screen?tickers=AAA,BBB,CCC",
                     "/api/health"):
            r = c.get(path)
            assert r.status_code == 200, f"{path} -> {r.status_code} {r.text[:200]}"
            r.json()  # must parse
            assert "NaN" not in r.text, f"{path} emitted NaN literal"

    # SafeJSONResponse must refuse NaN outright.
    from volcast.api.main import SafeJSONResponse

    resp = SafeJSONResponse(content={"x": float("nan"), "y": [float("inf")]})
    body = resp.body.decode()
    assert "NaN" not in body and "Infinity" not in body, body
    return "recursive sanitiser + SafeJSONResponse; 4 endpoints NaN-free and parseable"


# --------------------------------------------------------------------------- #
# Bug 3: security.decode_access_claims(cfg, token) -> dict
# --------------------------------------------------------------------------- #
def check_bug_3() -> str:
    from volcast.api import security
    from volcast.api.settings import Settings

    fn = getattr(security, "decode_access_claims", None)
    assert fn is not None, "decode_access_claims is missing"

    sig = inspect.signature(fn)
    params = list(sig.parameters)
    assert len(params) == 2, f"expected 2 parameters (cfg, token), got {params}"

    cfg = Settings(jwt_secret="test-secret", access_ttl_seconds=60)
    token = security.create_access_token(cfg, "usr_abc", scope="read")
    claims = fn(cfg, token)
    assert isinstance(claims, dict), f"must return dict, got {type(claims)}"
    assert claims["sub"] == "usr_abc"
    assert claims["typ"] == "access"
    assert claims["scope"] == "read"

    # read_access_token must delegate to it
    read_src = inspect.getsource(security.read_access_token)
    assert "decode_access_claims" in read_src, "read_access_token does not wrap decode_access_claims"

    # Bad tokens raise ValueError (not HTTPException) so it is unit-testable
    for bad in ("", "not-a-jwt", token + "x"):
        try:
            fn(cfg, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for token {bad!r}")

    # Expired token
    import jwt

    expired = jwt.encode(
        {"sub": "u", "typ": "access", "exp": 1, "iat": 1}, cfg.jwt_secret,
        algorithm=cfg.jwt_alg,
    )
    try:
        fn(cfg, expired)
    except ValueError as exc:
        assert "expired" in str(exc).lower(), str(exc)
    else:
        raise AssertionError("expired token accepted")
    return "signature (cfg, token) -> dict; validates type/exp; called by read_access_token"


# --------------------------------------------------------------------------- #
# Bug 4: refresh accepts passkey subjects via _known_subject
# --------------------------------------------------------------------------- #
def check_bug_4() -> str:
    from volcast.api import db as dbmod
    from volcast.api import security
    from volcast.api.settings import Settings

    helper = getattr(security, "_known_subject", None)
    assert helper is not None, "_known_subject helper is missing"

    with tempfile.TemporaryDirectory() as td:
        dbp = Path(td) / "t.db"
        dbmod.init_db(dbp)
        cfg = Settings(jwt_secret="s3cr3t")
        with dbmod.closing_conn(dbp) as conn:
            conn.execute(
                "INSERT INTO passkey_users (id, username, display_name, created_at) VALUES (?,?,?,?)",
                ("usr_passkey", "dan", "Daniel", dbmod.now()),
            )
            conn.execute(
                "INSERT INTO devices (id, label, created_at, last_seen_at) VALUES (?,?,?,?)",
                ("dev_1", "phone", dbmod.now(), dbmod.now()),
            )
            conn.commit()

            assert helper(conn, "usr_passkey") is True, "passkey subject rejected"
            assert helper(conn, "dev_1") is True, "device subject rejected"
            assert helper(conn, "usr_nope") is False
            assert helper(conn, "") is False

            # End-to-end: a passkey user's refresh token must rotate.
            _tok, payload = security.create_refresh_token(cfg, "usr_passkey")
            security.store_refresh(conn, cfg, payload)
            conn.commit()
            new_tok, new_payload = security.rotate_refresh(conn, cfg, payload)
            assert new_tok and new_payload["fam"] == payload["fam"]

            # Reuse of the old token must fail and revoke the family.
            try:
                security.rotate_refresh(conn, cfg, payload)
            except ValueError:
                pass
            else:
                raise AssertionError("refresh token reuse was accepted")

            row = dbmod.get(conn, "SELECT revoked FROM refresh_tokens WHERE id=?",
                            (new_payload["jti"],))
            assert row is not None and int(row["revoked"]) == 1, "family was not revoked on reuse"
    return "passkey AND device subjects accepted; rotation works; reuse revokes the family"


# --------------------------------------------------------------------------- #
# Bug 5: refresh(body, request, response) 3-arg + refresh_cookie reads cookie
# --------------------------------------------------------------------------- #
def check_bug_5() -> str:
    from volcast.api.routes_auth import refresh, refresh_cookie

    params = list(inspect.signature(refresh).parameters)
    assert len(params) == 3, f"refresh must take exactly 3 args, got {params}"
    assert params == ["body", "request", "response"], f"unexpected signature: {params}"

    cookie_params = list(inspect.signature(refresh_cookie).parameters)
    assert "request" in cookie_params, f"refresh_cookie cannot see the request: {cookie_params}"

    body = inspect.getsource(refresh_cookie)
    assert "cookies.get" in body, "refresh_cookie never reads a cookie"
    assert "volcast_rt" in body or "cfg.refresh_cookie" in body, \
        "refresh_cookie does not read the volcast_rt cookie"

    # Live behaviour: a real refresh round-trip through the cookie route, against
    # a throwaway database so the dev DB is never touched.

    from fastapi.testclient import TestClient

    from volcast.api import db as dbmod
    from volcast.api import security
    from volcast.api.settings import get_settings

    with tempfile.TemporaryDirectory() as td:
        dbp = Path(td) / "rt.db"
        dbmod.init_db(dbp)

        previous = os.environ.get("VOLCAST_DB_PATH")
        os.environ["VOLCAST_DB_PATH"] = str(dbp)
        get_settings.cache_clear()
        try:
            cfg = get_settings()
            assert Path(cfg.db_path) == dbp

            conn = dbmod.connect(dbp)
            conn.execute(
                "INSERT INTO passkey_users (id, username, display_name, created_at)"
                " VALUES (?,?,?,?)",
                ("usr_x", "x", "X", dbmod.now()),
            )
            tok, payload = security.create_refresh_token(cfg, "usr_x")
            security.store_refresh(conn, cfg, payload)
            conn.commit()
            conn.close()

            from volcast.api.main import create_app

            app = create_app()
            with TestClient(app) as c:
                r = c.post("/api/auth/refresh/cookie")
                assert r.status_code == 401, f"expected 401 with no cookie, got {r.status_code}"

                r = c.post("/api/auth/refresh/cookie", cookies={cfg.refresh_cookie: tok})
                assert r.status_code == 200, \
                    f"cookie refresh failed: {r.status_code} {r.text[:300]}"
                assert r.json()["access_token"], "no access token returned"

                rotated = r.cookies.get(cfg.refresh_cookie)
                assert rotated, "refresh cookie was not rotated"
                assert rotated != tok, "refresh token was not actually rotated"

                # Use a fresh client for follow-up calls: TestClient keeps a cookie
                # jar, and a jar-supplied cookie would shadow the explicit one, so
                # the reuse-detection check would test the wrong token.
                with TestClient(app) as c2:
                    r2 = c2.post("/api/auth/refresh/cookie",
                                 cookies={cfg.refresh_cookie: rotated})
                    assert r2.status_code == 200, \
                        f"rotated token unusable: {r2.status_code} {r2.text[:200]}"

                # The ORIGINAL token must now be rejected (reuse detection).
                with TestClient(app) as c3:
                    r3 = c3.post("/api/auth/refresh/cookie", cookies={cfg.refresh_cookie: tok})
                    assert r3.status_code == 401, f"reused token accepted: {r3.status_code}"
        finally:
            if previous is None:
                os.environ.pop("VOLCAST_DB_PATH", None)
            else:
                os.environ["VOLCAST_DB_PATH"] = previous
            get_settings.cache_clear()

    return "refresh(body, request, response); cookie round-trip OK; rotation + reuse detection"


# --------------------------------------------------------------------------- #
# Bug 6: PKCE handoff uses the *incoming* code_challenge
# --------------------------------------------------------------------------- #
def check_bug_6() -> str:
    candidates = [
        ROOT / "web" / "src" / "pages" / "SignIn.tsx",
        ROOT / "web" / "src" / "pages" / "SignIn.jsx",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    assert path is not None, "web/src/pages/SignIn.(tsx|jsx) is missing"
    text = path.read_text(encoding="utf-8")

    assert "code_challenge" in text, "SignIn never reads code_challenge from the query string"
    assert re.search(r"searchParams|URLSearchParams|useSearchParams", text), \
        "SignIn does not parse the query string"
    # It must NOT mint its own challenge.
    assert not re.search(r"generateCodeChallenge|createCodeChallenge|sha256\(.*verifier", text), \
        "SignIn generates its own PKCE challenge instead of using the incoming one"

    # The native helper must produce S256 only.
    helper = (ROOT / "web" / "src" / "auth" / "pkce.ts")
    assert helper.is_file(), "web/src/auth/pkce.ts is missing"
    htext = helper.read_text(encoding="utf-8")
    assert "S256" in htext, "pkce.ts does not declare S256"
    assert not re.search(r"code_challenge_method\s*[:=]\s*[\"']plain[\"']", htext), \
        "pkce.ts allows the 'plain' method"
    return "SignIn consumes incoming code_challenge (never generates one); pkce.ts is S256-only"


# --------------------------------------------------------------------------- #
# Bug 7: webauthn.ts post(path, body, token?)
# --------------------------------------------------------------------------- #
def check_bug_7() -> str:
    p = ROOT / "web" / "src" / "auth" / "webauthn.ts"
    assert p.is_file(), "web/src/auth/webauthn.ts is missing"
    text = p.read_text(encoding="utf-8")

    m = re.search(
        r"export\s+(?:async\s+)?function\s+post\s*(?:<[^>]*>)?\s*\(([^)]*)\)", text
    )
    assert m, "no `post(...)` export found in webauthn.ts"
    args = [a.strip() for a in m.group(1).split(",") if a.strip()]
    assert len(args) == 3, f"post must take 3 parameters (path, body, token?), got {args}"
    assert args[0].startswith("path"), f"first arg should be path, got {args[0]!r}"
    assert args[1].startswith("body"), f"second arg should be body, got {args[1]!r}"
    assert args[2].startswith("token"), f"third arg should be token, got {args[2]!r}"
    assert "token?" in m.group(1).replace(" ", "") or "token =" in m.group(1), \
        "token must be optional"

    # The body must actually be sent as JSON, not as an Authorization header.
    post_src = text[m.start(): text.find("export async function get", m.start())]
    assert "JSON.stringify(body)" in post_src, "post() does not serialise its body"
    assert "Authorization" in post_src, "post() never forwards the bearer token"
    return f"post({', '.join(args)}) with JSON body + optional bearer token"


# --------------------------------------------------------------------------- #
# Bug 8: cookie_secure defaults to is_prod
# --------------------------------------------------------------------------- #
def check_bug_8() -> str:
    from volcast.api.settings import Settings

    dev = Settings(env="dev", cookie_secure=None)
    prod = Settings(env="prod", cookie_secure=None)
    forced = Settings(env="dev", cookie_secure=True)

    assert dev.secure_cookies is False, "dev must not set Secure (http://localhost)"
    assert prod.secure_cookies is True, "prod must set Secure"
    assert forced.secure_cookies is True, "explicit override ignored"

    src_text = src("volcast/api/settings.py")
    assert "is_prod" in src_text, "settings does not reference is_prod"
    assert not re.search(r"cookie_secure\s*:\s*bool\s*=\s*True", src_text), \
        "cookie_secure is still hardcoded True"

    # Cookies set in dev must omit the Secure flag.
    from fastapi import Response

    from volcast.api import security

    r = Response()
    security.set_auth_cookies(dev, r, "a", "b")
    header = " ".join(r.headers.getlist("set-cookie"))
    assert "Secure" not in header, f"dev cookie unexpectedly Secure: {header}"
    return "dev -> insecure cookie, prod -> Secure, override honoured"


# --------------------------------------------------------------------------- #
# Bug 9: metrics() -> NaN Sharpe on zero variance
# --------------------------------------------------------------------------- #
def check_bug_9() -> str:
    import pandas as pd

    from volcast.evaluate import metrics

    m = metrics(pd.Series([0.0] * 200))
    assert math.isnan(m["sharpe"]), f"zero-variance Sharpe must be NaN, got {m['sharpe']}"
    assert math.isnan(m["sharpe_se"])

    flat = metrics(pd.Series([0.002] * 200))
    assert math.isnan(flat["sharpe"]), f"constant-return Sharpe must be NaN, got {flat['sharpe']}"

    # The test suite must assert NaN, not `sharpe > 5`.
    test_src = src("tests/test_backtest.py")
    assert "isnan" in test_src, "tests never assert math.isnan for the Sharpe"
    assert not re.search(r"sharpe\s*>\s*5", test_src), \
        "a test still asserts `sharpe > 5`"

    import pandas as pd

    rng_src = src("volcast/evaluate.py")
    assert "1e-15" in rng_src or "<= 0" in rng_src, "no zero-variance guard in metrics()"
    del pd
    return "zero-variance -> NaN sharpe and NaN SE; tests assert isnan (no `sharpe > 5`)"


# --------------------------------------------------------------------------- #
# Bug 10: pinball minimised at the claimed quantile
# --------------------------------------------------------------------------- #
def check_bug_10() -> str:
    import numpy as np

    from volcast.evaluate import mean_pinball, pinball

    assert pinball(0.01, 0.01, 0.9) == 0, f"perfect forecast must be 0, got {pinball(0.01,0.01,0.9)}"
    got = pinball(0.03, 0.01, 0.9)
    assert got == _Approx(0.9 * 0.02, 1e-15), f"expected {0.9*0.02}, got {got}"

    # Numerical optimality: argmin over a grid == sample quantile.
    rng = np.random.default_rng(42)
    y = rng.normal(0, 0.02, 6000)
    grid = np.linspace(-0.06, 0.06, 241)
    for q in (0.01, 0.1, 0.5, 0.9, 0.99):
        losses = [mean_pinball(y, np.full_like(y, f), q) for f in grid]
        best = grid[int(np.argmin(losses))]
        want = float(np.quantile(y, q))
        assert abs(best - want) < 3e-3, f"q={q}: argmin {best:.4f} != quantile {want:.4f}"
    return "pinball(0.01,0.01,0.9)==0; pinball(0.03,0.01,0.9)==0.9*0.02; argmin at quantile"


# --------------------------------------------------------------------------- #
# Bug 11: cross_sectional_weights re-demeans AFTER _cap
# --------------------------------------------------------------------------- #
def check_bug_11() -> str:
    import numpy as np
    import pandas as pd

    from volcast.signals import cross_sectional_weights

    cases = {
        "skewed": pd.Series({"A": 10.0, "B": 0.1, "C": -0.1, "D": -0.2}),
        "heavy_long": pd.Series({"A": 1.0, "B": 0.9, "C": 0.8, "D": -0.1}),
        "two_name": pd.Series({"A": 1.0, "B": -1.0}),
        "many": pd.Series(np.linspace(1, -1, 12), index=[f"T{i}" for i in range(12)]),
    }
    for name, s in cases.items():
        w = cross_sectional_weights(s, max_weight=0.25)
        assert abs(float(w.sum())) < 1e-9, f"{name}: book not dollar-neutral (net={w.sum():.3e})"
        assert float(w.abs().max()) <= 0.25 + 1e-12, \
            f"{name}: cap breached ({w.abs().max():.6f} > 0.25)"

    fn_src = inspect.getsource(cross_sectional_weights)
    # The demeaning must appear *after* the cap call, not only before it.
    cap_pos = fn_src.find("_cap(")
    assert cap_pos != -1, "cross_sectional_weights never calls _cap"
    assert re.search(r"mean\(\)", fn_src[cap_pos:]), \
        "no re-demeaning after _cap -> book can stay net-long"
    return "net==0 and |w|<=cap simultaneously for 4 books; demeaning occurs after _cap"


# --------------------------------------------------------------------------- #
# Bug 12: begin_registration tolerates missing exclude_credentials
# --------------------------------------------------------------------------- #
def check_bug_12() -> str:
    import webauthn

    from volcast.api import db as dbmod
    from volcast.api import passkeys
    from volcast.api.settings import Settings

    fn_src = inspect.getsource(passkeys.begin_registration)
    assert "except TypeError" in fn_src, "no TypeError fallback around the WebAuthn options builder"
    assert "exclude_credentials" in fn_src, "exclude_credentials is never passed at all"

    original = webauthn.generate_registration_options

    def strict(**kwargs):
        """Mimic a py_webauthn build that has no exclude_credentials parameter."""
        if "exclude_credentials" in kwargs:
            raise TypeError("generate_registration_options() got an unexpected keyword argument 'exclude_credentials'")
        return original(**kwargs)

    with tempfile.TemporaryDirectory() as td:
        dbp = Path(td) / "t.db"
        dbmod.init_db(dbp)
        cfg = Settings(rp_id="localhost", expected_origins="http://localhost:5173")
        with dbmod.closing_conn(dbp) as conn:
            webauthn.generate_registration_options = strict
            try:
                out = passkeys.begin_registration(cfg, conn, "dan", "Daniel", None)
            finally:
                webauthn.generate_registration_options = original
            pk = out["public_key"]
            assert pk.get("challenge"), "no challenge in options"
            rk = str(pk.get("authenticatorSelection", {}).get("residentKey", "")).lower()
            uv = str(pk.get("authenticatorSelection", {}).get("userVerification", "")).lower()
            assert "required" in rk, f"residentKey not required: {rk}"
            assert "required" in uv, f"userVerification not required: {uv}"
    return "registration succeeds even when exclude_credentials raises TypeError; rk/uv required"


# --------------------------------------------------------------------------- #
# Bug 13: no fillna(method="ffill") anywhere
# --------------------------------------------------------------------------- #
def check_bug_13() -> str:
    offenders: list[str] = []
    scanned = 0
    # Build the pattern from parts so this scanner never matches its own source
    # (a real self-referential bug: the literal appeared in the check itself).
    needle = re.compile(r"fillna\s*\(\s*method\s*=")
    pad = re.compile(r"\.pad\s*\(|fillna\s*\(\s*method\s*=\s*['\"]pad")
    for py in ROOT.rglob("*.py"):
        rel = py.relative_to(ROOT)
        if set(rel.parts) & {".venv", "node_modules", "build", "dist", "__pycache__", ".git"}:
            continue
        if rel.as_posix() == "tools/verify.py":
            continue  # this file necessarily mentions the pattern it bans
        scanned += 1
        text = py.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0]
            # Docstrings only *describe* the removed API; they cannot call it.
            stripped = code.strip()
            if stripped.startswith(('"""', "'''", '"', "'")) and "fillna(" not in stripped:
                continue
            if needle.search(code) or pad.search(code):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, "deprecated fillna(method=...) still present:\n    " + "\n    ".join(offenders)
    return f"scanned {scanned} python files; zero fillna(method=...) / .pad() calls"


# --------------------------------------------------------------------------- #
# Bug 14: pyproject [project.optional-dependencies].api
# --------------------------------------------------------------------------- #
def check_bug_14() -> str:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore

    with (ROOT / "pyproject.toml").open("rb") as fh:
        data = tomllib.load(fh)

    extras = data.get("project", {}).get("optional-dependencies", {})
    assert "api" in extras, f"no [project.optional-dependencies].api extra (have {list(extras)})"

    api = [str(d).lower() for d in extras["api"]]
    required = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "pydantic": "pydantic",
        "pydantic-settings": "pydantic-settings",
        "pyjwt": "pyjwt",
        "cryptography": "cryptography",
        "webauthn": "webauthn",
    }
    missing = [name for name, token in required.items() if not any(token in d for d in api)]
    assert not missing, f"api extra is missing: {missing}"

    webauthn_dep = next(d for d in api if d.startswith("webauthn"))
    assert ">=2.2" in webauthn_dep, f"webauthn pin must be >=2.2, got {webauthn_dep}"
    assert "<3" in webauthn_dep, f"webauthn pin must be <3, got {webauthn_dep}"
    assert any("standard" in d for d in api if d.startswith("uvicorn")), \
        "uvicorn must be uvicorn[standard]"

    # And torch/chronos must NOT be in the base or api extras.
    base = [str(d).lower() for d in data["project"].get("dependencies", [])]
    for banned in ("torch", "chronos"):
        assert not any(banned in d for d in base), f"{banned} in base dependencies"
        assert not any(banned in d for d in api), f"{banned} in the api extra"
    return "api extra has all 7 deps; webauthn>=2.2,<3; uvicorn[standard]; no torch/chronos"


# --------------------------------------------------------------------------- #
# Project structure + passkey requirements
# --------------------------------------------------------------------------- #
EXPECTED_FILES = """
volcast/__init__.py volcast/__main__.py volcast/cli.py volcast/config.py volcast/utils.py
volcast/data.py volcast/indicators.py volcast/panel.py volcast/forecast.py volcast/signals.py
volcast/sizing.py volcast/backtest.py volcast/evaluate.py volcast/finetune.py volcast/plot.py
volcast/api/__init__.py volcast/api/settings.py volcast/api/db.py volcast/api/security.py
volcast/api/schemas.py volcast/api/jobs.py volcast/api/service.py volcast/api/passkeys.py
volcast/api/routes_auth.py volcast/api/routes_market.py volcast/api/routes_passkey.py
volcast/api/main.py
volcast/commands/__init__.py volcast/commands/_args.py volcast/commands/forecast.py
volcast/commands/screen.py volcast/commands/strat.py volcast/commands/cs.py volcast/commands/ft.py
tools/verify.py tools/launch.py tools/make_shortcut.py tools/bench.py
tests/test_indicators.py tests/test_signals.py tests/test_sizing.py tests/test_backtest.py
pyproject.toml README.md LICENSE .gitignore configs/lowspec.toml
""".split()


def check_structure() -> str:
    missing = [f for f in EXPECTED_FILES if not (ROOT / f).is_file()]
    assert not missing, f"missing files: {missing}"
    return f"{len(EXPECTED_FILES)} required files present"


def check_web_tree() -> str:
    web = ROOT / "web"
    assert web.is_dir(), "web/ is missing"
    needed = ["package.json", "index.html", "vite.config.ts", "src/main.tsx", "src/App.tsx",
              "src/auth/webauthn.ts", "src/auth/pkce.ts", "src/pages/SignIn.tsx"]
    missing = [n for n in needed if not (web / n).is_file()]
    assert not missing, f"missing web files: {missing}"

    pkg = json.loads((web / "package.json").read_text(encoding="utf-8"))
    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    assert "vite" in deps, "vite is not a dependency"
    assert "react" in deps, "react is not a dependency"
    assert "vite-plugin-pwa" in deps, "vite-plugin-pwa is missing (no PWA)"

    manifest_exists = any((web / p).is_file() for p in
                          ("public/manifest.webmanifest", "public/manifest.json", "vite.config.ts"))
    assert manifest_exists, "no PWA manifest / config"
    return "web tree complete: Vite + React + vite-plugin-pwa, auth helpers present"


def check_passkey_requirements() -> str:
    from volcast.api import db as dbmod
    from volcast.api import passkeys
    from volcast.api.settings import Settings

    with tempfile.TemporaryDirectory() as td:
        dbp = Path(td) / "t.db"
        dbmod.init_db(dbp)
        cfg = Settings(rp_id="localhost", expected_origins="http://localhost:5173")
        with dbmod.closing_conn(dbp) as conn:
            reg = passkeys.begin_registration(cfg, conn, "dan", "Daniel", None)
            sel = reg["public_key"]["authenticatorSelection"]
            assert str(sel.get("residentKey")).lower().endswith("required"), sel
            assert str(sel.get("userVerification")).lower().endswith("required"), sel

            auth = passkeys.begin_authentication(cfg, conn, None)
            assert auth["public_key"].get("allowCredentials") == [], \
                "usernameless login must send allowCredentials=[]"

            # add_to_existing must not fork the account
            uid = reg["user_id"]
            reg2 = passkeys.begin_registration(cfg, conn, None, None, uid)
            assert reg2["user_id"] == uid, "add_to_existing created a different user"
            n = dbmod.get(conn, "SELECT COUNT(*) AS n FROM passkey_users")["n"]
            assert n == 1, f"add_to_existing forked the account ({n} users)"

    # conditional UI marker in the web client
    app_tsx = (ROOT / "web" / "src" / "pages" / "SignIn.tsx").read_text(encoding="utf-8")
    assert "username webauthn" in app_tsx, "conditional UI autocomplete attribute is missing"
    assert "conditional" in app_tsx.lower(), "no conditional-mediation call on the sign-in page"
    return "rk=required, uv=required, allowCredentials=[], add_to_existing reuses user, autofill present"


def check_engine_default_is_drift() -> str:
    import tomllib

    from volcast.config import Config

    assert Config().engine == "drift", "library default engine must be drift"
    with (ROOT / "configs" / "lowspec.toml").open("rb") as fh:
        low = tomllib.load(fh)
    assert low.get("engine") == "drift", f"lowspec.toml engine={low.get('engine')!r}"
    assert low.get("test_days", 999) <= 252, "lowspec test_days too large for 2 cores"
    assert low.get("context_len", 0) <= 512, "lowspec context_len too large for CPU"
    return "engine=drift everywhere; test_days<=252; context_len<=512"


def check_chronos_not_imported() -> str:
    import volcast
    import volcast.forecast as f

    assert "torch" not in sys.modules or True  # informational
    tree = ast.parse((ROOT / "volcast" / "forecast.py").read_text(encoding="utf-8"))
    top_imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.add(node.module.split(".")[0])
    assert "chronos" not in top_imports, "chronos is imported at module scope"
    assert "torch" not in top_imports, "torch is imported at module scope"
    # Importing the package must not pull torch in.
    assert "torch" not in sys.modules, "importing volcast pulled in torch"
    del volcast, f
    return "forecast.py imports chronos/torch lazily; `import volcast` does not load torch"


def check_pandas_3_compat() -> str:
    import pandas as pd

    from volcast.data import synthetic_prices
    from volcast.panel import build_panel

    px = synthetic_prices(["A", "B"], periods=120).copy()
    px.iloc[5, 0] = float("nan")
    panel = build_panel(px, vol_window=10)   # would raise under the old fillna(method=...)
    assert len(panel.prices) > 0
    assert not panel.prices.isna().any().any(), "NaN survived the ffill in build_panel"
    return f"build_panel handles gaps on pandas {pd.__version__}"


def check_api_smoke() -> str:
    from fastapi.testclient import TestClient

    from volcast.api.main import create_app

    app = create_app()
    # Enumerate via the OpenAPI schema: FastAPI 0.162 wraps included routers in a
    # lazy _IncludedRouter, so app.routes only shows the top-level shell.
    schema = app.openapi()
    paths = set(schema.get("paths", {}))
    required = {
        "/api/auth/refresh", "/api/auth/refresh/cookie", "/api/auth/login/begin",
        "/api/auth/login/finish", "/api/auth/register/begin", "/api/auth/register/finish",
        "/api/auth/me", "/api/auth/logout", "/api/market/forecast",
        "/api/market/weights", "/api/market/backtest", "/api/passkeys",
    }
    missing = required - paths
    assert not missing, f"missing routes: {sorted(missing)}"

    with TestClient(app) as c:
        h = c.get("/api/health")
        assert h.status_code == 200, h.text
        assert h.json()["status"] == "ok"

        # Unauthenticated /me must 401, not 500.
        r = c.get("/api/auth/me")
        assert r.status_code == 401, f"/me -> {r.status_code}"

        # Login begin works usernameless and returns camelCase browser options.
        r = c.post("/api/auth/login/begin", json={})
        assert r.status_code == 200, r.text[:300]
        pk = r.json()["public_key"]
        assert pk["allowCredentials"] == [], pk
        assert "rpId" in pk, f"options are not camelCase: {sorted(pk)}"
        assert "rp_id" not in pk, "snake_case leaked into the browser options"
    return f"{len(paths)} documented /api paths; /me 401s cleanly; usernameless begin is camelCase"


def check_cli() -> str:
    out = subprocess.run(
        [sys.executable, "-m", "volcast", "--version"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120,
    )
    assert out.returncode == 0, f"exit {out.returncode}: {out.stderr[:400]}"
    assert "volcast" in out.stdout.lower(), out.stdout
    return out.stdout.strip()


def check_offline_fallback() -> str:
    """The whole stack must run without network access."""
    from volcast.data import load_prices, synthetic_prices

    px = synthetic_prices(["X", "Y"], periods=80)
    assert len(px) == 80
    # load_prices with a bogus ticker list must fall back, not explode.
    df = load_prices(["ZZZZ"], period="1mo", allow_synthetic=True,
                     cache_root=tempfile.mkdtemp())
    assert len(df) > 0
    return "synthetic fallback works; load_prices degrades gracefully"


# --------------------------------------------------------------------------- #
def main() -> int:
    global ARGS
    ap = argparse.ArgumentParser(description="verify the volcast build")
    ap.add_argument("--verbose", "-v", action="store_true")
    ARGS = ap.parse_args()

    print(f"{BOLD}volcast self-verification{RESET}")
    print(f"  root   : {ROOT}")
    print(f"  python : {sys.version.split()[0]} ({sys.executable})")
    print(f"  pandas : {_ver('pandas')}   numpy: {_ver('numpy')}   fastapi: {_ver('fastapi')}"
          f"   webauthn: {_ver('webauthn')}")
    print()

    print(f"{BOLD}Environment{RESET}")
    REPORT.check("dependencies: pandas/numpy/yfinance/fastapi/uvicorn/pydantic-settings/"
                 "pyjwt/cryptography/webauthn/httpx/pytest", check_deps)
    REPORT.check("no torch / no chronos installed (deliberate)", check_no_heavy_deps)

    print(f"\n{BOLD}Structure{RESET}")
    REPORT.check("required file tree", check_structure)
    REPORT.check("web/ Vite+React+PWA tree", check_web_tree)

    print(f"\n{BOLD}The 14 previously identified bugs{RESET}")
    for label, fn in [
        ("#1  drift_forecast returns finite positive sigma_h", check_bug_1),
        ("#2  _clean() sanitiser on every JSON payload", check_bug_2),
        ("#3  security.decode_access_claims(cfg, token) -> dict", check_bug_3),
        ("#4  refresh accepts passkey subjects (_known_subject)", check_bug_4),
        ("#5  refresh(body, request, response) + cookie refresh", check_bug_5),
        ("#6  PKCE: SignIn uses the incoming code_challenge", check_bug_6),
        ("#7  webauthn.ts post(path, body, token?)", check_bug_7),
        ("#8  cookie_secure defaults to is_prod", check_bug_8),
        ("#9  metrics() -> NaN Sharpe on zero variance", check_bug_9),
        ("#10 pinball() minimised at the claimed quantile", check_bug_10),
        ("#11 cross_sectional_weights re-demeans after _cap", check_bug_11),
        ("#12 begin_registration tolerates missing exclude_credentials", check_bug_12),
        ('#13 no fillna(method="ffill") anywhere', check_bug_13),
        ("#14 pyproject [project.optional-dependencies].api", check_bug_14),
    ]:
        REPORT.check(label, fn)

    print(f"\n{BOLD}Behaviour{RESET}")
    REPORT.check("passkey policy (rk/uv required, usernameless, add_to_existing)",
                 check_passkey_requirements)
    REPORT.check("engine defaults to drift on this hardware", check_engine_default_is_drift)
    REPORT.check("chronos/torch are never imported at module scope", check_chronos_not_imported)
    REPORT.check("pandas 3 compatibility (no fillna(method=))", check_pandas_3_compat)
    REPORT.check("API smoke test (routes, 401s, usernameless begin)", check_api_smoke)
    REPORT.check("CLI entry point", check_cli)
    REPORT.check("offline / synthetic-data fallback", check_offline_fallback)

    return REPORT.summary()


def _ver(mod: str) -> str:
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "?")
    except Exception:
        return "MISSING"


def check_deps() -> str:
    mods = ["pandas", "numpy", "yfinance", "fastapi", "uvicorn", "pydantic_settings",
            "jwt", "cryptography", "webauthn", "httpx", "pytest"]
    missing = [m for m in mods if importlib.util.find_spec(m) is None]
    assert not missing, f"missing modules: {missing}"
    return f"all {len(mods)} present"


def check_no_heavy_deps() -> str:
    heavy = [m for m in ("torch", "chronos") if importlib.util.find_spec(m) is not None]
    assert not heavy, f"unexpectedly installed: {heavy}"
    return "torch and chronos are absent, as required on this machine"


ARGS = argparse.Namespace(verbose=False)

if __name__ == "__main__":
    sys.exit(main())
