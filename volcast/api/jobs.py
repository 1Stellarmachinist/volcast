"""Background job registry: forecasts/backtests run off the request thread."""

from __future__ import annotations

import threading
import time
import traceback
import uuid

from . import db as dbmod
from .service import backtest_payload, forecast_payload, screen_payload

_LOCK = threading.Lock()
_THREADS: dict[str, threading.Thread] = {}


def create_job(conn, kind: str, payload: dict) -> str:
    jid = uuid.uuid4().hex
    ts = dbmod.now()
    conn.execute(
        "INSERT INTO jobs (id, kind, status, payload, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        (jid, kind, "queued", dbmod.dumps(payload), ts, ts),
    )
    conn.commit()
    return jid


def update_job(conn, jid: str, **fields) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    params = list(fields.values()) + [dbmod.now(), jid]
    conn.execute(f"UPDATE jobs SET {sets}, updated_at=? WHERE id=?", params)
    conn.commit()


def get_job(conn, jid: str) -> dict | None:
    row = dbmod.get(conn, "SELECT * FROM jobs WHERE id=?", (jid,))
    if not row:
        return None
    return {
        "id": row["id"],
        "kind": row["kind"],
        "status": row["status"],
        "payload": dbmod.loads(row["payload"]),
        "result": dbmod.loads(row["result"]),
        "error": row["error"],
        "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]),
    }


def _run(kind: str, params: dict) -> dict:
    tickers = params.get("tickers") or []
    if kind == "forecast":
        return forecast_payload(tickers, horizon=params.get("horizon", 5),
                                window=params.get("window", 20),
                                engine=params.get("engine", "drift"))
    if kind == "screen":
        return screen_payload(tickers, top=params.get("top", 10))
    if kind == "backtest":
        return backtest_payload(
            tickers,
            cost_bps=params.get("cost_bps", 5.0),
            test_days=params.get("test_days", 126),
            rebalance=params.get("rebalance", 5),
            target_vol=params.get("target_vol", 0.15),
            max_weight=params.get("max_weight", 0.25),
        )
    raise ValueError(f"unknown job kind: {kind}")


def submit(db_path: str, kind: str, params: dict) -> str:
    """Queue a job and run it on a daemon thread. Returns the job id."""
    with dbmod.closing_conn(db_path) as conn:
        jid = create_job(conn, kind, params)

    def _work() -> None:
        with dbmod.closing_conn(db_path) as conn:
            update_job(conn, jid, status="running")
        try:
            result = _run(kind, params)
            with dbmod.closing_conn(db_path) as conn:
                update_job(conn, jid, status="done", result=dbmod.dumps(result), error=None)
        except Exception as exc:  # pragma: no cover - surfaced via the API
            with dbmod.closing_conn(db_path) as conn:
                update_job(conn, jid, status="error", error=f"{exc}\n{traceback.format_exc()[-1200:]}")
        finally:
            with _LOCK:
                _THREADS.pop(jid, None)

    t = threading.Thread(target=_work, name=f"volcast-job-{jid[:8]}", daemon=True)
    with _LOCK:
        _THREADS[jid] = t
    t.start()
    return jid


def wait(db_path: str, jid: str, timeout: float = 30.0) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with dbmod.closing_conn(db_path) as conn:
            job = get_job(conn, jid)
        if job and job["status"] in {"done", "error"}:
            return job
        time.sleep(0.05)
    return get_job_dict(db_path, jid)


def get_job_dict(db_path: str, jid: str) -> dict | None:
    with dbmod.closing_conn(db_path) as conn:
        return get_job(conn, jid)
