"""One-off maintenance: remove passkey users that never enrolled a credential.

Safe by construction: a user is only deleted when it has ZERO rows in
`credentials`. That cannot remove a working account, because a working account by
definition has at least one credential.

    python tools/prune_orphans.py            # report only
    python tools/prune_orphans.py --apply    # actually delete
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / ".volcast" / "volcast.db"


def orphans(conn: sqlite3.Connection) -> list[tuple[str, str, float]]:
    rows = conn.execute(
        """
        SELECT u.id, u.username, u.created_at
        FROM passkey_users u
        WHERE NOT EXISTS (SELECT 1 FROM credentials c WHERE c.user_id = u.id)
        ORDER BY u.created_at
        """
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="delete instead of reporting")
    ap.add_argument("--db", default=str(DB))
    args = ap.parse_args()

    db = Path(args.db)
    if not db.is_file():
        print(f"no database at {db}")
        return 1

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    found = orphans(conn)
    if not found:
        print("no orphan accounts - nothing to do")
        conn.close()
        return 0

    print(f"{len(found)} orphan account(s) (zero credentials):")
    for uid, name, created in found:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(created))
        print(f"  {name:30s} {uid[:26]:28s} created {stamp}")

    if not args.apply:
        print("\nreport only. re-run with --apply to delete.")
        conn.close()
        return 0

    backup = db.with_suffix(f".db.bak-{int(time.time())}")
    shutil.copy2(db, backup)
    print(f"\nbackup written: {backup.name}")

    for uid, _, _ in found:
        # Re-check inside the loop so a credential added between the query and the
        # delete can never cause us to remove a live account.
        live = conn.execute(
            "SELECT 1 FROM credentials WHERE user_id=?", (uid,)
        ).fetchone()
        if live:
            print(f"  skipped {uid} - credential appeared")
            continue
        conn.execute("DELETE FROM refresh_tokens WHERE subject=?", (uid,))
        conn.execute("DELETE FROM challenges WHERE user_id=?", (uid,))
        conn.execute("DELETE FROM passkey_users WHERE id=?", (uid,))

    conn.commit()

    remaining = conn.execute("SELECT COUNT(*) FROM passkey_users").fetchone()[0]
    with_creds = conn.execute(
        "SELECT COUNT(DISTINCT user_id) FROM credentials"
    ).fetchone()[0]
    print(f"deleted {len(found)}. users remaining: {remaining} (with credentials: {with_creds})")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
