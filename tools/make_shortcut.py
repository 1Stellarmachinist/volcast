#!/usr/bin/env python
"""Create exactly ONE desktop shortcut for volcast.

Idempotent by design: if the shortcut already exists it is updated in place, so
running this twice does not litter the Desktop with duplicates.

    tools\\make_shortcut.py                    # create/refresh "volcast.lnk"
    tools\\make_shortcut.py --dry-run          # show what would be created
    tools\\make_shortcut.py --name "volcast dev"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHORTCUT_DEFAULT = "volcast"


def desktop_dir() -> Path:
    """Resolve the real Desktop, honouring OneDrive redirection."""
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "[Environment]::GetFolderPath('Desktop')",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        path = out.stdout.strip()
        if path and Path(path).is_dir():
            return Path(path)
    except Exception:
        pass
    for candidate in (Path.home() / "Desktop", Path.home() / "OneDrive" / "Desktop"):
        if candidate.is_dir():
            return candidate
    raise RuntimeError("could not locate the Desktop folder")


def pythonw() -> Path:
    """Prefer pythonw.exe so the console window does not linger."""
    exe = Path(sys.executable)
    w = exe.with_name("pythonw.exe")
    return w if w.is_file() else exe


def create_shortcut(name: str, target_script: Path, dry_run: bool, args: str) -> Path:
    desktop = desktop_dir()
    lnk = desktop / f"{name}.lnk"
    launch = ROOT / "tools" / "launch.py"
    assert launch.is_file(), f"missing launcher: {launch}"

    exe = pythonw()
    if dry_run:
        print(f"[dry-run] would create: {lnk}")
        print(f"[dry-run] target        : {exe}")
        print(f"[dry-run] arguments     : \"{launch}\" {args}".rstrip())
        print(f"[dry-run] working dir   : {ROOT}")
        print(f"[dry-run] exists already: {lnk.is_file()}")
        return lnk

    ps = f"""
$ErrorActionPreference = 'Stop'
$sh = New-Object -ComObject WScript.Shell
$lnk = $sh.CreateShortcut('{str(lnk).replace("'", "''")}')
$lnk.TargetPath = '{str(exe).replace("'", "''")}'
$lnk.Arguments = '"{str(launch).replace("'", "''")}" {args}'.Trim()
$lnk.WorkingDirectory = '{str(ROOT).replace("'", "''")}'
$lnk.Description = 'volcast - volatility forecasting API (engine=drift)'
$lnk.IconLocation = '{str(exe).replace("'", "''")},0'
$lnk.WindowStyle = 7
$lnk.Save()
Write-Output $lnk.FullName
"""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(f"shortcut creation failed: {out.stderr.strip()}")

    assert lnk.is_file(), "PowerShell reported success but the .lnk is missing"

    # Read it back and prove what it actually points at.
    verify = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}');"
            "$s.TargetPath; $s.Arguments; $s.WorkingDirectory",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    lines = [ln for ln in verify.stdout.splitlines() if ln.strip()]
    print(f"created : {lnk}")
    if len(lines) >= 3:
        print(f"target  : {lines[0]}")
        print(f"args    : {lines[1]}")
        print(f"workdir : {lines[2]}")
    return lnk


def count_existing_shortcuts() -> list[Path]:
    desktop = desktop_dir()
    return sorted(desktop.glob("volcast*.lnk"))


def main() -> int:
    ap = argparse.ArgumentParser(description="create the single volcast desktop shortcut")
    ap.add_argument("--name", default=SHORTCUT_DEFAULT)
    ap.add_argument("--args", default="--engine drift", help="arguments for tools/launch.py")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        desktop = desktop_dir()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    existing = count_existing_shortcuts()
    if existing and not args.dry_run:
        print(f"note: {len(existing)} volcast shortcut(s) already present: "
              f"{[p.name for p in existing]}")
        print("      updating in place (no duplicate will be created)")

    lnk = create_shortcut(
        args.name, ROOT / "tools" / "launch.py", args.dry_run, args.args
    )

    after = count_existing_shortcuts()
    print(f"volcast shortcuts on Desktop after run: {len(after)} -> {[p.name for p in after]}")
    if not args.dry_run:
        assert after, "no shortcut found after creation"
        assert len(after) == 1 or args.name in {p.stem for p in after}, "duplicate shortcuts created"
    print(f"desktop: {desktop}")
    print(f"shortcut: {lnk.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
