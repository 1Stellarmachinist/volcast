#!/usr/bin/env python
"""Start the volcast API (and optionally the web dev server), then print the URL.

Defaults to the `drift` engine: Chronos-2 inference on a 2-core 1.8 GHz CPU is
unusably slow, so it is never selected here unless explicitly asked for.

    tools\\launch.py --engine drift
    tools\\launch.py --engine drift --port 8000 --no-open
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG_PATH = ROOT / ".volcast" / "launch.log"


def _ensure_streams() -> None:
    """Guarantee sys.stdout/sys.stderr exist.

    The desktop shortcut deliberately launches this with ``pythonw.exe`` so no
    console window appears. Under pythonw without redirection ``sys.stdout`` is
    ``None``, and the first ``print()`` then raises AttributeError *after* the
    port was bound -- the server dies silently and the shortcut looks broken.
    Everything is mirrored to a log file so the output is still recoverable.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_PATH, "a", encoding="utf-8", buffering=1)

    if sys.stdout is None or sys.stderr is None:
        class _Tee:
            """Minimal file-like object: logs everything, behaves like a stream.

            uvicorn's logging config calls ``sys.stdout.isatty()``, so the full
            stream protocol (isatty/encoding/fileno/writable) is required -- a
            bare object with only ``write`` crashes uvicorn at startup.
            """

            encoding = "utf-8"

            def __init__(self, stream, fallback):
                self._stream = stream
                self._fallback = fallback

            def write(self, data):
                if not isinstance(data, str):
                    data = str(data)
                try:
                    self._stream.write(data)
                    self._stream.flush()
                except Exception:
                    pass
                try:
                    self._fallback.write(data)
                except Exception:
                    pass
                return len(data)

            def writelines(self, lines):
                for line in lines:
                    self.write(line)

            def flush(self):
                for target in (self._stream, self._fallback):
                    try:
                        target.flush()
                    except Exception:
                        pass

            def isatty(self) -> bool:
                return False

            def writable(self) -> bool:
                return True

            def readable(self) -> bool:
                return False

            def seekable(self) -> bool:
                return False

            def fileno(self) -> int:
                return self._stream.fileno()

        if sys.stdout is None:
            sys.stdout = _Tee(log, open(os.devnull, "w", encoding="utf-8"))
        if sys.stderr is None:
            sys.stderr = _Tee(log, open(os.devnull, "w", encoding="utf-8"))


def free_port(start: int) -> int:
    port = start
    while port < start + 50:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    raise RuntimeError(f"no free port near {start}")


def print_banner(cfg, app_cfg, port: int) -> None:
    print("=" * 66)
    print("  volcast api")
    print("=" * 66)
    print(f"  engine        : {app_cfg.engine}")
    print(f"  horizon       : {cfg.horizon} bars")
    print(f"  universe      : {len(cfg.universe)} tickers")
    print(f"  env           : {app_cfg.env}")
    print(f"  secure cookies: {app_cfg.secure_cookies}")
    print(f"  url           : http://127.0.0.1:{port}")
    print(f"  docs          : http://127.0.0.1:{port}/docs")
    print("=" * 66)


def main() -> int:
    _ensure_streams()
    ap = argparse.ArgumentParser(description="launch the volcast API")
    ap.add_argument("--engine", default="drift", choices=["drift", "chronos"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--offline", action="store_true", help="force synthetic price data")
    args = ap.parse_args()

    if args.engine == "chronos":
        import importlib.util

        if importlib.util.find_spec("chronos") is None:
            print("ERROR: engine=chronos needs the 'chronos' extra (torch).", file=sys.stderr)
            print("       On this CPU-only machine use --engine drift.", file=sys.stderr)
            return 2
        os.environ["VOLCAST_ALLOW_CHRONOS"] = "1"

    # Make the chosen engine authoritative for the API process.
    os.environ["VOLCAST_ENGINE"] = args.engine
    os.environ.setdefault("VOLCAST_ENV", "dev")

    if args.offline:
        os.environ["VOLCAST_CACHE_DIR"] = str(ROOT / ".volcast" / "cache")

    try:
        import uvicorn
    except ModuleNotFoundError:
        print("ERROR: uvicorn is not installed. Run: pip install -e \".[api]\"", file=sys.stderr)
        return 3

    port = free_port(args.port)

    # Two distinct objects on purpose: Config holds the modelling knobs, Settings
    # holds env/JWT/cookie policy. Merging them silently was a real bug here.
    from volcast.api.settings import get_settings
    from volcast.config import load_config

    app_cfg = get_settings()
    model_cfg = load_config(ROOT / "configs" / "lowspec.toml")
    model_cfg.engine = args.engine
    print_banner(model_cfg, app_cfg, port)

    if not args.no_open:
        def _open() -> None:
            time.sleep(2.0)
            try:
                webbrowser.open(f"http://127.0.0.1:{port}/")
            except Exception:
                pass

        import threading

        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(
        "volcast.api.main:app",
        host=args.host,
        port=port,
        reload=args.reload,
        log_level="info",
        access_log=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
