"""ML Arena launcher.

Starts the local server and opens the interface in the default browser.

    python run.py                 # default port 8765
    python run.py --port 9000
    python run.py --no-browser
    python run.py --reload        # developer mode
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Make `mlarena` importable no matter where the launcher is invoked from.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlarena import config  # noqa: E402  (must precede polars)

BANNER = r"""
  __  __ _        _                    _
 |  \/  | |      / \   _ __ ___ _ __   __ _
 | |\/| | |     / _ \ | '__/ _ \ '_ \ / _` |
 | |  | | |___ / ___ \| | |  __/ | | | (_| |
 |_|  |_|_____/_/   \_\_|  \___|_| |_|\__,_|

  Where models compete. You just pick the goal.
  Runs entirely on this computer — nothing is uploaded anywhere.
"""


def free_port(preferred: int) -> int:
    """Return `preferred` when it is available, otherwise the next free port."""
    for candidate in range(preferred, preferred + 40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return candidate
    raise SystemExit(f"No free port found between {preferred} and {preferred + 40}.")


def open_when_ready(url: str, timeout: float = 20.0) -> None:
    """Open the browser once the server actually answers, not before."""
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=1.5):
                webbrowser.open(url)
                return
        except Exception:
            time.sleep(0.35)
    webbrowser.open(url)  # last resort: open it anyway


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ML Arena locally.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="preferred port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes (dev)")
    args = parser.parse_args()

    config.ensure_dirs()
    port = free_port(args.port)
    url = f"http://{args.host}:{port}"

    print(BANNER)
    caps = config.capability_report()
    print(f"  Python      {caps['python']}")
    print(f"  Cores       {caps['cpu_count']}")
    print(f"  Memory      {caps['memory_gb']} GB")
    print(f"  Accelerator {'GPU' if caps['gpu'] else 'CPU only (no GPU needed)'}")
    print(f"  Engines     polars {caps.get('polars') or 'n/a'}, scikit-learn {caps.get('sklearn')}")
    print()
    print(f"  Interface   {url}")
    print("  Stop with   Ctrl+C")
    print()

    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()

    import uvicorn

    try:
        uvicorn.run(
            "mlarena.api.server:app",
            host=args.host,
            port=port,
            reload=args.reload,
            log_level="warning",
            access_log=False,
        )
    except KeyboardInterrupt:
        pass
    print("\nML Arena stopped.")


if __name__ == "__main__":
    main()
