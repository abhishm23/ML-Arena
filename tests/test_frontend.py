"""ML Arena — frontend (AGENT A) checks.

Plain ``assert``-based script so it runs with or without pytest:

    C:\\ML Arena\\.venv\\Scripts\\python.exe "C:\\ML Arena\\tests\\test_frontend.py"
    C:\\ML Arena\\.venv\\Scripts\\python.exe -m pytest tests -q

It proves three things the product promises:

1. The five web assets exist and are non-empty.
2. ``index.html`` wires up ``styles.css`` / ``app.js`` with *relative* paths
   (the app is served by FastAPI with ``web/`` as the static root).
3. ``app.js`` and ``charts.js`` contain no ``http://`` / ``https://`` strings,
   so the UI cannot reach the network even by accident — it is offline-safe.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"

FILES = {
    "index.html": WEB / "index.html",
    "styles.css": WEB / "styles.css",
    "app.js": WEB / "app.js",
    "charts.js": WEB / "charts.js",
    "favicon.svg": WEB / "favicon.svg",
}

# the chart types the contract requires to be hand-drawn as inline SVG
REQUIRED_CHARTS = [
    "bar",
    "hbar",
    "line",
    "histogram",
    "donut",
    "scatter",
    "gauge",
    "radar",
]

# HTTP endpoints the UI is allowed to depend on (contract section 14)
REQUIRED_ENDPOINTS = [
    "/api/datasets",
    "/columns/preview",
    "/columns/apply",
    "/suggestions",
    "/api/usecases",
    "/api/runs",
    "/api/samples",
    "/api/health",
    "/ws/runs/",
    "/export?fmt=",
]

NODE_CANDIDATES = [
    Path(r"C:/Users/Abhishek Mishra/.workbuddy-ai/binaries/node/versions/22.22.2-2/node.exe"),
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------- tests --


def test_files_exist_and_are_non_empty() -> None:
    for name, path in FILES.items():
        assert path.is_file(), f"{name} is missing at {path}"
        assert path.stat().st_size > 0, f"{name} is empty"
        assert _read(path).strip(), f"{name} contains only whitespace"


def test_index_references_styles_and_app() -> None:
    html = _read(FILES["index.html"])
    assert "styles.css" in html, "index.html does not reference styles.css"
    assert "app.js" in html, "index.html does not reference app.js"
    assert "charts.js" in html, "index.html does not reference charts.js"
    assert "favicon.svg" in html, "index.html does not reference favicon.svg"

    # relative paths only: the app is served from web/ as the static root
    assert 'href="/web/' not in html, "index.html uses an absolute /web/ path"
    assert 'src="/web/' not in html, "index.html uses an absolute /web/ path"
    assert 'href="/styles.css"' not in html, "styles.css must be linked relatively"
    assert 'src="/app.js"' not in html, "app.js must be linked relatively"


def test_js_is_offline_safe() -> None:
    for name in ("app.js", "charts.js"):
        text = _read(FILES[name])
        for scheme in ("http://", "https://"):
            assert scheme not in text, f"{name} contains {scheme} — it must work fully offline"
        assert "createElementNS" not in text, (
            f"{name} uses createElementNS, which would need the SVG namespace URI"
        )
        assert "cdn" not in text.lower(), f"{name} mentions a CDN"
        assert "fonts.googleapis" not in text, f"{name} references an external font"


def test_index_has_no_external_resources() -> None:
    html = _read(FILES["index.html"])
    for scheme in ("http://", "https://"):
        assert scheme not in html, f"index.html contains {scheme}"
    # no remote scripts/styles/images
    assert not re.search(r'src\s*=\s*"//', html), "protocol-relative resource found"
    assert "integrity=" not in html, "an SRI attribute implies an external resource"


def test_styles_has_no_external_fonts_or_images() -> None:
    css = _read(FILES["styles.css"])
    assert "http://" not in css and "https://" not in css, "styles.css reaches off-box"
    assert "@import" not in css, "styles.css must not @import anything"
    assert "url(" not in css, "styles.css must not pull in external images"
    assert "font-family" in css, "styles.css should define the system font stack"


def test_charts_exports_every_required_type() -> None:
    js = _read(FILES["charts.js"])
    assert "global.Charts" in js, "charts.js does not publish window.Charts"
    for chart in REQUIRED_CHARTS:
        assert re.search(rf"\b{chart}:\s*{chart}\b", js), f"Charts.{chart} is not exported"
        assert re.search(rf"function\s+{chart}\s*\(", js), f"function {chart}() is missing"
    assert "svg" in js and "viewBox" in js, "charts.js should emit viewBox-scaled inline SVG"


def test_app_covers_the_six_steps_and_api() -> None:
    js = _read(FILES["app.js"])
    for endpoint in REQUIRED_ENDPOINTS:
        assert endpoint in js, f"app.js never calls {endpoint}"
    # six-step rail, websocket + polling fallback, exports, start over
    assert "data-goto" in js, "app.js does not drive the step rail"
    assert "WebSocket" in js, "app.js does not open the live event stream"
    assert "startPolling" in js, "app.js has no polling fallback for the socket"
    assert "startOver" in js, "app.js has no Start over action"
    assert "prefers-reduced-motion" in js or "reducedMotion" in js, (
        "app.js should respect prefers-reduced-motion"
    )


def test_index_has_six_steps_and_an_accessible_rail() -> None:
    html = _read(FILES["index.html"])
    for n in range(1, 7):
        assert f'id="step-{n}"' in html, f"step {n} section is missing"
    assert html.count('class="rail-step"') == 6, "the progress rail must show all six steps"
    assert "<button" in html, "the rail must use real buttons for keyboard access"
    # labels tied to inputs, and Escape-friendly modal scaffolding
    assert 'for="file-input"' in html, "the file input needs a <label for=...>"
    assert 'for="derived-name"' in html and 'for="derived-expr"' in html, (
        "the column builder inputs need labels"
    )
    assert 'id="modal"' in html, "the modal host is missing"


def test_favicon_is_a_real_svg() -> None:
    svg = _read(FILES["favicon.svg"])
    assert svg.lstrip().startswith("<svg"), "favicon.svg must start with an <svg> tag"
    assert svg.rstrip().endswith("</svg>"), "favicon.svg must be a complete SVG document"
    assert "viewBox" in svg, "favicon.svg needs a viewBox so it scales"


def test_javascript_parses_with_node() -> None:
    """Best-effort: node is not a hard dependency of this project."""
    node = None
    for candidate in NODE_CANDIDATES:
        if candidate.is_file():
            node = str(candidate)
            break
    if node is None:
        node = shutil.which("node")
    if node is None:
        print("  (skipped: node is not available to --check the JavaScript)")
        return
    for name in ("app.js", "charts.js"):
        proc = subprocess.run(
            [node, "--check", str(FILES[name])],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"node --check failed for {name}:\n{proc.stderr}"


# -------------------------------------------------------------- runner --


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}\n        {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}\n        {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print()
    if failures:
        print(f"{failures} of {len(tests)} frontend checks FAILED")
        return 1
    print(f"all {len(tests)} frontend checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
