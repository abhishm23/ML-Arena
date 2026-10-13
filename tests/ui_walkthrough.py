"""Drive ML Arena through a real browser via the Chrome DevTools Protocol.

Walks the actual six-step wizard the way a person would — clicking the sample
card, picking columns, pressing run — and screenshots each stage. This is the
only check that proves the interface, the API and the engine work together.

    python tests/_ui_walkthrough.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "shots"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
APP = "http://127.0.0.1:8770/"
DEBUG_PORT = 9333

SHOTS.mkdir(exist_ok=True)


def launch_chrome() -> subprocess.Popen:
    profile = ROOT / "shots" / "_chrome-profile"
    # Start from a clean profile every run, otherwise Chrome serves the previous
    # app.js from its disk cache and we end up screenshotting stale code.
    if profile.exists():
        import shutil

        shutil.rmtree(profile, ignore_errors=True)
    profile.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-application-cache",
            f"--remote-debugging-port={DEBUG_PORT}",
            f"--user-data-dir={profile}",
            "--window-size=1500,1700",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_for_devtools(timeout: float = 25.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json/list", timeout=2) as r:
                targets = json.load(r)
            for t in targets:
                if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                    return t["webSocketDebuggerUrl"]
        except Exception:
            time.sleep(0.4)
    raise SystemExit("Chrome DevTools endpoint never came up.")


class Session:
    def __init__(self, ws) -> None:
        self.ws = ws
        self.n = 0

    async def call(self, method: str, params: dict | None = None) -> dict:
        self.n += 1
        msg_id = self.n
        await self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        while True:
            raw = json.loads(await self.ws.recv())
            if raw.get("id") == msg_id:
                return raw

    async def js(self, expression: str, *, await_promise: bool = True):
        res = await self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": await_promise,
                "returnByValue": True,
                "userGesture": True,
            },
        )
        result = res.get("result", {})
        if "exceptionDetails" in result:
            raise RuntimeError(json.dumps(result["exceptionDetails"])[:600])
        return result.get("result", {}).get("value")

    async def shot(self, name: str) -> Path:
        res = await self.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
        data = res["result"]["data"]
        path = SHOTS / name
        path.write_bytes(base64.b64decode(data))
        print(f"    saved {name} ({path.stat().st_size / 1024:.0f} KB)")
        return path

    async def shot_viewport(self, name: str) -> Path:
        """Capture only what is on screen, so fixed overlays land in the right place."""
        res = await self.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
        path = SHOTS / name
        path.write_bytes(base64.b64decode(res["result"]["data"]))
        print(f"    saved {name} ({path.stat().st_size / 1024:.0f} KB)")
        return path

    async def until(self, condition: str, label: str, timeout: float = 40.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if await self.js(f"(() => {{ try {{ return !!({condition}); }} catch(e) {{ return false; }} }})()"):
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.5)
        print(f"    TIMEOUT waiting for: {label}")
        return False


SET_ROLE = """
window.__setRole = function (key, values) {
  var sel = document.querySelector('#role-pickers select[data-role="' + key + '"]');
  if (!sel) return 'missing';
  Array.prototype.forEach.call(sel.options, function (o) { o.selected = values.indexOf(o.value) !== -1; });
  sel.dispatchEvent(new Event('change', { bubbles: true }));
  return 'ok';
};
'ready'
"""


async def main() -> int:
    import websockets

    print("Launching Chrome (headless) ...")
    chrome = launch_chrome()
    ws_url = wait_for_devtools()
    print(f"  devtools at {ws_url[:70]}...")

    ok = True
    try:
        async with websockets.connect(ws_url, max_size=80 * 1024 * 1024) as ws:
            s = Session(ws)
            await s.call("Page.enable")
            await s.call("Runtime.enable")

            print("1. Loading the interface ...")
            await s.call("Page.navigate", {"url": APP})
            await s.until("document.querySelectorAll('.sample-card').length > 0", "samples to render")
            await asyncio.sleep(1.0)
            await s.shot("03-step1-start.png")

            print("2. Opening the flagship sample ...")
            await s.js("document.querySelector('[data-sample=\"orders_discount_leakage.csv\"]').click()")
            if not await s.until(
                "!document.getElementById('step-2').hidden && "
                "document.querySelectorAll('#column-table tbody tr').length > 5",
                "dataset profile to render",
                timeout=60,
            ):
                ok = False
            await asyncio.sleep(0.8)
            await s.shot("04-step2-data.png")

            shape = await s.js(
                "(() => { const s = document.getElementById('dataset-summary');"
                " return s ? s.innerText.replace(/\\s+/g,' ').slice(0,220) : 'none'; })()"
            )
            print(f"    dataset summary: {shape}")

            print("3. Choosing the business question ...")
            await s.js("document.getElementById('to-step-3').click()")
            await s.until("document.querySelectorAll('.uc-card').length > 0", "use-case cards")
            await asyncio.sleep(0.6)
            await s.shot("05-step3-usecases.png")

            print("4. Picking Discount leakage and mapping columns ...")
            await s.js("document.querySelector('.uc-card[data-usecase=\"discount_leakage\"]').click()")
            await s.until("document.querySelectorAll('#role-pickers select[data-role]').length > 0",
                          "role pickers")
            await asyncio.sleep(0.6)

            mapping = {
                "discount": ["discount_pct"],
                "money": ["revenue"],
                "customer": ["customer_id"],
                "segment": ["customer_segment"],
                "product": ["product_category"],
                "channel": ["channel"],
                "quantity": ["quantity"],
                "date": ["order_date"],
            }
            await s.js(SET_ROLE)
            for key, values in mapping.items():
                outcome = await s.js(f"window.__setRole({json.dumps(key)}, {json.dumps(values)})")
                if outcome != "ok":
                    print(f"    role {key}: {outcome}")
            await asyncio.sleep(0.8)
            await s.shot("06-step4-columns.png")

            offered = await s.js(
                "(() => { const sel = document.querySelector('#role-pickers select[data-role=\"discount\"]');"
                " return sel ? Array.prototype.map.call(sel.options, o => o.value).join(', ') : 'none'; })()"
            )
            print(f"    discount picker offered only: {offered}")

            print("5. Running the arena ...")
            await s.js("document.getElementById('btn-run').click()")
            await s.until("!document.getElementById('step-5').hidden", "step 5 to appear", timeout=20)
            await asyncio.sleep(4.0)
            await s.shot("07-step5-live.png")

            mid = await s.js(
                "(() => { const c = document.getElementById('console');"
                " return c ? c.innerText.replace(/\\s+/g,' ').slice(0,300) : 'none'; })()"
            )
            print(f"    live log: {mid}")

            print("    waiting for the race to finish ...")
            finished = await s.until(
                "!document.getElementById('step-6').hidden", "step 6 (results)", timeout=200
            )
            if not finished:
                ok = False
            await asyncio.sleep(2.0)
            await s.shot("08-step6-results.png")

            champion = await s.js(
                "(() => { const c = document.getElementById('step-6');"
                " return c ? c.innerText.replace(/\\s+/g,' ').slice(0, 700) : 'none'; })()"
            )
            print(f"    results text: {champion}")

            impact = await s.js(
                "(() => { const n = document.querySelector('.impact-num');"
                " const l = document.querySelector('.impact-label');"
                " return (n ? n.innerText.replace(/\\s+/g,' ') : 'none') + '  ||  ' +"
                "        (l ? l.innerText.replace(/\\s+/g,' ') : 'none'); })()"
            )
            print(f"    impact callout: {impact}")

            # a tight crop of the top of the results page
            await s.js("window.scrollTo(0, 0); true")
            await asyncio.sleep(0.5)
            res = await s.call("Page.captureScreenshot", {
                "format": "png",
                "clip": {"x": 0, "y": 0, "width": 1500, "height": 950, "scale": 1},
            })
            (SHOTS / "10-results-top.png").write_bytes(base64.b64decode(res["result"]["data"]))
            print("    saved 10-results-top.png (tight crop of the payoff screen)")

            # scroll to the lower half of the results and shoot again
            await s.js("window.scrollTo(0, Math.floor(document.body.scrollHeight * 0.45)); true")
            await asyncio.sleep(1.0)
            await s.shot("09-step6-results-lower.png")

            counts = await s.js(
                "(() => ({svgs: document.querySelectorAll('#step-6 svg').length,"
                " tables: document.querySelectorAll('#step-6 table').length,"
                " cards: document.querySelectorAll('#step-6 .insight-card, #step-6 .insight').length,"
                " why: document.querySelectorAll('.why-card').length,"
                " toasts: document.querySelectorAll('.toast').length}))()"
            )
            print(f"    results DOM: {counts}")

            why = await s.js(
                "(() => { const c = document.querySelectorAll('.why-card');"
                " if (!c.length) return 'NONE';"
                " return c.length + ' cards | ' +"
                " c[0].querySelector('.why-label').innerText.trim() + ' | ' +"
                " c[0].querySelector('.why-summary').innerText.trim(); })()"
            )
            print(f"    why-this-row: {why}")

            # open the full explanation in the modal
            modal = await s.js(
                "(() => { const b = document.querySelector('[data-why]');"
                " if (!b) return 'no show-all button'; b.click();"
                " const m = document.getElementById('modal');"
                " if (!m || m.hidden) return 'modal did not open';"
                " return document.getElementById('modal-title').innerText + ' | drivers: ' +"
                " document.querySelectorAll('#modal-body .why-driver').length; })()"
            )
            print(f"    explanation modal: {modal}")
            await asyncio.sleep(0.7)
            await s.shot_viewport("08-why-this-row.png")
            await s.js("var c=document.getElementById('modal-close'); if(c) c.click(); true")

            if counts.get("why", 0) < 5:
                print("    PROBLEM: expected at least 5 why-this-row cards")
                ok = False

            # capture any console errors we may have caused
            errors = await s.js("(window.__errors || []).slice(0, 5).join(' | ')")
            if errors:
                print(f"    page errors: {errors}")
                ok = False
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=10)
        except Exception:
            chrome.kill()

    print("\nUI walkthrough", "COMPLETED" if ok else "HAD PROBLEMS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
