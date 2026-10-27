"""Record the README demo GIF and screenshots by driving headless Chrome.

    pip install -r requirements-demo.txt
    make run                                  # the app, with the models you want to show
    python scripts/record_demo.py             # writes docs/images/demo.gif and screenshots

The storyboard: an analyst asks a two-part question and the run sheet fills in stage by
stage (plan, route, access, retrieve, verify); a row of the claim ledger opens its
source with the figure highlighted; then an external consultant asks the same question
and the access stage denies the internal-research part. Frames are captured at
2x and downscaled, so text stays sharp. A drawn cursor shows where each click lands.
"""

from __future__ import annotations

import argparse
import base64
import io
import itertools
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
QUESTION = "What was Uber's revenue in 2021, and what is our price target on Uber?"
WIDTH, HEIGHT, SCALE = 1200, 800, 2
CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
]

CURSOR_JS = """
(() => {
  if (document.getElementById('demo-cursor')) return;
  const c = document.createElement('div');
  c.id = 'demo-cursor';
  c.innerHTML = '<svg width="22" height="22" viewBox="0 0 24 24"><path d="M5 3l14 8-6 1.6L10 19z" fill="#101a17" stroke="#fff" stroke-width="1.5" stroke-linejoin="round"/></svg>';
  Object.assign(c.style, {position: 'fixed', left: '640px', top: '420px', zIndex: 99999, pointerEvents: 'none',
    transition: 'left .55s cubic-bezier(.3,.7,.2,1), top .55s cubic-bezier(.3,.7,.2,1)'});
  document.body.appendChild(c);
})();
"""


class Browser:
    """A minimal Chrome DevTools Protocol client: commands, screenshots, input."""

    def __init__(self, chrome: str, profile: str):
        self.proc = subprocess.Popen(
            [
                chrome,
                "--headless=new",
                "--remote-debugging-port=9333",
                f"--user-data-dir={profile}",
                f"--window-size={WIDTH},{HEIGHT}",
                "--hide-scrollbars",
                "--force-color-profile=srgb",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(50):
            try:
                targets = json.load(urllib.request.urlopen("http://127.0.0.1:9333/json"))
                page = next(t for t in targets if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        else:
            raise RuntimeError("Chrome did not start")
        self.ws = websocket.create_connection(page["webSocketDebuggerUrl"])
        self.ids = itertools.count(1)
        self.frames: list[tuple[Image.Image, float]] = []
        self.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=HEIGHT, deviceScaleFactor=SCALE, mobile=False)
        self.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])

    def call(self, method: str, **params):
        msg_id = next(self.ids)
        self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == msg_id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expression: str):
        result = self.call("Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True)
        return result.get("result", {}).get("value")

    def goto(self, url: str) -> None:
        self.call("Page.navigate", url=url)
        self.wait_for("document.readyState === 'complete' && !!document.body", 15)
        self.js("document.fonts.ready.then(() => true)")
        self.js(CURSOR_JS)

    def wait_for(self, condition: str, timeout: float) -> None:
        end = time.time() + timeout
        while time.time() < end:
            if self.js(f"!!({condition})"):
                return
            time.sleep(0.1)
        raise TimeoutError(condition)

    # -- capture --------------------------------------------------------------------------

    def snap(self) -> Image.Image:
        data = self.call("Page.captureScreenshot", format="png")["data"]
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")

    def record(self, seconds: float, until: str | None = None, timeout: float = 90) -> None:
        """Capture frames for `seconds`, or until the JS condition holds (then `seconds` more)."""
        end = time.time() + (timeout if until else seconds)
        while time.time() < end:
            self.frames.append((self.snap(), time.time()))
            if until and self.js(f"!!({until})"):
                until, end = None, time.time() + seconds

    def save_png(self, path: Path) -> None:
        self.snap().save(path, optimize=True)
        print(f"wrote {path}")

    # -- input ----------------------------------------------------------------------------

    def center(self, selector: str) -> tuple[float, float]:
        rect = self.js(
            f"(() => {{ const r = document.querySelector({json.dumps(selector)}).getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }})()"
        )
        return rect[0], rect[1]

    def move_to(self, selector: str) -> tuple[float, float]:
        x, y = self.center(selector)
        self.js(f"Object.assign(document.getElementById('demo-cursor').style, {{left: '{x - 4}px', top: '{y - 3}px'}})")
        self.record(0.7)
        self.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
        return x, y

    def click(self, selector: str, hold: float = 0.4) -> None:
        x, y = self.move_to(selector)
        for kind in ("mousePressed", "mouseReleased"):
            self.call("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", clickCount=1)
        self.record(hold)

    def type(self, selector: str, text: str) -> None:
        self.js(f"document.querySelector({json.dumps(selector)}).focus()")
        for i, char in enumerate(text):
            self.call("Input.insertText", text=char)
            if i % 2 == 1:
                self.frames.append((self.snap(), time.time()))

    def press_enter(self) -> None:
        for kind in ("keyDown", "keyUp"):
            self.call("Input.dispatchKeyEvent", type=kind, key="Enter", code="Enter", windowsVirtualKeyCode=13)

    def close(self) -> None:
        self.ws.close()
        self.proc.terminate()
        self.proc.wait(timeout=10)


def build_gif(frames: list[tuple[Image.Image, float]], path: Path, width: int = 1000) -> None:
    """Downscale, de-duplicate and quantize to one shared palette (no dithering keeps text crisp)."""
    height = round(HEIGHT * width / WIDTH)
    images, durations = [], []
    for (frame, at), (_, after) in zip(frames, [*frames[1:], (None, frames[-1][1] + 2.5)], strict=True):
        small = frame.resize((width, height), Image.LANCZOS)
        ms = max(40, int((after - at) * 1000))
        if images and small.tobytes() == images[-1].tobytes():
            durations[-1] += ms
        else:
            images.append(small)
            durations.append(ms)
    picks = sorted({round(i * (len(images) - 1) / 7) for i in range(8)})
    sample = Image.new("RGB", (width, height * len(picks)))
    for i, idx in enumerate(picks):
        sample.paste(images[idx], (0, height * i))
    palette = sample.quantize(colors=200, method=Image.Quantize.MAXCOVERAGE)
    quantized = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    quantized[0].save(path, save_all=True, append_images=quantized[1:], duration=durations, loop=0, optimize=True, disposal=1)
    print(f"wrote {path}: {len(quantized)} frames, {path.stat().st_size / 1e6:.1f} MB, {sum(durations) / 1000:.0f} s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "images")
    parser.add_argument("--chrome", default=next((p for p in CHROME_PATHS if Path(p).exists()), shutil.which("chromium") or ""))
    args = parser.parse_args()
    if not args.chrome:
        print("Chrome or Chromium not found; pass --chrome", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    base = args.base_url.rstrip("/")
    host = base.split("//", 1)[1].split(":")[0]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        b = Browser(args.chrome, profile)
        try:
            b.call("Network.enable")
            b.call("Network.setCookie", name="tickmark_user", value="analyst", domain=host, path="/")

            done = "!!document.querySelector('.turn:last-of-type .receipt:not([hidden])')"

            # 1. The desk, then an analyst asks a two-part question and watches the run sheet fill in.
            b.goto(f"{base}/")
            b.save_png(args.out / "home.png")
            b.record(1.4)
            b.click("#q", hold=0.2)
            b.type("#q", QUESTION)
            b.record(0.5)
            b.press_enter()
            b.record(1.6, until=done)
            b.save_png(args.out / "run-sheet.png")

            # 2. Scroll to the claim ledger and open a figure's source: the passage, highlighted.
            b.js("window.scrollTo({top: document.querySelector('.ledger').offsetTop - 140, behavior: 'smooth'})")
            b.record(1.4)
            b.click(".ledger-row.ok", hold=2.6)
            b.save_png(args.out / "claim-ledger.png")
            b.click("#drawer-close", hold=0.5)

            # 3. Switch to the external consultant and ask the same question.
            b.js("window.scrollTo({top: 0})")
            b.click(".who summary", hold=1.0)
            b.click(".menu form:last-of-type .menu-item", hold=0.3)  # external consultant
            b.wait_for(
                "document.readyState === 'complete' && document.querySelector('.who-name').textContent.includes('consultant')", 15
            )
            b.js(CURSOR_JS)
            b.record(0.8)
            b.click("#q", hold=0.2)
            b.type("#q", QUESTION)
            b.press_enter()
            b.record(3.0, until=done)
            b.save_png(args.out / "access-denied.png")

            build_gif(b.frames, args.out / "demo.gif")

            # 4. The audit trail, as a portfolio manager sees it.
            b.call("Network.setCookie", name="tickmark_user", value="pm", domain=host, path="/")
            b.goto(f"{base}/activity")
            b.js("document.getElementById('demo-cursor').remove()")
            b.save_png(args.out / "activity.png")
        finally:
            b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
