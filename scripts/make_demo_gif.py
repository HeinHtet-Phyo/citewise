"""Capture a demo GIF of the live site using Playwright screenshots + Pillow/ffmpeg.

Usage: python scripts/make_demo_gif.py [url]
Saves: images/demo.gif (≤5 MB, ~960 px wide, 10–20 s)
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "https://citewise-v5w7.onrender.com/"
QUESTION = "How much is Statutory Sick Pay per week?"
OUT = Path("images/demo.gif")
FRAMES_DIR = Path("/tmp/demo_frames")
WIDTH, HEIGHT = 1280, 800
FPS = 5  # target frame rate for GIF
SCALE = 960  # output width in pixels


def wait_alive(url: str, timeout: int = 300) -> None:
    health = url.rstrip("/") + "/health"
    deadline = time.monotonic() + timeout
    print("Waiting for server...", end="", flush=True)
    while time.monotonic() < deadline:
        try:
            if httpx.get(health, timeout=15).status_code == 200:
                print(" ready.")
                time.sleep(3)
                return
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(6)
    raise RuntimeError("Server did not wake within timeout")


def capture_frames(url: str) -> list[Path]:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    n = 0

    def snap(page: object) -> None:
        nonlocal n
        p = FRAMES_DIR / f"frame_{n:04d}.png"
        page.screenshot(path=str(p))  # type: ignore[attr-defined]
        frames.append(p)
        n += 1

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.goto(url, timeout=120_000)
        time.sleep(1.5)
        snap(page)

        # Type the question slowly
        box = page.locator("#question")
        box.click()
        for char in QUESTION:
            box.type(char, delay=55)
            if n % 3 == 0:
                snap(page)

        snap(page)
        # Pause on the typed question so viewers can read it
        for _ in range(12):
            time.sleep(0.25)
            snap(page)

        # Click Ask and capture the spinner
        page.click("#askBtn")
        for _ in range(6):
            time.sleep(0.4)
            snap(page)

        # Wait for result, capturing every ~200 ms
        page.wait_for_selector("#result .answer-card, #result .refused-banner", timeout=120_000)
        for _ in range(8):
            time.sleep(0.25)
            snap(page)

        # Scroll down a little to show citations
        page.evaluate("window.scrollBy(0, 320)")
        time.sleep(0.4)
        for _ in range(6):
            time.sleep(0.3)
            snap(page)

        # Expand "Other pages searched"
        details = page.locator("details.other-pages")
        if details.count():
            details.locator("summary").click()
            time.sleep(0.5)
            for _ in range(8):
                time.sleep(0.3)
                snap(page)

        # Hold the final frame (4 s worth at FPS)
        for _ in range(FPS * 4):
            snap(page)

        browser.close()

    print(f"Captured {len(frames)} frames.")
    return frames


def build_gif(frames: list[Path]) -> None:
    # Use ffmpeg: scale to SCALE wide, palette-optimised GIF
    frame_pattern = str(FRAMES_DIR / "frame_%04d.png")
    palette = str(FRAMES_DIR / "palette.png")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            frame_pattern,
            "-vf",
            f"scale={SCALE}:-1:flags=lanczos,palettegen=stats_mode=diff",
            palette,
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            frame_pattern,
            "-i",
            palette,
            "-lavfi",
            f"scale={SCALE}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer",
            str(OUT),
        ],
        check=True,
        capture_output=True,
    )
    size_mb = OUT.stat().st_size / 1_048_576
    n_frames = len(frames)
    duration_s = n_frames / FPS
    print(f"GIF saved: {OUT} ({size_mb:.2f} MB, {duration_s:.1f}s, {n_frames} frames)")
    if size_mb > 5:
        print(f"WARNING: file is {size_mb:.2f} MB > 5 MB limit")


def warm_up(url: str) -> None:
    """Send a real /ask request (not just health) to ensure LLM workers are ready."""
    base = url.rstrip("/")
    wait_alive(url)
    print("Warming up server with a real question...", end="", flush=True)
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        try:
            r = httpx.post(
                f"{base}/ask",
                json={"question": "What is Statutory Sick Pay?"},
                timeout=180,
            )
            if r.status_code == 200:
                print(" warm.")
                return
            print(f"({r.status_code})", end="", flush=True)
        except Exception as exc:
            print(f"(err:{type(exc).__name__})", end="", flush=True)
        time.sleep(10)
    raise RuntimeError("Server did not warm up within timeout")


if __name__ == "__main__":
    warm_up(URL)
    frames = capture_frames(URL)
    build_gif(frames)
