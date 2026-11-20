"""Check the live UI in a real browser at desktop and phone size and save screenshots."""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "https://citewise-v5w7.onrender.com/"
OUT = Path("images")
VIEWPORTS = {"desktop": {"width": 1280, "height": 900}, "phone": {"width": 390, "height": 844}}


def ask(page: Page, question: str) -> None:
    """Type a question, submit it and wait for the result."""
    page.goto(URL, timeout=120_000)
    page.fill("#question", question)
    page.click("#askBtn")
    page.wait_for_selector("#result .answer-card, #result .refused-banner", timeout=90_000)


def main() -> None:
    """Run the answered and refused checks for each viewport."""
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, size in VIEWPORTS.items():
            page = browser.new_page(viewport=size)
            ask(page, "What is the State Pension age?")
            cited = page.locator("#result > .citation").count()
            other = page.locator("details.other-pages")
            print(name, "main source cards:", cited, "other-pages blocks:", other.count())
            if other.count():
                print(name, "other summary:", other.locator("summary").inner_text())
                other.locator("summary").click()
                print(name, "other open:", other.get_attribute("open") is not None)
            print(name, "overflow-x:", page.evaluate("document.body.scrollWidth > innerWidth"))
            page.screenshot(path=str(OUT / f"ui_answer_{name}.png"), full_page=True)
            ask(page, "What is the best pizza topping?")
            print(
                name,
                "refusal text:",
                page.locator("#result").inner_text()[:200].replace("\n", " | "),
            )
            page.screenshot(path=str(OUT / f"ui_refusal_{name}.png"), full_page=True)
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
