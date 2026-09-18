"""Drive every demo preset and check the walkthrough carousel against the run.

Spends real API tokens: one full screening run per preset.
Usage: python scripts/verify_walkthrough.py http://127.0.0.1:8123
"""

from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8123"
PRESETS = ["good_fit", "missing_must_have", "prompt_injection", "gray_zone"]

failures: list[str] = []

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 950})
    page.on("pageerror", lambda e: failures.append(f"pageerror: {e}"))
    page.on(
        "console",
        lambda m: failures.append(f"console.error: {m.text}") if m.type == "error" else None,
    )

    for preset in PRESETS:
        page.goto(BASE, wait_until="networkidle")
        options = page.eval_on_selector_all(
            "#presetSelect option", "els => els.map(e => e.value)"
        )
        if preset not in options:
            failures.append(f"{preset}: preset missing from the dropdown")
            continue

        page.select_option("#presetSelect", preset)
        page.click("#btnRunScreen")
        page.wait_for_selector(
            "#pipelineStatusBadge:has-text('Hoàn thành')", timeout=240_000
        )
        # Count, not visibility: the stepper auto-scrolls to its newest card, so
        # the first card's button is off-screen the moment the run finishes.
        page.wait_for_function(
            "document.querySelectorAll('.step-detail-btn').length > 0", timeout=30_000
        )

        buttons = page.locator(".step-detail-btn")
        traces = page.locator(
            ".step-card.completed, .step-card.shortcut-quarantine, "
            ".step-card.shortcut-reject, .step-card.shortcut-review"
        )
        if buttons.count() != traces.count():
            failures.append(
                f"{preset}: {buttons.count()} walkthrough buttons "
                f"for {traces.count()} trace cards"
            )

        # A finished run lands on Scorecard or Evidence, so the Trace tab -- and
        # every walkthrough button on it -- is hidden until we go back to it.
        page.click('#outputTabNav .tab-btn[data-tab="traceTab"]')
        page.wait_for_timeout(300)

        buttons.first.scroll_into_view_if_needed()
        buttons.first.click()
        page.wait_for_timeout(500)

        total = int(page.text_content("#wtPosition").split("/")[1])
        seen = []
        for _ in range(total):
            seen.append(page.text_content("#wtNodeName").strip())
            # Every mark must sit inside the text it is drawn on.
            bad = page.eval_on_selector_all(
                "#wtDocs .wt-mark",
                "els => els.filter(e => !e.textContent.trim()).length",
            )
            if bad:
                failures.append(f"{preset}: {bad} empty marks on node {seen[-1]}")
            page.keyboard.press("ArrowRight")
            page.wait_for_timeout(250)

        path_taken = page.evaluate("nodeSnapshots.map(s => s.node)")
        if seen != path_taken:
            failures.append(f"{preset}: carousel {seen} != path {path_taken}")

        page.keyboard.press("Escape")
        print(f"{preset}: {total} slides · {' → '.join(seen)}")

    browser.close()

print("\nfailures:", failures or "none")
sys.exit(1 if failures else 0)
