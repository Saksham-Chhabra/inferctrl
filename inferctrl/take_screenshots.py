"""
Take screenshots of the running INFERCTRL web UI and embed them into the
existing INFERCTRL_Presentation.pptx.

Usage:
    python inferctrl/take_screenshots.py

Requires:
    - Server running on port 8765 (started by parent script)
    - playwright chromium installed
    - python-pptx installed
"""

import asyncio
import time
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCREENSHOTS_DIR = ROOT / "screenshots"
PPTX_PATH = ROOT / "INFERCTRL_Presentation.pptx"
PORT = 8765

PAGES = [
    ("control_room",  f"http://localhost:{PORT}/",            1920, 1080, 3000),
    ("chatbot",        f"http://localhost:{PORT}/chat",         1920, 1080, 2000),
    ("health_api",     f"http://localhost:{PORT}/health",        1920, 600,  1500),
    ("models_api",     f"http://localhost:{PORT}/v1/models",     1920, 600,  1500),
    ("docs",           f"http://localhost:{PORT}/docs",          1920, 1080, 3000),
    ("presentation",   f"http://localhost:{PORT}/presentation",  1920, 1080, 3000),
]


async def take_screenshots():
    from playwright.async_api import async_playwright

    SCREENSHOTS_DIR.mkdir(exist_ok=True)
    saved = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=1,
        )

        for name, url, width, height, wait_ms in PAGES:
            print(f"  Capturing {name} -> {url}")
            page = await context.new_page()
            await page.set_viewport_size({"width": width, "height": height})
            try:
                await page.goto(url, wait_until="networkidle", timeout=15000)
                await page.wait_for_timeout(wait_ms)
            except Exception as e:
                print(f"    WARNING: {e} — using domcontentloaded fallback")
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=10000)
                    await page.wait_for_timeout(wait_ms)
                except Exception as e2:
                    print(f"    ERROR: could not load {url}: {e2}")
                    await page.close()
                    continue

            out_path = SCREENSHOTS_DIR / f"{name}.png"
            await page.screenshot(path=str(out_path), full_page=False)
            saved[name] = out_path
            print(f"    Saved: {out_path.name}  ({out_path.stat().st_size // 1024} KB)")
            await page.close()

        await browser.close()

    return saved


def embed_into_pptx(screenshots: dict):
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    BG      = RGBColor(0x0b, 0x0f, 0x17)
    SURFACE = RGBColor(0x13, 0x19, 0x28)
    BORDER  = RGBColor(0x1e, 0x29, 0x44)
    BLUE    = RGBColor(0x38, 0xbd, 0xf8)
    EMERALD = RGBColor(0x34, 0xd3, 0x99)
    WHITE   = RGBColor(0xff, 0xff, 0xff)
    DIM     = RGBColor(0x94, 0xa3, 0xb8)
    AMBER   = RGBColor(0xfb, 0xbf, 0x24)

    prs = Presentation(str(PPTX_PATH))

    def blank_slide(prs):
        layout = prs.slide_layouts[6]
        return prs.slides.add_slide(layout)

    def fill_bg(slide):
        bg = slide.background
        fill = bg.fill
        fill.solid()
        fill.fore_color.rgb = BG

    def add_rect(slide, left, top, width, height, fill=SURFACE, line=BORDER, lw=1.0):
        s = slide.shapes.add_shape(1,
            Inches(left), Inches(top), Inches(width), Inches(height))
        if fill:
            s.fill.solid(); s.fill.fore_color.rgb = fill
        else:
            s.fill.background()
        if line:
            s.line.color.rgb = line; s.line.width = Pt(lw)
        else:
            s.line.fill.background()
        return s

    def add_label(slide, text, left, top, width, height, size=13, bold=False,
                  color=WHITE, align=PP_ALIGN.LEFT, italic=False):
        txb = slide.shapes.add_textbox(
            Inches(left), Inches(top), Inches(width), Inches(height))
        tf = txb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.alignment = align
        run = p.add_run(); run.text = text
        run.font.size = Pt(size); run.font.bold = bold
        run.font.italic = italic; run.font.color.rgb = color
        return txb

    def add_screenshot(slide, img_path, left, top, width, height):
        """Add screenshot image with a border frame."""
        # Outer border frame
        add_rect(slide, left - 0.04, top - 0.04, width + 0.08, height + 0.08,
                 fill=SURFACE, line=BLUE, lw=1.5)
        slide.shapes.add_picture(
            str(img_path),
            Inches(left), Inches(top), Inches(width), Inches(height)
        )

    # ── SCREENSHOT SLIDE A: Control Room Dashboard ──────────────────────────
    if "control_room" in screenshots:
        s = blank_slide(prs)
        fill_bg(s)

        add_label(s, "Live: Control Room Dashboard", 0.4, 0.12, 10, 0.55,
                  size=26, bold=True, color=BLUE)
        add_label(s, "http://localhost:8000  —  real-time request metrics, routing decisions, experiment log",
                  0.4, 0.65, 12.5, 0.32, size=11.5, color=DIM)

        add_screenshot(s, screenshots["control_room"],
                       left=0.35, top=1.05, width=12.6, height=6.2)

        # URL badge
        add_rect(s, 0.35, 7.18, 3.2, 0.24, fill=SURFACE, line=BLUE, lw=1.0)
        add_label(s, "http://localhost:8000", 0.5, 7.2, 3.0, 0.2,
                  size=9, color=BLUE)

    # ── SCREENSHOT SLIDE B: Chatbot Demo ────────────────────────────────────
    if "chatbot" in screenshots:
        s = blank_slide(prs)
        fill_bg(s)

        add_label(s, "Live: Interactive Chatbot Demo", 0.4, 0.12, 10, 0.55,
                  size=26, bold=True, color=EMERALD)
        add_label(s, "http://localhost:8000/chat  —  OpenAI-compatible gateway driving the chat UI in real-time",
                  0.4, 0.65, 12.5, 0.32, size=11.5, color=DIM)

        add_screenshot(s, screenshots["chatbot"],
                       left=0.35, top=1.05, width=12.6, height=6.2)

        add_rect(s, 0.35, 7.18, 3.6, 0.24, fill=SURFACE, line=EMERALD, lw=1.0)
        add_label(s, "http://localhost:8000/chat", 0.5, 7.2, 3.4, 0.2,
                  size=9, color=EMERALD)

    # ── SCREENSHOT SLIDE C: REST API Docs ───────────────────────────────────
    if "docs" in screenshots:
        s = blank_slide(prs)
        fill_bg(s)

        add_label(s, "Live: FastAPI Auto-Generated REST API Docs", 0.4, 0.12, 12, 0.55,
                  size=26, bold=True, color=AMBER)
        add_label(s, "http://localhost:8000/docs  —  interactive Swagger UI showing all gateway endpoints",
                  0.4, 0.65, 12.5, 0.32, size=11.5, color=DIM)

        add_screenshot(s, screenshots["docs"],
                       left=0.35, top=1.05, width=12.6, height=6.2)

        add_rect(s, 0.35, 7.18, 3.6, 0.24, fill=SURFACE, line=AMBER, lw=1.0)
        add_label(s, "http://localhost:8000/docs", 0.5, 7.2, 3.4, 0.2,
                  size=9, color=AMBER)

    # ── SCREENSHOT SLIDE D: Health + Models API side-by-side ────────────────
    if "health_api" in screenshots and "models_api" in screenshots:
        s = blank_slide(prs)
        fill_bg(s)

        add_label(s, "Live: Provider Health & Models API", 0.4, 0.12, 12, 0.55,
                  size=26, bold=True, color=WHITE)
        add_label(s, "Raw JSON responses from GET /health and GET /v1/models — providers auto-detected at startup",
                  0.4, 0.65, 12.5, 0.32, size=11.5, color=DIM)

        add_screenshot(s, screenshots["health_api"],
                       left=0.35, top=1.05, width=6.15, height=3.5)
        add_screenshot(s, screenshots["models_api"],
                       left=6.82, top=1.05, width=6.15, height=3.5)

        # Labels
        add_rect(s, 0.35, 4.65, 6.15, 0.35, fill=SURFACE, line=BLUE, lw=1.0)
        add_label(s, "GET /health  —  provider availability status", 0.5, 4.68, 5.9, 0.28,
                  size=10.5, color=BLUE)

        add_rect(s, 6.82, 4.65, 6.15, 0.35, fill=SURFACE, line=EMERALD, lw=1.0)
        add_label(s, "GET /v1/models  —  registered models + metadata", 6.97, 4.68, 5.9, 0.28,
                  size=10.5, color=EMERALD)

        # Explanation
        add_rect(s, 0.35, 5.1, 12.62, 2.25, fill=SURFACE, line=BORDER, lw=1.0)
        explanations = [
            ("Provider Auto-Detection:", "On startup, INFERCTRL pings every configured backend.", BLUE),
            ("Gemini Adapter:",          "Checks for GEMINI_API_KEY env var — marks ONLINE or UNAVAILABLE.", WHITE),
            ("Ollama Adapter:",          "Connects to localhost:11434/api/tags — marks ONLINE or UNAVAILABLE.", WHITE),
            ("Simulator:",               "Always ONLINE — fully deterministic, requires no external services.", EMERALD),
            ("Models Endpoint:",         "Returns all registered models with provider, capabilities, pricing.", AMBER),
        ]
        for i, (label, desc, col) in enumerate(explanations):
            add_label(s, label, 0.55, 5.22 + i * 0.38, 2.4, 0.32, size=11, bold=True, color=col)
            add_label(s, desc,  2.98, 5.22 + i * 0.38, 10.0, 0.32, size=11, color=DIM)

    # ── SCREENSHOT SLIDE E: Presentation HTML ───────────────────────────────
    if "presentation" in screenshots:
        s = blank_slide(prs)
        fill_bg(s)

        add_label(s, "In-Browser Pitch Deck", 0.4, 0.12, 10, 0.55,
                  size=26, bold=True, color=RGBColor(0xc0, 0x84, 0xfc))
        add_label(s, "http://localhost:8000/presentation  —  zero-dependency HTML slide deck served by the same FastAPI app",
                  0.4, 0.65, 12.5, 0.32, size=11.5, color=DIM)

        add_screenshot(s, screenshots["presentation"],
                       left=0.35, top=1.05, width=12.6, height=6.2)

        add_rect(s, 0.35, 7.18, 4.5, 0.24, fill=SURFACE,
                 line=RGBColor(0xc0, 0x84, 0xfc), lw=1.0)
        add_label(s, "http://localhost:8000/presentation", 0.5, 7.2, 4.3, 0.2,
                  size=9, color=RGBColor(0xc0, 0x84, 0xfc))

    out_path = ROOT / "INFERCTRL_Presentation_v2.pptx"
    prs.save(str(out_path))
    total_slides = len(prs.slides)
    print(f"\n[SUCCESS] Updated PPTX saved -> {out_path}")
    print(f"   Total slides now: {total_slides}")
    print(f"   Size: {out_path.stat().st_size / 1024:.1f} KB")


async def main():
    print("=== INFERCTRL Screenshot & PPTX Embedder ===\n")

    # Wait for server to be ready
    import urllib.request
    print(f"Waiting for server on port {PORT}...")
    for i in range(30):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
            print(f"  Server is up!\n")
            break
        except Exception:
            time.sleep(1)
    else:
        print("  ERROR: Server did not start in time. Exiting.")
        sys.exit(1)

    print("Taking screenshots...")
    screenshots = await take_screenshots()

    if not screenshots:
        print("ERROR: No screenshots captured.")
        sys.exit(1)

    print(f"\nEmbedding {len(screenshots)} screenshot(s) into presentation...")
    embed_into_pptx(screenshots)


if __name__ == "__main__":
    asyncio.run(main())
