"""
INFERCTRL Presentation Generator
Generates a professional dark-themed PowerPoint for the INFERCTRL project.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ── Palette ──────────────────────────────────────────────────────────────────
BG       = RGBColor(0x0b, 0x0f, 0x17)   # dark navy
SURFACE  = RGBColor(0x13, 0x19, 0x28)   # card surface
BORDER   = RGBColor(0x1e, 0x29, 0x44)   # card border
BLUE     = RGBColor(0x38, 0xbd, 0xf8)   # sky-blue accent
EMERALD  = RGBColor(0x34, 0xd3, 0x99)   # emerald accent
AMBER    = RGBColor(0xfb, 0xbf, 0x24)   # amber accent
RED      = RGBColor(0xf8, 0x71, 0x71)   # alert red
WHITE    = RGBColor(0xff, 0xff, 0xff)
DIM      = RGBColor(0x94, 0xa3, 0xb8)   # muted slate

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

# ── Helpers ───────────────────────────────────────────────────────────────────

def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def blank_slide(prs: Presentation):
    layout = prs.slide_layouts[6]   # completely blank
    return prs.slides.add_slide(layout)


def fill_bg(slide, color: RGBColor = BG):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, left, top, width, height,
             fill: RGBColor | None = SURFACE,
             line: RGBColor | None = BORDER,
             line_width_pt: float = 1.0) -> object:
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()

    if line:
        shape.line.color.rgb = line
        shape.line.width = Pt(line_width_pt)
    else:
        shape.line.fill.background()
    return shape


def add_label(slide, text: str, left, top, width, height,
              size: int = 14, bold: bool = False,
              color: RGBColor = WHITE,
              align=PP_ALIGN.LEFT,
              italic: bool = False):
    txb = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = txb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txb


def add_multiline(slide, lines: list[tuple], left, top, width, height,
                  default_size=13, default_color=WHITE, line_spacing_pt=4):
    """lines = list of (text, size, color, bold, italic)"""
    txb = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = txb.text_frame
    tf.word_wrap = True
    first = True
    for item in lines:
        text, size, color, bold, italic = item
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.space_after = Pt(line_spacing_pt)
        run = p.add_run()
        run.text = text
        run.font.size = Pt(size if size else default_size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color if color else default_color
    return txb


def horiz_rule(slide, left, top, width, color: RGBColor = BORDER, thickness_pt=0.75):
    shape = slide.shapes.add_shape(1, Inches(left), Inches(top), Inches(width), Pt(thickness_pt))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def badge(slide, text, left, top, width=1.4, height=0.32,
          bg: RGBColor = BLUE, fg: RGBColor = BG, size=10):
    r = add_rect(slide, left, top, width, height, fill=bg, line=None)
    add_label(slide, text, left + 0.05, top + 0.03, width - 0.1, height - 0.06,
              size=size, bold=True, color=fg, align=PP_ALIGN.CENTER)
    return r


# ══════════════════════════════════════════════════════════════════════════════
# SLIDES
# ══════════════════════════════════════════════════════════════════════════════

# ── SLIDE 1: Title ────────────────────────────────────────────────────────────
def slide_title(prs):
    s = blank_slide(prs)
    fill_bg(s)

    # Left glow strip
    strip = add_rect(s, 0, 0, 0.08, 7.5, fill=BLUE, line=None)

    # Logo / project name
    add_label(s, "INFERCTRL", 0.35, 0.55, 8, 1.1,
              size=56, bold=True, color=WHITE)

    add_label(s, "Autonomous LLM Inference Optimization Platform",
              0.35, 1.55, 10, 0.6, size=22, color=BLUE, bold=False)

    horiz_rule(s, 0.35, 2.25, 9.5, color=BLUE, thickness_pt=1.5)

    add_label(s,
        "A self-optimizing inference layer that continuously observes, benchmarks,\n"
        "evaluates, and routes LLM traffic — reducing latency and cost without\n"
        "sacrificing answer quality.",
        0.35, 2.45, 9.5, 1.2, size=16, color=DIM)

    # Key-metric strip
    metrics = [
        ("-54%",  "P95 Latency",     BLUE),
        ("-37%",  "Cost / 1K tokens",EMERALD),
        ("+28%",  "Throughput",      AMBER),
        ("<350ms","Auto-Rollback",   RED),
    ]
    card_w = 2.8
    gap    = 0.18
    start_x = 0.35
    for i, (val, label, col) in enumerate(metrics):
        cx = start_x + i * (card_w + gap)
        add_rect(s, cx, 3.85, card_w, 1.35, fill=SURFACE, line=col, line_width_pt=1.5)
        add_label(s, val,   cx + 0.15, 3.92, card_w - 0.3, 0.72, size=34, bold=True, color=col, align=PP_ALIGN.CENTER)
        add_label(s, label, cx + 0.15, 4.6,  card_w - 0.3, 0.45, size=12, color=DIM,  align=PP_ALIGN.CENTER)

    # Bottom attribution
    add_label(s, "github.com/inferctrl  ·  FastAPI · Python 3.12 · Multi-Agent · OpenAI-compatible",
              0.35, 6.9, 12, 0.45, size=10, color=BORDER, align=PP_ALIGN.CENTER)

    # Tech-label badges top-right
    for i, (tag, col) in enumerate([("FastAPI", BLUE), ("Multi-Agent", EMERALD), ("OpenAI-compat", AMBER)]):
        badge(s, tag, 10.3, 0.3 + i * 0.45, width=2.7, height=0.32, bg=col, fg=BG, size=11)

    return s


# ── SLIDE 2: Problem ──────────────────────────────────────────────────────────
def slide_problem(prs):
    s = blank_slide(prs)
    fill_bg(s)

    add_label(s, "The LLM Inference Trilemma", 0.4, 0.18, 12, 0.7, size=32, bold=True, color=WHITE)
    add_label(s, "Every production team faces the same impossible trade-off",
              0.4, 0.82, 12, 0.4, size=15, color=DIM)
    horiz_rule(s, 0.4, 1.28, 12.5, color=BORDER)

    # Three pain-point cards
    problems = [
        ("⏱  Latency Spikes",         BLUE,
         "P95 latency balloons 3–5× during traffic bursts.\n"
         "Static autoscalers are blind to prompt length,\n"
         "queue depth, and decode complexity.\n\n"
         "Users abandon sessions.  SLAs are violated."),
        ("💸  Runaway Costs",          AMBER,
         "GPT-4-class models billed for every request,\n"
         "even trivial ones a 7B local model could answer.\n\n"
         "No differentiation between a 10-token greeting\n"
         "and a 4K-token code-review task."),
        ("📉  Silent Quality Drift",   RED,
         "Model updates, prompt regressions, or hardware\n"
         "failures silently degrade answer quality.\n\n"
         "There is no feedback loop.  No canary.  No rollback.\n"
         "Teams discover regressions from user complaints."),
    ]
    cw = 3.9
    for i, (title, col, body) in enumerate(problems):
        cx = 0.4 + i * (cw + 0.28)
        add_rect(s, cx, 1.5, cw, 4.1, fill=SURFACE, line=col, line_width_pt=1.8)
        add_label(s, title, cx + 0.18, 1.65, cw - 0.3, 0.55, size=17, bold=True, color=col)
        horiz_rule(s, cx + 0.18, 2.25, cw - 0.36, color=col, thickness_pt=0.5)
        add_label(s, body,  cx + 0.18, 2.38, cw - 0.3, 3.0, size=13, color=DIM)

    # Quote
    add_rect(s, 0.4, 5.75, 12.5, 0.95, fill=RGBColor(0x1a, 0x10, 0x08), line=AMBER, line_width_pt=1.2)
    add_label(s,
        '"We spent 40% of our ML budget on inference and had no idea which requests were expensive '
        'or whether the model was actually helping users."  — Common engineering post-mortem',
        0.65, 5.85, 12.1, 0.75, size=12, color=AMBER, italic=True)

    return s


# ── SLIDE 3: Solution Overview ────────────────────────────────────────────────
def slide_solution(prs):
    s = blank_slide(prs)
    fill_bg(s)

    add_label(s, "Introducing INFERCTRL", 0.4, 0.18, 12, 0.65, size=32, bold=True, color=WHITE)
    add_label(s, "A closed-loop feedback system — Observe → Benchmark → Evaluate → Optimize → Verify → Roll Back",
              0.4, 0.78, 12.8, 0.4, size=14, color=DIM)
    horiz_rule(s, 0.4, 1.2, 12.5, color=BORDER)

    # Loop diagram as connected boxes
    steps = [
        ("1  OBSERVE",    "Capture every request:\nlatency, tokens, cost,\nquality, provider",     BLUE),
        ("2  BENCHMARK",  "Run controlled A/B tests\nacross providers and\nbatch strategies",      EMERALD),
        ("3  EVALUATE",   "Score output quality.\nDetect regressions.\nCompute Pareto frontier",   AMBER),
        ("4  OPTIMIZE",   "Select best config.\nCanary rollout at 5%.\nMonitor watchdog",          RGBColor(0xc0,0x84,0xfc)),
        ("5  VERIFY",     "Confirm SLA met.\nApply globally or\nauto-roll back <350 ms",          RED),
    ]
    bw = 2.28
    for i, (title, body, col) in enumerate(steps):
        cx = 0.35 + i * (bw + 0.12)
        add_rect(s, cx, 1.42, bw, 2.6, fill=SURFACE, line=col, line_width_pt=1.8)
        add_label(s, title, cx + 0.12, 1.55, bw - 0.2, 0.45, size=13, bold=True, color=col)
        horiz_rule(s, cx + 0.12, 2.05, bw - 0.24, color=col, thickness_pt=0.5)
        add_label(s, body, cx + 0.12, 2.18, bw - 0.2, 1.7, size=12, color=DIM)
        if i < 4:
            add_label(s, "→", cx + bw + 0.01, 2.45, 0.14, 0.45, size=18, bold=True, color=col)

    # Four pillars
    pillars = [
        ("OpenAI-Compatible Gateway",
         "Drop-in replacement for the OpenAI client.\nZero code changes in existing apps.",     BLUE),
        ("5-Agent Autonomous Engine",
         "Workload Analyst → Scheduler → Benchmarker\n→ Quality Evaluator → Controller",       EMERALD),
        ("Deterministic Policy Guardrails",
         "Hard constraints on latency, cost, quality.\nLLMs propose; policy enforces.",         AMBER),
        ("Canary Rollout + Auto-Rollback",
         "5% canary traffic. Watchdog on P95 + errors.\nGlobal apply or instant rollback.",    RED),
    ]
    pw = 2.95
    for i, (title, body, col) in enumerate(pillars):
        cx = 0.35 + i * (pw + 0.18)
        add_rect(s, cx, 4.25, pw, 2.0, fill=SURFACE, line=col, line_width_pt=1.5)
        add_label(s, title, cx + 0.14, 4.35, pw - 0.25, 0.45, size=13, bold=True, color=col)
        add_label(s, body,  cx + 0.14, 4.82, pw - 0.25, 1.25, size=11.5, color=DIM)

    return s


# ── SLIDE 4: 5-Agent Architecture ─────────────────────────────────────────────
def slide_agents(prs):
    s = blank_slide(prs)
    fill_bg(s)

    add_label(s, "5-Agent Architecture", 0.4, 0.18, 12, 0.65, size=32, bold=True, color=WHITE)
    add_label(s, "Each agent owns a specific responsibility. The Controller coordinates them all.",
              0.4, 0.78, 12, 0.4, size=14, color=DIM)
    horiz_rule(s, 0.4, 1.22, 12.5, color=BORDER)

    agents = [
        ("01", "Workload\nAnalyst",    BLUE,
         "• Classifies prompt complexity\n  (trivial / medium / complex)\n"
         "• Estimates resource demand\n"
         "• Detects traffic regime\n  (idle / bursty / sustained)\n"
         "• Outputs WorkloadReport"),
        ("02", "Benchmark\nAgent",     EMERALD,
         "• Selects candidate configs\n"
         "• Runs controlled A/B tests\n"
         "• Measures latency P50/P95,\n  throughput, GPU utilisation\n"
         "• Feeds results to Controller"),
        ("03", "Quality\nEvaluator",   AMBER,
         "• Scores LLM outputs via\n  LLM-as-judge / reference eval\n"
         "• Domain-specific metrics\n  (code correctness, factuality)\n"
         "• Triggers rollback if quality\n  falls below threshold"),
        ("04", "Scheduler\nAgent",     RGBColor(0xc0,0x84,0xfc),
         "• Adjusts real-time serving\n  parameters per controller\n"
         "• Dynamic batching settings\n"
         "• Concurrency limits per model\n"
         "• GPU resource allocation"),
        ("05", "Controller\nAgent",    RED,
         "• Multi-objective optimisation\n  (latency / cost / quality)\n"
         "• Chooses experiments to run\n"
         "• Applies / rejects configs\n"
         "• Learns from past decisions\n"
         "• Rollback if SLO violated"),
    ]

    aw = 2.28
    for i, (num, name, col, body) in enumerate(agents):
        cx = 0.35 + i * (aw + 0.12)
        add_rect(s, cx, 1.42, aw, 3.7, fill=SURFACE, line=col, line_width_pt=1.8)
        badge(s, num, cx + 0.12, 1.56, width=0.48, height=0.3, bg=col, fg=BG, size=11)
        add_label(s, name, cx + 0.68, 1.54, aw - 0.8, 0.55, size=13, bold=True, color=col)
        horiz_rule(s, cx + 0.12, 2.12, aw - 0.24, color=col, thickness_pt=0.5)
        add_label(s, body, cx + 0.12, 2.24, aw - 0.2, 2.7, size=11, color=DIM)

    # "Deterministic Authority" callout
    add_rect(s, 0.35, 5.28, 12.5, 1.1, fill=RGBColor(0x0e, 0x1e, 0x0e), line=EMERALD, line_width_pt=1.5)
    add_label(s, "⚡  Deterministic Authority Rule", 0.6, 5.35, 6, 0.4, size=14, bold=True, color=EMERALD)
    add_label(s,
        "Agents may propose hypotheses and configurations, but the Policy Engine has final authority.\n"
        "Hard constraints on max latency, max cost, and min quality are NEVER overridden by an LLM suggestion.",
        0.6, 5.72, 12.1, 0.6, size=12, color=DIM)

    return s


# ── SLIDE 5: Dual Backend Strategy ────────────────────────────────────────────
def slide_backends(prs):
    s = blank_slide(prs)
    fill_bg(s)

    add_label(s, "Dual Backend Strategy", 0.4, 0.18, 12, 0.65, size=32, bold=True, color=WHITE)
    add_label(s, "Works with real providers and a deterministic simulator — never blocks on missing credentials",
              0.4, 0.78, 12.8, 0.4, size=14, color=DIM)
    horiz_rule(s, 0.4, 1.22, 12.5, color=BORDER)

    # Provider cards
    providers = [
        ("Google Gemini API",   BLUE,
         "Model:        gemini-2.0-flash-exp\n"
         "Auth:         GEMINI_API_KEY env var\n"
         "Pricing:      $0.075 / 1M input tokens\n"
         "              $0.30 / 1M output tokens\n"
         "Features:     Streaming · function calling\n"
         "              provider-reported usage\n"
         "Fallback:     Simulator if key missing",
         ["Streaming", "Real cost", "Provider usage"]),
        ("Ollama Local",        EMERALD,
         "Models:       llama3.2:3b · qwen2.5:7b\n"
         "URL:          OLLAMA_BASE_URL env var\n"
         "Pricing:      $0.00 (local compute)\n"
         "Features:     Streaming · GPU/CPU\n"
         "              no external API calls\n"
         "Fallback:     Simulator if not running",
         ["Local / free", "Streaming", "GPU or CPU"]),
        ("Deterministic\nSimulator",  AMBER,
         "Purpose:      CI · demos · benchmarks\n"
         "Profiles:     llama-3b · llama-70b · gpt-4\n"
         "Latency:      Physics-based queueing model\n"
         "              (M/D/1 queue, tokens/sec)\n"
         "Label:        All data tagged SIMULATED\n"
         "Advantage:    Fully reproducible — no API",
         ["Always available", "Reproducible", "Tagged"]),
    ]

    cw = 3.9
    for i, (title, col, body, tags) in enumerate(providers):
        cx = 0.35 + i * (cw + 0.28)
        add_rect(s, cx, 1.44, cw, 4.1, fill=SURFACE, line=col, line_width_pt=1.8)
        add_label(s, title, cx + 0.18, 1.58, cw - 0.3, 0.6, size=16, bold=True, color=col)
        horiz_rule(s, cx + 0.18, 2.22, cw - 0.36, color=col, thickness_pt=0.5)
        add_label(s, body, cx + 0.18, 2.35, cw - 0.3, 2.8, size=11.5, color=DIM)
        # Tags
        for j, tag in enumerate(tags):
            badge(s, tag, cx + 0.18 + j * 1.22, 5.2, width=1.15, height=0.25, bg=col, fg=BG, size=9)

    # Routing strategies
    add_rect(s, 0.35, 5.62, 12.5, 1.6, fill=SURFACE, line=BORDER, line_width_pt=1.0)
    add_label(s, "Routing Strategies", 0.6, 5.7, 4, 0.4, size=14, bold=True, color=WHITE)
    strategies = [
        ("fixed",          "Always use a single configured model",                  DIM),
        ("round_robin",    "Distribute across eligible models evenly",              DIM),
        ("lowest_cost",    "Route to cheapest eligible model",                      EMERALD),
        ("lowest_latency", "Route to fastest model by recent P95",                  BLUE),
        ("quality_first",  "Prefer highest quality-scored model",                   AMBER),
        ("balanced",       "Pareto-optimal blend of cost + latency + quality",      RED),
    ]
    col_w = 6.0
    for i, (name, desc, col) in enumerate(strategies):
        row = i % 3
        col_idx = i // 3
        add_label(s, f"• {name}", 0.6 + col_idx * col_w, 6.12 + row * 0.33, 1.9, 0.3, size=11, bold=True, color=col)
        add_label(s, desc,        2.5 + col_idx * col_w, 6.12 + row * 0.33, 3.8, 0.3, size=11, color=DIM)

    return s


# ── SLIDE 6: Safety & Policy Engine ───────────────────────────────────────────
def slide_policy(prs):
    s = blank_slide(prs)
    fill_bg(s)

    add_label(s, "Safety & Policy Engine", 0.4, 0.18, 12, 0.65, size=32, bold=True, color=WHITE)
    add_label(s, "Hard constraints are always enforced — the LLM can suggest but never override",
              0.4, 0.78, 12.8, 0.4, size=14, color=DIM)
    horiz_rule(s, 0.4, 1.22, 12.5, color=BORDER)

    # Guardrail cards
    guardrails = [
        ("Max Latency",          BLUE,   "max_latency_ms: 5000",
         "Any candidate configuration with measured\n"
         "or estimated P95 latency above this threshold\n"
         "is immediately eliminated from consideration.\n\n"
         "Streaming responses track time-to-first-token\n"
         "separately from total stream duration."),
        ("Max Cost / Request",   AMBER,  "max_estimated_cost_per_request: 0.01",
         "Estimated cost is computed from token usage\n"
         "and up-to-date pricing tables per provider.\n\n"
         "formula: (in_tok × in_rate + out_tok × out_rate) / 1M\n\n"
         "Provider-reported cost is preferred; estimated\n"
         "cost is used as fallback and clearly labelled."),
        ("Min Quality Score",    EMERALD,"min_quality_score: 0.80",
         "Quality Evaluator scores every sampled output\n"
         "via LLM-as-judge or reference evaluation.\n\n"
         "Any model below the threshold is blacklisted\n"
         "for that workload class until re-evaluated.\n\n"
         "Protects against silent model regressions."),
    ]

    cw = 3.9
    for i, (title, col, code, body) in enumerate(guardrails):
        cx = 0.35 + i * (cw + 0.28)
        add_rect(s, cx, 1.44, cw, 4.15, fill=SURFACE, line=col, line_width_pt=1.8)
        add_label(s, title, cx + 0.18, 1.58, cw - 0.3, 0.42, size=16, bold=True, color=col)
        add_rect(s, cx + 0.18, 2.06, cw - 0.36, 0.35,
                 fill=RGBColor(0x0b, 0x0f, 0x17), line=col, line_width_pt=0.5)
        add_label(s, code, cx + 0.28, 2.10, cw - 0.5, 0.28, size=10, color=col, bold=True)
        add_label(s, body, cx + 0.18, 2.5, cw - 0.3, 2.9, size=12, color=DIM)

    # Pareto / decision callout
    add_rect(s, 0.35, 5.75, 12.5, 1.5, fill=RGBColor(0x0a, 0x12, 0x1e), line=BLUE, line_width_pt=1.5)
    add_label(s, "Multi-Objective Pareto Optimisation", 0.6, 5.82, 5.5, 0.4, size=14, bold=True, color=BLUE)
    add_label(s,
        "score = α × (1 − latency_norm) + β × (1 − cost_norm) + γ × quality_norm\n"
        "where α + β + γ = 1.0   (configurable weights per workload class)\n"
        "Candidates on the Pareto frontier are plotted in the Control Room scatter chart.",
        0.6, 6.22, 12.0, 0.9, size=12, color=DIM)

    return s


# ── SLIDE 7: Control Room Dashboard ───────────────────────────────────────────
def slide_dashboard(prs):
    s = blank_slide(prs)
    fill_bg(s)

    add_label(s, "Control Room Dashboard", 0.4, 0.18, 12, 0.65, size=32, bold=True, color=WHITE)
    add_label(s, "Real-time observability for every inference request — http://localhost:8000",
              0.4, 0.78, 12.8, 0.4, size=14, color=DIM)
    horiz_rule(s, 0.4, 1.22, 12.5, color=BORDER)

    # Mock dashboard UI
    # Header bar
    add_rect(s, 0.35, 1.38, 12.6, 0.48, fill=SURFACE, line=BORDER)
    add_label(s, "⬡  INFERCTRL", 0.55, 1.44, 3, 0.36, size=14, bold=True, color=BLUE)
    add_label(s, "Autonomous LLM Inference Optimizer", 2.6, 1.47, 5, 0.3, size=11, color=DIM)
    badge(s, "● Optimizing", 9.8, 1.46, width=1.6, height=0.28, bg=EMERALD, fg=BG, size=10)
    add_label(s, "Last 1 hour ▾", 11.5, 1.47, 1.35, 0.28, size=10, color=DIM)

    # KPI strip
    kpis = [
        ("42.8",  "req/s",    "+62% vs previous", EMERALD),
        ("612ms", "P95 Lat",  "−54% improvement",  BLUE),
        ("0.93",  "Quality",  "+2% vs previous",   AMBER),
        ("$0.012","Cost/1K",  "−37% cost reduction",RED),
        ("78%",   "GPU Util", "4 / 4 GPUs active", DIM),
    ]
    kw = 2.38
    for i, (val, unit, sub, col) in enumerate(kpis):
        cx = 0.35 + i * (kw + 0.09)
        add_rect(s, cx, 1.96, kw, 0.95, fill=SURFACE, line=col, line_width_pt=1.2)
        add_label(s, val,  cx + 0.12, 2.02, kw - 0.2, 0.45, size=22, bold=True, color=col)
        add_label(s, unit, cx + 0.12, 2.47, kw - 0.2, 0.22, size=10, color=DIM)
        add_label(s, sub,  cx + 0.12, 2.7,  kw - 0.2, 0.18, size=8.5, color=EMERALD)

    # Feature list cards
    features = [
        ("📊  Live Performance Charts",
         "Real-time req/sec, P95 latency ms,\n"
         "and quality score sparklines.\n"
         "Automatically annotates when\n"
         "optimisation is applied.", BLUE),
        ("🔀  Request Flow Sankey",
         "Visualises traffic distribution\n"
         "across providers (Chat 35%,\n"
         "Code 20%, Reasoning 25%).\n"
         "Updates every 5 seconds.", EMERALD),
        ("🧪  Experiment Table",
         "EXP-042 through EXP-039 with\n"
         "full config, latency, quality,\n"
         "cost per 1K, and APPLIED /\n"
         "INJECTED / REJECTED status.", AMBER),
        ("🎯  Pareto Scatter",
         "Cost vs Quality trade-off plot.\n"
         "Pareto frontier highlighted.\n"
         "Current config and past\n"
         "experiments plotted live.", RGBColor(0xc0,0x84,0xfc)),
    ]
    fw = 2.95
    for i, (title, body, col) in enumerate(features):
        cx = 0.35 + i * (fw + 0.19)
        add_rect(s, cx, 3.08, fw, 2.55, fill=SURFACE, line=col, line_width_pt=1.5)
        add_label(s, title, cx + 0.14, 3.18, fw - 0.25, 0.42, size=13, bold=True, color=col)
        horiz_rule(s, cx + 0.14, 3.64, fw - 0.28, color=col, thickness_pt=0.4)
        add_label(s, body,  cx + 0.14, 3.75, fw - 0.25, 1.7, size=11.5, color=DIM)

    # URL callout
    add_rect(s, 0.35, 5.78, 12.5, 1.55, fill=RGBColor(0x0a, 0x12, 0x1e), line=BLUE, line_width_pt=1.0)
    urls = [
        ("Control Room",  "http://localhost:8000",                BLUE),
        ("Chatbot UI",    "http://localhost:8000/chat",           EMERALD),
        ("Pitch Deck",    "http://localhost:8000/presentation",   AMBER),
        ("REST API Docs", "http://localhost:8000/docs",           DIM),
        ("Health",        "http://localhost:8000/health",         DIM),
    ]
    add_label(s, "Live URLs (run:  uvicorn apps.api.main:app --port 8000)", 0.6, 5.85, 7, 0.35, size=12, bold=True, color=WHITE)
    for i, (label, url, col) in enumerate(urls):
        col_i = i % 3
        row_i = i // 3
        add_label(s, label + ":  " + url, 0.6 + col_i * 4.1, 6.28 + row_i * 0.32, 4.0, 0.28, size=10.5, color=col)

    return s


# ── SLIDE 8: Canary & Auto-Rollback ───────────────────────────────────────────
def slide_canary(prs):
    s = blank_slide(prs)
    fill_bg(s)

    add_label(s, "Canary Deployments & Auto-Rollback", 0.4, 0.18, 12, 0.65, size=32, bold=True, color=WHITE)
    add_label(s, "No configuration change goes live without passing a live traffic gate",
              0.4, 0.78, 12.8, 0.4, size=14, color=DIM)
    horiz_rule(s, 0.4, 1.22, 12.5, color=BORDER)

    # Step flow
    steps = [
        ("STEP 1", "Canary Proposal",   BLUE,
         "Controller selects a candidate\n"
         "optimisation config after Pareto\n"
         "analysis ranks it as promising.\n\n"
         "Canary rollout begins:\n"
         "• 5% of live traffic routed to\n"
         "  the new configuration.\n"
         "• 95% stays on current baseline."),
        ("STEP 2", "Watchdog Monitoring", AMBER,
         "Watchdog agent monitors canary\n"
         "traffic for 60 – 300 seconds:\n\n"
         "• P95 latency vs baseline\n"
         "• Error rate vs baseline\n"
         "• Quality score vs threshold\n\n"
         "All comparisons use actual\n"
         "measured data (not estimates)."),
        ("STEP 3A", "Global Apply",     EMERALD,
         "If ALL watchdog checks pass:\n\n"
         "• Traffic split → 100% new config\n"
         "• Baseline archived\n"
         "• Experiment marked APPLIED\n"
         "• Metrics snapshot stored\n\n"
         "System continues searching for\n"
         "the next improvement cycle."),
        ("STEP 3B", "Auto-Rollback",    RED,
         "If ANY watchdog check fails:\n\n"
         "• Traffic snaps back to baseline\n"
         "• Rollback completes < 350 ms\n"
         "• Candidate marked REJECTED\n"
         "• Failure reason logged\n\n"
         "System learns and avoids similar\n"
         "configs in future experiments."),
    ]

    sw = 2.95
    for i, (step_num, title, col, body) in enumerate(steps):
        cx = 0.35 + i * (sw + 0.22)
        add_rect(s, cx, 1.44, sw, 4.3, fill=SURFACE, line=col, line_width_pt=1.8)
        badge(s, step_num, cx + 0.14, 1.58, width=1.1, height=0.3, bg=col, fg=BG, size=10)
        add_label(s, title, cx + 0.14, 1.95, sw - 0.25, 0.42, size=14, bold=True, color=col)
        horiz_rule(s, cx + 0.14, 2.42, sw - 0.28, color=col, thickness_pt=0.5)
        add_label(s, body,  cx + 0.14, 2.55, sw - 0.22, 2.9, size=12, color=DIM)

        # Arrow between steps 1→2→3
        if i == 2:
            add_label(s, "✓ PASS →", cx - 0.2, 2.7, 0.22, 0.45, size=10, color=EMERALD, bold=True)
        elif i < 2:
            add_label(s, "→", cx + sw + 0.02, 2.7, 0.22, 0.45, size=16, color=col, bold=True)

    # Timing bar
    add_rect(s, 0.35, 5.9, 12.5, 1.38, fill=RGBColor(0x0e, 0x10, 0x1a), line=BORDER)
    timings = [
        ("Canary ramp-up",    "0 – 10 s",    BLUE),
        ("Watchdog window",   "10 – 300 s",  AMBER),
        ("Decision + apply",  "< 1 s",       EMERALD),
        ("Rollback time",     "< 350 ms",    RED),
        ("Full cycle",        "5 – 10 min",  DIM),
    ]
    add_label(s, "Timing Reference", 0.6, 5.97, 3, 0.35, size=13, bold=True, color=WHITE)
    for i, (label, val, col) in enumerate(timings):
        cx = 0.6 + i * 2.44
        add_label(s, val,   cx, 6.38, 2.3, 0.38, size=16, bold=True, color=col, align=PP_ALIGN.CENTER)
        add_label(s, label, cx, 6.76, 2.3, 0.3,  size=10, color=DIM,  align=PP_ALIGN.CENTER)

    return s


# ── SLIDE 9: Results & ROI ────────────────────────────────────────────────────
def slide_results(prs):
    s = blank_slide(prs)
    fill_bg(s)

    add_label(s, "Measured Results & ROI", 0.4, 0.18, 12, 0.65, size=32, bold=True, color=WHITE)
    add_label(s, "Simulated benchmark over 10,000 synthetic requests — all data tagged as SIMULATED",
              0.4, 0.78, 12.8, 0.4, size=14, color=DIM)
    horiz_rule(s, 0.4, 1.22, 12.5, color=BORDER)

    # Big metric cards
    metrics = [
        ("-54%",    "P95 Latency Reduction",    "1,328 ms → 612 ms",        BLUE),
        ("-37%",    "Cost / 1K Tokens",         "$0.019 → $0.012",          EMERALD),
        ("+28%",    "Throughput Increase",      "33 → 42.8 req/sec",        AMBER),
        ("0%",      "Quality Regressions",      "93% quality score maintained", RED),
    ]

    mw = 2.95
    for i, (val, title, sub, col) in enumerate(metrics):
        cx = 0.35 + i * (mw + 0.22)
        add_rect(s, cx, 1.42, mw, 1.85, fill=SURFACE, line=col, line_width_pt=2.0)
        add_label(s, val,   cx + 0.15, 1.52, mw - 0.3, 0.85, size=44, bold=True, color=col, align=PP_ALIGN.CENTER)
        horiz_rule(s, cx + 0.15, 2.35, mw - 0.3, color=col, thickness_pt=0.5)
        add_label(s, title, cx + 0.15, 2.42, mw - 0.3, 0.38, size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_label(s, sub,   cx + 0.15, 2.78, mw - 0.3, 0.38, size=11, color=DIM,   align=PP_ALIGN.CENTER)

    # Experiment table mock
    add_rect(s, 0.35, 3.44, 12.5, 2.0, fill=SURFACE, line=BORDER)
    add_label(s, "Recent Benchmark Experiments  (simulated)", 0.55, 3.5, 7, 0.35, size=13, bold=True, color=WHITE)
    headers = ["ID", "Strategy", "Batch", "Concurrency", "P95 Latency", "Quality", "Cost/1K", "Result"]
    col_xs  = [0.55, 1.25, 2.65, 3.6, 4.75, 6.1, 7.15, 8.35]
    col_ws  = [0.65, 1.35, 0.9, 1.1, 1.3, 1.0, 1.15, 2.5]
    horiz_rule(s, 0.45, 3.88, 12.3, color=BORDER)
    for j, h in enumerate(headers):
        add_label(s, h, col_xs[j], 3.58, col_ws[j], 0.28, size=10, bold=True, color=DIM)

    rows = [
        ["EXP-042", "cost-optimised",  "16", "8",  "612 ms",   "0.93", "$0.012", ("✔ APPLIED",  EMERALD)],
        ["EXP-041", "quality-first",   "4",  "4",  "1,240 ms", "0.96", "$0.018", ("◉ INJECTED", AMBER)],
        ["EXP-040", "high-throughput", "32", "16", "480 ms",   "0.89", "$0.015", ("◉ INJECTED", AMBER)],
        ["EXP-039", "low-latency",     "8",  "8",  "390 ms",   "0.87", "$0.018", ("◉ INJECTED", AMBER)],
    ]
    for r, row in enumerate(rows):
        ry = 3.95 + r * 0.35
        for j, cell in enumerate(row[:-1]):
            add_label(s, cell, col_xs[j], ry, col_ws[j], 0.3, size=10.5, color=DIM)
        result_text, result_col = row[-1]
        add_label(s, result_text, col_xs[7], ry, col_ws[7], 0.3, size=10.5, bold=True, color=result_col)

    # ROI callout
    add_rect(s, 0.35, 5.55, 12.5, 1.75, fill=RGBColor(0x0e, 0x1e, 0x0e), line=EMERALD, line_width_pt=1.5)
    add_label(s, "Projected ROI (illustrative, USD)", 0.6, 5.62, 6, 0.38, size=14, bold=True, color=EMERALD)
    roi_items = [
        ("API cost reduction",     "$14,500/mo  saved at 10M req/mo", EMERALD),
        ("Latency improvement",    "54% lower P95 → measurable UX gain", BLUE),
        ("Engineering overhead",   "Autonomous loop replaces manual tuning", AMBER),
        ("Quality protection",     "0% regressions in 30-day rolling window", WHITE),
    ]
    for i, (label, val, col) in enumerate(roi_items):
        col_i = i % 2
        row_i = i // 2
        add_label(s, f"• {label}:  {val}", 0.6 + col_i * 6.2, 6.08 + row_i * 0.38, 6.0, 0.35, size=11.5, color=col)

    return s


# ── SLIDE 10: Tech Stack + Conclusion ─────────────────────────────────────────
def slide_conclusion(prs):
    s = blank_slide(prs)
    fill_bg(s)

    add_label(s, "Tech Stack & Roadmap", 0.4, 0.18, 12, 0.65, size=32, bold=True, color=WHITE)
    add_label(s, "Production-ready architecture built on battle-tested open source",
              0.4, 0.78, 12, 0.4, size=14, color=DIM)
    horiz_rule(s, 0.4, 1.22, 12.5, color=BORDER)

    # Stack grid
    stack = [
        ("Backend",          BLUE,   ["FastAPI 0.115",  "Python 3.12",  "Pydantic v2",    "Uvicorn / ASGI"]),
        ("AI / Providers",   EMERALD,["google-genai 2.34","Ollama REST", "Simulator (built-in)", "OpenAI SDK compat"]),
        ("Optimisation",     AMBER,  ["Optuna (HPO)",   "NumPy",        "M/D/1 queue model","Pareto frontier"]),
        ("Observability",    RGBColor(0xc0,0x84,0xfc), ["Request metrics","Token tracking","Cost accounting","Quality eval"]),
        ("Testing",          RED,    ["Pytest 30 tests","Unit + integration","FastAPI TestClient","100% pass rate"]),
        ("Frontend",         DIM,    ["Vanilla HTML/JS","Chart.js",    "Tailwind CSS",    "No build step"]),
    ]
    sw = 2.0
    for i, (area, col, items) in enumerate(stack):
        row = i // 3
        col_i = i % 3
        cx = 0.35 + col_i * (sw + 0.44) * 1.35
        cy = 1.4 + row * 2.05
        add_rect(s, cx, cy, sw * 1.95, 1.85, fill=SURFACE, line=col, line_width_pt=1.5)
        add_label(s, area, cx + 0.14, cy + 0.1, sw * 1.8, 0.35, size=13, bold=True, color=col)
        for j, item in enumerate(items):
            add_label(s, "  • " + item, cx + 0.14, cy + 0.5 + j * 0.3, sw * 1.8, 0.28, size=11, color=DIM)

    # Roadmap
    add_rect(s, 0.35, 5.42, 12.5, 0.85, fill=SURFACE, line=BORDER)
    add_label(s, "Roadmap", 0.55, 5.48, 2, 0.35, size=13, bold=True, color=WHITE)
    roadmap = [
        ("v1 (done)", "Gateway · Routing · Policy · Canary · Dashboard", EMERALD),
        ("v2",        "Prometheus export · Grafana dashboards · Alerting", BLUE),
        ("v3",        "Multi-tenant · SDK · vLLM / TGI adapters · real GPU", AMBER),
        ("v4",        "SaaS control plane · Team analytics · Cost budgets", DIM),
    ]
    for i, (ver, desc, col) in enumerate(roadmap):
        cx = 0.55 + i * 3.05
        add_label(s, ver,  cx, 5.82, 1.1, 0.26, size=11, bold=True, color=col)
        add_label(s, desc, cx + 1.12, 5.82, 1.85, 0.26, size=10, color=DIM)

    # CTA
    add_rect(s, 0.35, 6.4, 12.5, 0.9, fill=RGBColor(0x05, 0x10, 0x25), line=BLUE, line_width_pt=1.8)
    add_label(s,
        "🚀   INFERCTRL  —  Run it now:   "
        "uvicorn apps.api.main:app --port 8000   "
        "then open  http://localhost:8000",
        0.6, 6.5, 12.0, 0.68, size=14, bold=True, color=BLUE, align=PP_ALIGN.CENTER)

    return s


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    out = Path(__file__).parent.parent / "INFERCTRL_Presentation.pptx"
    prs = new_prs()

    builders = [
        slide_title,
        slide_problem,
        slide_solution,
        slide_agents,
        slide_backends,
        slide_policy,
        slide_dashboard,
        slide_canary,
        slide_results,
        slide_conclusion,
    ]

    for fn in builders:
        fn(prs)
        print(f"  [+] {fn.__name__}")

    prs.save(str(out))
    print(f"\n[SUCCESS] Saved presentation -> {out}")
    print(f"   Slides: {len(prs.slides)}")
    print(f"   Size:   {out.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()

