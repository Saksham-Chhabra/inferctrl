# INFERCTRL — Autonomous LLM Inference Optimizer

> **Closed-Loop Autonomous Control Plane for LLM Inference Clusters**  
> Optimizes the Trilemma of LLM Inference: **P95 Latency vs Cost vs Quality** under strict deterministic Service-Level Objectives (SLOs) and automated canary rollbacks.

---

## 1. Overview & System Architecture

INFERCTRL continuously observes LLM serving workloads, detects regime shifts, proposes candidate configurations (model routing mix, batch wait window, max batch size, concurrency), benchmarks candidates on replay traces, evaluates response quality against golden task suites, and safely promotes candidates via canaries with instant rollback on regressions.

```text
                  ┌─────────────────────────────────────┐
                  │ Web Console (High-Contrast Room)    │
                  │ Live Performance · Routing · Pareto │
                  └──────────────────┬──────────────────┘
                                     │ REST / WebSocket
                  ┌──────────────────▼──────────────────┐
                  │ FastAPI Control Plane (port 8000)   │
                  │ State Machine · Guardrails · API    │
                  └──────────────┬─────────────┬─────────┘
                                 │             │
                       ┌─────────▼─────┐ ┌─────▼──────────┐
                       │ Agent Runtime │ │ Telemetry API  │
                       │ State Machine │ │ Synthetic Load │
                       └──────┬────────┘ └─────┬──────────┘
                              │                │
       ┌──────────────────────┼────────────────┼─────────────────────┐
       │                      │                │                     │
┌──────▼────────┐  ┌──────────▼─────┐  ┌───────▼────────┐  ┌─────────▼──────┐
│ Workload      │  │ Experiment     │  │ Quality        │  │ Safety &       │
│ Analyst Agent │  │ Orchestrator   │  │ Evaluator      │  │ Policy Engine  │
└──────┬────────┘  └──────────┬─────┘  └───────┬────────┘  └─────────┬──────┘
       └──────────────────────┼────────────────┼─────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Serving Adapter    │
                    │ Discrete Simulator │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Candidate Configs  │
                    │ Baseline · Canary  │
                    └────────────────────┘
```

---

## 2. Key Features

- **Multi-Agent Optimization Loop:**
  - **Workload Analyst Agent:** CUSUM drift detection, statistical burst classification, and feature extraction.
  - **Scheduler Agent:** Multi-knob configuration proposals (batch size, wait time, model mix) with explicit hypotheses.
  - **Benchmark Agent:** Multi-trial warm-up execution and percentile latency isolation (TTFT, ITL, Total).
  - **Quality Evaluator Agent:** Golden test suite across 5 task domains (Code, Reasoning, JSON, Summary, Chat) enforcing sub-1.0% regression limits.
  - **Controller Agent:** Formal state machine coordinating observation, benchmarking, decision, canarying, and rollbacks.
- **Deterministic Safety & Policy Engine:**
  - Non-LLM final authority enforcing hard SLO boundaries.
  - Multi-objective Pareto frontier scoring ($w_{\text{lat}} \cdot L + w_{\text{cost}} \cdot C + w_{\text{err}} \cdot E$).
- **Dual-Backend Support:**
  - **Backend A (Discrete-Event Simulator):** 100% reproducible, zero-GPU required for CI and laptop demos.
  - **Backend B (vLLM Adapter):** Real hardware OpenAI-compatible endpoint interface.
- **High-Contrast Control-Room UI:**
  - Real-time time-series performance charts.
  - Interactive Request Flow & Dynamic Model Routing diagram.
  - 2D Cost vs. Quality Pareto Frontier scatter plot.
  - Optimization decision timeline and audit log trail.
  - Instant action buttons: Traffic Surge injection, Closed-Loop optimization, and Fault Injection Rollback.

---

## 3. Quick Start & Execution

### Prerequisites
- Python 3.11+
- pip

### Setup
```bash
# Clone or navigate to the project root
cd C:\Users\schha\.gemini\antigravity\scratch\inferctrl

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 1. Launch the Web Console & API Server
```bash
uvicorn apps.api.main:app --reload --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser to view the live dashboard.

### 2. Run the Automated 3-Minute Hackathon Demo Script
```bash
python -m inferctrl.demo
```
This runs the complete 10-step scripted presentation:
1. Baseline setup in simulated mode.
2. Nominal load benchmarking.
3. Injected traffic surge (+2.1x burst).
4. Workload Analyst drift detection.
5. Multi-candidate benchmarking.
6. Guardrail rejection of unsafe / low-quality candidates.
7. Promotion of best candidate to 5% Canary.
8. Simulated GPU throttle fault injection.
9. Automated watchdog rollback within milliseconds.
10. Immutable audit log report export.

### 3. Run the Test Suite
```bash
pytest -v
```
Runs 16 unit and integration tests covering schemas, simulator reproducibility, policy engine guardrails, closed-loop state transitions, canary rollbacks, and FastAPI endpoints.

---

## 4. Repository Structure

```text
inferctrl/
├── apps/
│   ├── api/                      # FastAPI control plane routes & event streaming
│   │   └── main.py
│   └── web/                      # High-contrast control room dashboard
│       └── index.html
├── inferctrl/
│   ├── agents/                   # Typed agent implementations
│   │   ├── workload.py           # Workload Analyst Agent (Drift detection)
│   │   ├── scheduler.py          # Scheduler Agent (Candidate generation)
│   │   └── benchmark.py          # Benchmark Agent (Trial orchestration)
│   ├── controller/
│   │   └── state_machine.py      # Autonomous closed-loop Controller Agent
│   ├── policy/
│   │   └── engine.py             # Deterministic Safety & Policy Engine
│   ├── quality/
│   │   └── golden_dataset.py     # Golden dataset evaluator & regression checks
│   ├── serving/
│   │   ├── base.py               # Abstract ServingAdapter protocol
│   │   └── simulator.py          # Discrete-event serving simulator
│   ├── telemetry/
│   │   └── generator.py          # Deterministic trace generator
│   ├── schemas/                  # Pydantic v2 data contracts
│   │   ├── configs.py
│   │   ├── decisions.py
│   │   ├── experiments.py
│   │   └── traces.py
│   └── demo.py                   # 10-step hackathon pitch demo script
├── configs/
│   ├── models.yaml               # Model profiles (Llama 3 8B vs 70B)
│   ├── policies.yaml             # SLO thresholds & guardrail rules
│   └── simulator.yaml            # Hardware pool & scenario parameters
├── tests/
│   ├── unit/                     # Unit test suites (11 tests)
│   └── integration/              # FastAPI integration test suite (5 tests)
├── pyproject.toml
├── requirements.txt
└── README.md
```

