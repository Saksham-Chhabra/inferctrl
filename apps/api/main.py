"""FastAPI Control Plane and Real-Time Event Bus for INFERCTRL."""

import asyncio
import json
import os
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from inferctrl.controller.state_machine import ControllerAgent, OptimizationLoopResult
from inferctrl.cost.calculator import CostCalculator
from inferctrl.gateway.models import ChatCompletionRequest, ChatCompletionResponse
from inferctrl.gateway.service import GatewayService
from inferctrl.observability.metrics import ObservabilityRegistry
from inferctrl.policy.engine import PolicyEngine
from inferctrl.providers.gemini import GeminiProviderAdapter
from inferctrl.providers.ollama import OllamaProviderAdapter
from inferctrl.providers.simulator import SimulatorProviderAdapter
from inferctrl.quality.golden_dataset import QualityEvaluator
from inferctrl.routing.engine import RoutingConstraints, RoutingEngine, RoutingStrategy
from inferctrl.schemas import (
    AuditEvent,
    CanarySession,
    CandidateProposal,
    ControllerState,
    DecisionOutcome,
    DecisionRecord,
    ServingConfig,
    WorkloadReport,
)
from inferctrl.serving.simulator import InferenceSimulator
from inferctrl.telemetry.generator import TraceGenerator

app = FastAPI(
    title="INFERCTRL Control Plane API",
    version="0.1.0",
    description="Autonomous closed-loop LLM inference optimizer control plane.",
)

# Enable CORS for frontend development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global system singleton state
class SystemState:
    def __init__(self):
        # 1. Observability, Cost & Providers
        self.cost_calculator = CostCalculator()
        self.observability = ObservabilityRegistry(capacity=1000)
        self.providers = {
            "gemini": GeminiProviderAdapter(),
            "ollama": OllamaProviderAdapter(),
            "simulator": SimulatorProviderAdapter(),
        }
        self.routing_engine = RoutingEngine(
            providers=self.providers,
            cost_calculator=self.cost_calculator,
            observability=self.observability,
            default_strategy=RoutingStrategy.BALANCED,
            constraints=RoutingConstraints(),
        )
        self.gateway = GatewayService(
            providers=self.providers,
            routing_engine=self.routing_engine,
            cost_calculator=self.cost_calculator,
            observability=self.observability,
        )

        # 2. Simulator & Closed-Loop Controller
        self.sim = InferenceSimulator()
        self.policy = PolicyEngine(
            target_p95_latency_ms=1000.0,
            max_error_rate=0.02,
            max_quality_regression_pct=1.0,
        )
        self.quality_evaluator = QualityEvaluator()
        self.baseline_config = ServingConfig(
            config_id="cfg-default",
            max_concurrency=8,
            max_batch_size=16,
            batch_wait_ms=15.0,
            routing_mix={"llama-3-8b": 0.70, "llama-3-70b": 0.30},
        )
        self.controller = ControllerAgent(
            serving_adapter=self.sim,
            policy_engine=self.policy,
            quality_evaluator=self.quality_evaluator,
            baseline_config=self.baseline_config,
        )
        self.runs: Dict[str, OptimizationLoopResult] = {}
        self.active_run_id: Optional[str] = None
        self.latest_loop_result: Optional[OptimizationLoopResult] = None
        self.connected_websockets: List[WebSocket] = []


state = SystemState()


# Broadcast helper
async def broadcast_event(event_type: str, payload: dict):
    message = json.dumps({"type": event_type, "data": payload}, default=str)
    for ws in list(state.connected_websockets):
        try:
            await ws.send_text(message)
        except Exception:
            state.connected_websockets.remove(ws)


# --- REST API Endpoints ---


@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    """Serve the Web Console directly from root."""
    html_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "web", "index.html"))
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>INFERCTRL Control Plane</h1>"


@app.get("/presentation", response_class=HTMLResponse)
def get_presentation():
    """Serve the Presentation Pitch Deck."""
    html_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "web", "presentation.html"))
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>INFERCTRL Presentation Deck</h1>"


@app.get("/chat", response_class=HTMLResponse)
def get_chatbot_ui():
    """Serve the Demonstration Chatbot UI."""
    html_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "web", "chatbot.html"))
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>INFERCTRL Chatbot</h1>"


# --- OpenAI-Compatible Gateway Endpoints ---


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible inference gateway endpoint.
    Routes to Gemini, Ollama, or Simulator according to active optimization policy.
    """
    try:
        response = await state.gateway.execute_chat_completion(req)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"INFERCTRL Gateway Error: {str(e)}")


@app.get("/v1/models")
def get_openai_models():
    """OpenAI-compatible models catalog with live availability and pricing metadata."""
    models_list = state.gateway.list_all_models()
    return {
        "object": "list",
        "data": [
            {
                "id": m["id"],
                "object": "model",
                "created": 1730000000,
                "owned_by": m["provider"],
                "inferctrl_metadata": m,
            }
            for m in models_list
        ],
    }


@app.get("/health")
async def get_gateway_health():
    """Comprehensive gateway & provider health endpoint."""
    provider_healths = {}
    for p_name, adapter in state.providers.items():
        try:
            h = await adapter.health_check()
            provider_healths[p_name] = {
                "healthy": h.healthy,
                "latency_ms": h.latency_ms,
                "message": h.message,
                "available_models": h.available_models,
            }
        except Exception as e:
            provider_healths[p_name] = {"healthy": False, "message": str(e)}

    return {
        "status": "healthy",
        "gateway": "online",
        "active_routing_strategy": state.routing_engine.default_strategy.value,
        "providers": provider_healths,
    }


@app.get("/api/v1/requests")
def get_recent_requests(limit: int = 50, provider: Optional[str] = None, status: Optional[str] = None):
    """Query recent request observability logs and aggregate statistics."""
    records = state.observability.get_recent(limit=limit, provider=provider, status=status)
    summary = state.observability.compute_summary(window_seconds=300.0)
    return {
        "summary": summary,
        "requests": [r.model_dump() for r in records],
    }


class UpdateRoutingConfigRequest(BaseModel):
    strategy: Optional[str] = None
    max_latency_ms: Optional[float] = None
    max_estimated_cost_per_request: Optional[float] = None
    min_quality_score: Optional[float] = None
    fallback_enabled: Optional[bool] = None


@app.get("/api/v1/routing/config")
def get_routing_config():
    """Get active routing strategy and policy constraints."""
    return {
        "strategy": state.routing_engine.default_strategy.value,
        "constraints": {
            "max_latency_ms": state.routing_engine.constraints.max_latency_ms,
            "max_estimated_cost_per_request": state.routing_engine.constraints.max_estimated_cost_per_request,
            "min_quality_score": state.routing_engine.constraints.min_quality_score,
            "fallback_enabled": state.routing_engine.constraints.fallback_enabled,
        },
    }


@app.post("/api/v1/routing/config")
def update_routing_config(req: UpdateRoutingConfigRequest):
    """Dynamically update routing strategy and policy constraints."""
    if req.strategy:
        try:
            state.routing_engine.default_strategy = RoutingStrategy(req.strategy.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid strategy: '{req.strategy}'")

    if req.max_latency_ms is not None:
        state.routing_engine.constraints.max_latency_ms = req.max_latency_ms
    if req.max_estimated_cost_per_request is not None:
        state.routing_engine.constraints.max_estimated_cost_per_request = req.max_estimated_cost_per_request
    if req.min_quality_score is not None:
        state.routing_engine.constraints.min_quality_score = req.min_quality_score
    if req.fallback_enabled is not None:
        state.routing_engine.constraints.fallback_enabled = req.fallback_enabled

    return {"status": "updated", "config": get_routing_config()}


@app.get("/api/v1/health")
def get_health():
    return {
        "status": "healthy",
        "backend": "deterministic_simulator",
        "controller_state": state.controller.state.value,
        "active_config_id": state.controller.active_config.config_id,
    }


@app.get("/api/v1/overview")
def get_overview():
    latest = state.latest_loop_result
    active_cfg = state.controller.active_config

    if latest is not None:
        b_res = latest.baseline_result
        rps = round(b_res.throughput_req_per_sec, 1)
        p95 = round(b_res.latency.total_p95_ms, 1)
        cost_1k = round(b_res.estimated_cost_per_1k_tokens, 4)
        gpu_util = b_res.gpu_utilization_pct
        quality = 0.93
        if latest.recommended_candidate:
            # Show deltas against recommended candidate
            p95_delta = latest.decision.latency_delta_pct
            cost_delta = latest.decision.cost_delta_pct
            qual_delta = latest.decision.quality_delta_pct
        else:
            p95_delta, cost_delta, qual_delta = 0.0, 0.0, 0.0
    else:
        rps = 42.8
        p95 = 612.0
        cost_1k = 0.012
        gpu_util = 78.0
        quality = 0.93
        p95_delta, cost_delta, qual_delta = -54.0, -37.0, 2.0

    return {
        "system_status": "optimizing" if state.controller.state != ControllerState.IDLE else "idle",
        "controller_state": state.controller.state.value,
        "metrics": {
            "requests_per_sec": rps,
            "requests_per_sec_delta": 68.0,
            "p95_latency_ms": p95,
            "p95_latency_delta_pct": p95_delta,
            "quality_score": quality,
            "quality_score_delta_pct": qual_delta,
            "cost_per_1k_tokens": cost_1k,
            "cost_delta_pct": cost_delta,
            "gpu_utilization_pct": gpu_util,
            "gpu_count": 4,
        },
        "active_configuration": {
            "config_id": active_cfg.config_id,
            "routing_mix": active_cfg.routing_mix,
            "max_batch_size": active_cfg.max_batch_size,
            "batch_wait_ms": active_cfg.batch_wait_ms,
            "max_concurrency": active_cfg.max_concurrency,
            "policy": "Cost-aware (balanced)",
        },
        "active_canary": state.controller.active_canary.model_dump() if state.controller.active_canary else None,
    }


@app.get("/api/v1/models")
def get_models():
    return [
        {
            "id": "llama-3-8b",
            "name": "Llama 3 8B Instruct",
            "category": "fast",
            "traffic_share_pct": state.controller.active_config.routing_mix.get("llama-3-8b", 0.70) * 100.0,
            "cost_per_1k": 0.0005,
            "quality_score": 0.88,
        },
        {
            "id": "llama-3-70b",
            "name": "Llama 3 70B Instruct",
            "category": "quality",
            "traffic_share_pct": state.controller.active_config.routing_mix.get("llama-3-70b", 0.30) * 100.0,
            "cost_per_1k": 0.0040,
            "quality_score": 0.96,
        },
    ]


class OptimizationTriggerRequest(BaseModel):
    scenario: str = "bursty_spike"
    request_count: int = 100
    queue_depth: int = 15


@app.post("/api/v1/optimization-runs")
async def trigger_optimization_run(req: OptimizationTriggerRequest, background_tasks: BackgroundTasks):
    traces = TraceGenerator.generate_trace(scenario=req.scenario, request_count=req.request_count, random_seed=42)

    # Execute closed loop synchronously for immediate response or async
    result = state.controller.run_closed_loop(traces, current_queue_depth=req.queue_depth)
    state.runs[result.run_id] = result
    state.latest_loop_result = result
    state.active_run_id = result.run_id

    # Broadcast optimization completed event
    background_tasks.add_task(
        broadcast_event,
        "optimization_completed",
        {
            "run_id": result.run_id,
            "decision": result.decision.model_dump(),
            "recommended_candidate": result.recommended_candidate.model_dump() if result.recommended_candidate else None,
        },
    )

    return {
        "run_id": result.run_id,
        "workload_regime": result.workload_report.regime,
        "recommended_candidate": result.recommended_candidate.candidate_id if result.recommended_candidate else None,
        "decision": result.decision.model_dump(),
    }


@app.get("/api/v1/optimization-runs/{run_id}")
def get_optimization_run(run_id: str):
    if run_id not in state.runs:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    res = state.runs[run_id]

    candidates_payload = []
    for prop, b_res, q_res, passed, viol in res.candidate_results:
        candidates_payload.append(
            {
                "candidate_id": prop.candidate_id,
                "hypothesis": prop.hypothesis,
                "targeted_knob": prop.targeted_knob,
                "config": prop.config.model_dump(),
                "p95_latency_ms": round(b_res.latency.total_p95_ms, 1),
                "quality_score": round(q_res.overall_score, 3),
                "cost_per_1k": round(b_res.estimated_cost_per_1k_tokens, 4),
                "passed_guardrails": passed,
                "violations": viol,
            }
        )

    return {
        "run_id": res.run_id,
        "workload_report": res.workload_report.model_dump(),
        "baseline": {
            "p95_latency_ms": round(res.baseline_result.latency.total_p95_ms, 1),
            "cost_per_1k": round(res.baseline_result.estimated_cost_per_1k_tokens, 4),
        },
        "candidates": candidates_payload,
        "decision": res.decision.model_dump(),
    }


class ApproveCandidateRequest(BaseModel):
    traffic_pct: float = 5.0


@app.post("/api/v1/candidates/{candidate_id}/approve")
async def approve_candidate(candidate_id: str, req: ApproveCandidateRequest):
    latest = state.latest_loop_result
    if not latest or not latest.recommended_candidate:
        raise HTTPException(status_code=400, detail="No active recommended candidate to approve")

    canary = state.controller.start_canary(latest.recommended_candidate, traffic_pct=req.traffic_pct)
    await broadcast_event("canary_started", canary.model_dump())
    return {"status": "canary_started", "canary": canary.model_dump()}


class RollbackRequest(BaseModel):
    reason: str = "Manual operator rollback"


@app.post("/api/v1/canaries/{canary_id}/rollback")
async def rollback_canary(canary_id: str, req: RollbackRequest):
    success = state.controller.trigger_rollback(req.reason)
    await broadcast_event("canary_rolled_back", {"canary_id": canary_id, "reason": req.reason})
    return {"status": "rolled_back", "success": success}


@app.get("/api/v1/audit-events")
def get_audit_events():
    return [e.model_dump() for e in reversed(state.controller.audit_events)]


# --- WebSocket Streaming ---


@app.websocket("/api/v1/events")
async def websocket_events(websocket: WebSocket):
    await websocket.accept()
    state.connected_websockets.append(websocket)
    try:
        while True:
            # Echo keepalive ping or receive client messages
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in state.connected_websockets:
            state.connected_websockets.remove(websocket)
