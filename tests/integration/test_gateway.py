"""Integration tests for OpenAI-compatible Gateway endpoints."""

import pytest
from fastapi.testclient import TestClient
from apps.api.main import app

client = TestClient(app)


def test_openai_chat_completions_auto():
    payload = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "Explain Redis in-memory caching"}
        ],
        "temperature": 0.5,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert "Redis" in data["choices"][0]["message"]["content"]
    assert "usage" in data
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0

    # Verify INFERCTRL metadata
    assert "inferctrl" in data
    meta = data["inferctrl"]
    assert meta["routing_strategy"] == "balanced"
    assert meta["ttft_ms"] > 0
    assert meta["total_latency_ms"] >= meta["ttft_ms"]
    assert meta["estimated_cost_usd"] > 0
    assert meta["measurement_source"] in ["simulated", "actual"]


def test_openai_chat_completions_with_strategy_override():
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Why is latency critical?"}],
        "routing_strategy": "lowest_cost",
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["inferctrl"]["routing_strategy"] == "lowest_cost"


def test_openai_models_endpoint():
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 3
    model_ids = [m["id"] for m in data["data"]]
    assert any("sim-llama-3-8b" in m for m in model_ids)


def test_comprehensive_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "providers" in data
    assert "simulator" in data["providers"]
    assert data["providers"]["simulator"]["healthy"] is True


def test_observability_requests_endpoint():
    # Make a completion first
    client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "test telemetry"}]},
    )

    res = client.get("/api/v1/requests?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert "summary" in data
    assert "requests" in data
    assert len(data["requests"]) > 0
    latest = data["requests"][0]
    assert "request_id" in latest
    assert "total_latency_ms" in latest
    assert "estimated_cost_usd" in latest


def test_routing_config_crud():
    # 1. Get current config
    res = client.get("/api/v1/routing/config")
    assert res.status_code == 200
    initial_strategy = res.json()["strategy"]

    # 2. Update config
    update_res = client.post(
        "/api/v1/routing/config",
        json={"strategy": "lowest_cost", "max_latency_ms": 3500.0},
    )
    assert update_res.status_code == 200
    assert update_res.json()["config"]["strategy"] == "lowest_cost"
    assert update_res.json()["config"]["constraints"]["max_latency_ms"] == 3500.0

    # 3. Restore strategy
    client.post("/api/v1/routing/config", json={"strategy": initial_strategy})


def test_chatbot_html_page():
    res = client.get("/chat")
    assert res.status_code == 200
    assert "INFERCTRL CHATBOT" in res.text

