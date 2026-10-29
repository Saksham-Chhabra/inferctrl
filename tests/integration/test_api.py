"""Integration tests for FastAPI control plane endpoints."""

import pytest
from fastapi.testclient import TestClient
from apps.api.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["backend"] == "deterministic_simulator"


def test_overview_endpoint():
    res = client.get("/api/v1/overview")
    assert res.status_code == 200
    data = res.json()
    assert "metrics" in data
    assert "active_configuration" in data
    assert data["metrics"]["gpu_count"] == 4


def test_trigger_optimization_run():
    payload = {"scenario": "bursty_spike", "request_count": 80, "queue_depth": 15}
    res = client.post("/api/v1/optimization-runs", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "run_id" in data
    assert "decision" in data
    assert data["workload_regime"] in ["bursty", "steady_short", "low_traffic"]

    # Verify run inspection endpoint
    run_id = data["run_id"]
    res_run = client.get(f"/api/v1/optimization-runs/{run_id}")
    assert res_run.status_code == 200
    run_data = res_run.json()
    assert len(run_data["candidates"]) > 0


def test_canary_and_rollback_flow():
    # 1. Trigger run
    client.post("/api/v1/optimization-runs", json={"scenario": "bursty_spike", "request_count": 80})

    # 2. Approve candidate
    res_app = client.post("/api/v1/candidates/cand-test/approve", json={"traffic_pct": 5.0})
    if res_app.status_code == 200:
        canary = res_app.json()["canary"]
        canary_id = canary["canary_id"]

        # 3. Rollback
        res_rb = client.post(f"/api/v1/canaries/{canary_id}/rollback", json={"reason": "Test regression rollback"})
        assert res_rb.status_code == 200
        assert res_rb.json()["status"] == "rolled_back"


def test_audit_events_endpoint():
    res = client.get("/api/v1/audit-events")
    assert res.status_code == 200
    events = res.json()
    assert isinstance(events, list)
    assert len(events) > 0

