"""Unit tests for the autonomous closed-loop Controller Agent (Exit Gate 3 & 5)."""

import pytest
from inferctrl.controller.state_machine import ControllerAgent
from inferctrl.policy.engine import PolicyEngine
from inferctrl.quality.golden_dataset import QualityEvaluator
from inferctrl.schemas import ControllerState, DecisionOutcome, ServingConfig
from inferctrl.serving.simulator import InferenceSimulator
from inferctrl.telemetry.generator import TraceGenerator


@pytest.fixture
def baseline_setup():
    traces = TraceGenerator.generate_trace(scenario="bursty_spike", request_count=100, random_seed=42)
    base_config = ServingConfig(
        config_id="cfg-initial-base",
        max_concurrency=8,
        max_batch_size=16,
        batch_wait_ms=15.0,
        routing_mix={"llama-3-8b": 0.70, "llama-3-70b": 0.30},
    )
    sim = InferenceSimulator()
    policy = PolicyEngine(target_p95_latency_ms=5000.0, max_error_rate=0.03, max_quality_regression_pct=1.0)
    quality_evaluator = QualityEvaluator()

    controller = ControllerAgent(
        serving_adapter=sim,
        policy_engine=policy,
        quality_evaluator=quality_evaluator,
        baseline_config=base_config,
    )
    return controller, traces


def test_closed_loop_execution(baseline_setup):
    """Exit Gate 3: Workload shift triggers autonomous closed loop with valid recommendation."""
    controller, traces = baseline_setup

    loop_res = controller.run_closed_loop(traces, current_queue_depth=20)

    # 1. Workload Analyst must have characterized the workload
    assert loop_res.workload_report.features.requests_per_second > 0
    assert len(loop_res.workload_report.evidence) > 0

    # 2. Rejection check: Candidate 4 (extreme 98% 8B) must have been rejected
    rejected_cands = [res for res in loop_res.candidate_results if not res[3]]
    assert len(rejected_cands) >= 1
    # Check that rejection reason specifically mentions quality
    extreme_reject = [c for c in rejected_cands if "extreme-8b" in c[0].candidate_id]
    if extreme_reject:
        assert any("Quality" in v for v in extreme_reject[0][4])

    # 3. Best candidate recommended
    assert loop_res.recommended_candidate is not None
    assert loop_res.decision.outcome == DecisionOutcome.APPROVE_RECOMMENDATION
    assert len(loop_res.decision.evidence) > 0


def test_canary_and_automatic_rollback(baseline_setup):
    """Exit Gate 5: Canary regression triggers instant automated rollback and audit log."""
    controller, traces = baseline_setup
    loop_res = controller.run_closed_loop(traces, current_queue_depth=10)

    # Start canary with approved candidate
    canary = controller.start_canary(loop_res.recommended_candidate, traffic_pct=5.0)
    assert controller.state == ControllerState.CANARYING
    assert canary.status == "active"

    # Inject regression: P95 latency spikes to 6000ms (exceeds SLO 5000ms)
    healthy, reason = controller.check_canary_health(measured_p95_ms=6200.0, measured_error_rate=0.0)
    assert healthy is False
    assert "breached SLO" in reason

    # Trigger rollback
    rollback_success = controller.trigger_rollback(reason)
    assert rollback_success is True
    assert controller.state == ControllerState.ROLLED_BACK
    assert canary.status == "rolled_back"
    assert canary.rollback_reason == reason

    # Verify audit event was logged
    audit_events = controller.audit_events
    rollback_audits = [e for e in audit_events if e.action == "rollback"]
    assert len(rollback_audits) == 1
    assert rollback_audits[0].result == "ROLLED_BACK"
