"""Unit tests for Safety & Policy Engine and Quality Evaluator (Exit Gate 2)."""

import pytest
from inferctrl.policy.engine import PolicyEngine
from inferctrl.quality.golden_dataset import QualityEvaluator
from inferctrl.schemas import (
    BenchmarkResult,
    CandidateProposal,
    LatencyMetrics,
    QualityScore,
    ServingConfig,
)


@pytest.fixture
def policy_engine():
    return PolicyEngine(
        target_p95_latency_ms=1000.0,
        max_error_rate=0.02,
        max_quality_regression_pct=1.0,
    )


@pytest.fixture
def dummy_latency():
    return LatencyMetrics(
        queue_time_p50_ms=20.0,
        queue_time_p95_ms=50.0,
        ttft_p50_ms=100.0,
        ttft_p95_ms=250.0,
        itl_p50_ms=15.0,
        itl_p95_ms=30.0,
        total_p50_ms=350.0,
        total_p95_ms=650.0,
        total_p99_ms=900.0,
    )


def test_rejection_on_latency_slo_breach(policy_engine, dummy_latency):
    """Candidate with P95 latency > 1000ms must be rejected."""
    bad_latency = dummy_latency.model_copy()
    bad_latency.total_p95_ms = 1250.0  # Exceeds 1000.0ms SLO

    bench_res = BenchmarkResult(
        candidate_id="cand-slow",
        run_id="run-1",
        trace_id="tr-1",
        trials_count=1,
        total_requests=100,
        successful_requests=100,
        failed_requests=0,
        error_rate=0.0,
        latency=bad_latency,
        throughput_req_per_sec=30.0,
        tokens_per_second=500.0,
        estimated_cost_per_request=0.001,
        estimated_cost_per_1k_tokens=0.002,
        gpu_utilization_pct=75.0,
    )

    quality = QualityScore(
        candidate_id="cand-slow",
        overall_score=0.95,
        baseline_score=0.94,
        score_delta=1.0,
        passed_guardrails=True,
    )

    passed, violations = policy_engine.evaluate_guardrails(bench_res, quality)
    assert passed is False
    assert any("exceeds target SLO" in v for v in violations)


def test_rejection_on_quality_regression(policy_engine, dummy_latency):
    """Candidate that improves latency but drops quality beyond budget must be rejected."""
    bench_res = BenchmarkResult(
        candidate_id="cand-low-qual",
        run_id="run-1",
        trace_id="tr-1",
        trials_count=1,
        total_requests=100,
        successful_requests=100,
        failed_requests=0,
        error_rate=0.0,
        latency=dummy_latency,  # Fast!
        throughput_req_per_sec=50.0,
        tokens_per_second=900.0,
        estimated_cost_per_request=0.0005,
        estimated_cost_per_1k_tokens=0.001,
        gpu_utilization_pct=60.0,
    )

    # 100% 8B model will cause overall quality to drop from 0.94 to ~0.87 (> 1.0% regression)
    evaluator = QualityEvaluator()
    all_8b_config = ServingConfig(
        config_id="cfg-all-8b",
        routing_mix={"llama-3-8b": 1.0, "llama-3-70b": 0.0},
    )
    quality = evaluator.evaluate_config(all_8b_config, baseline_score=0.94, max_allowed_regression_pct=1.0)

    assert quality.passed_guardrails is False

    passed, violations = policy_engine.evaluate_guardrails(bench_res, quality)
    assert passed is False
    assert any("Quality guardrail failed" in v for v in violations)


def test_pareto_frontier_extraction(dummy_latency):
    """Verify that non-dominated candidates are properly identified."""
    res_a = BenchmarkResult(
        candidate_id="A",
        run_id="r",
        trace_id="t",
        trials_count=1,
        total_requests=10,
        successful_requests=10,
        failed_requests=0,
        error_rate=0.0,
        latency=dummy_latency.model_copy(update={"total_p95_ms": 500.0}),
        throughput_req_per_sec=20.0,
        tokens_per_second=200.0,
        estimated_cost_per_request=0.001,
        estimated_cost_per_1k_tokens=0.001,
        gpu_utilization_pct=50.0,
    )
    qual_a = QualityScore(candidate_id="A", overall_score=0.90, baseline_score=0.90, score_delta=0.0)

    # B has lower latency (400ms) and equal quality and cost -> dominates A
    res_b = res_a.model_copy(update={"candidate_id": "B", "latency": dummy_latency.model_copy(update={"total_p95_ms": 400.0})})
    qual_b = QualityScore(candidate_id="B", overall_score=0.90, baseline_score=0.90, score_delta=0.0)

    # C has higher latency (800ms) but higher quality (0.96) -> Pareto non-dominated
    res_c = res_a.model_copy(update={"candidate_id": "C", "latency": dummy_latency.model_copy(update={"total_p95_ms": 800.0})})
    qual_c = QualityScore(candidate_id="C", overall_score=0.96, baseline_score=0.90, score_delta=6.0)

    candidates = [(res_a, qual_a), (res_b, qual_b), (res_c, qual_c)]
    pareto = PolicyEngine.extract_pareto_frontier(candidates)

    pareto_ids = [res.candidate_id for res, _ in pareto]
    assert "B" in pareto_ids
    assert "C" in pareto_ids
    assert "A" not in pareto_ids  # A is dominated by B

