"""Unit tests for Pydantic schemas and validation."""

import pytest
from pydantic import ValidationError
from inferctrl.schemas import (
    RequestTrace,
    TaskType,
    WorkloadFeatures,
    ServingConfig,
    CandidateProposal,
    BenchmarkResult,
    LatencyMetrics,
    QualityScore,
)


def test_request_trace_valid():
    trace = RequestTrace(
        request_id="req-001",
        arrival_time_ms=100.5,
        task_type=TaskType.CODE,
        prompt_tokens=350,
        output_tokens=120,
    )
    assert trace.request_id == "req-001"
    assert trace.prompt_tokens == 350
    assert trace.task_type == TaskType.CODE


def test_serving_config_routing_validation():
    # Valid routing
    config = ServingConfig(
        config_id="cfg-test",
        max_concurrency=16,
        max_batch_size=32,
        batch_wait_ms=25.0,
        routing_mix={"llama-3-8b": 0.8, "llama-3-70b": 0.2},
    )
    assert config.max_concurrency == 16

    # Invalid routing (sum != 1.0)
    with pytest.raises(ValidationError):
        ServingConfig(
            config_id="cfg-invalid",
            routing_mix={"llama-3-8b": 0.5, "llama-3-70b": 0.2},
        )


def test_quality_score_calculation():
    score = QualityScore(
        candidate_id="cand-01",
        overall_score=0.93,
        baseline_score=0.94,
        score_delta=-0.01,
        task_scores={"code": 0.92, "reasoning": 0.91},
        passed_guardrails=True,
    )
    assert score.score_delta == -0.01
    assert score.passed_guardrails is True

