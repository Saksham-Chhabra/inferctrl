"""Unit tests for the discrete-event inference simulator (Exit Gate 1)."""

import pytest
from inferctrl.schemas import ServingConfig
from inferctrl.serving.simulator import InferenceSimulator
from inferctrl.telemetry.generator import TraceGenerator


def test_simulator_reproducibility():
    """Verify that replaying a trace with the same seed yields 100% identical metrics."""
    traces = TraceGenerator.generate_trace(scenario="steady_short", request_count=100, random_seed=42)
    config = ServingConfig(
        config_id="cfg-baseline",
        max_concurrency=8,
        max_batch_size=16,
        batch_wait_ms=15.0,
        routing_mix={"llama-3-8b": 0.70, "llama-3-70b": 0.30},
    )

    sim = InferenceSimulator()
    res1 = sim.run_benchmark(config, traces, warmup_count=20, random_seed=123)
    res2 = sim.run_benchmark(config, traces, warmup_count=20, random_seed=123)

    assert res1.latency.total_p95_ms == res2.latency.total_p95_ms
    assert res1.latency.total_p50_ms == res2.latency.total_p50_ms
    assert res1.estimated_cost_per_request == res2.estimated_cost_per_request
    assert res1.tokens_per_second == res2.tokens_per_second


def test_simulator_routing_impact():
    """Verify that shifting traffic to 70B increases quality and cost, while 8B lowers latency and cost."""
    traces = TraceGenerator.generate_trace(scenario="steady_short", request_count=100, random_seed=42)
    sim = InferenceSimulator()

    # Cheap/Fast config (90% 8B)
    cheap_cfg = ServingConfig(
        config_id="cfg-cheap",
        max_concurrency=12,
        max_batch_size=16,
        batch_wait_ms=10.0,
        routing_mix={"llama-3-8b": 0.90, "llama-3-70b": 0.10},
    )
    # Quality-heavy config (90% 70B)
    quality_cfg = ServingConfig(
        config_id="cfg-quality",
        max_concurrency=12,
        max_batch_size=16,
        batch_wait_ms=10.0,
        routing_mix={"llama-3-8b": 0.10, "llama-3-70b": 0.90},
    )

    res_cheap = sim.run_benchmark(cheap_cfg, traces, warmup_count=20, random_seed=999)
    res_quality = sim.run_benchmark(quality_cfg, traces, warmup_count=20, random_seed=999)

    # 70B must have higher latency and higher cost than 8B
    assert res_quality.latency.total_p95_ms > res_cheap.latency.total_p95_ms
    assert res_quality.estimated_cost_per_1k_tokens > res_cheap.estimated_cost_per_1k_tokens


def test_simulator_rollback():
    """Verify state rollback to last known good configuration."""
    sim = InferenceSimulator()
    cfg1 = ServingConfig(config_id="cfg-1", max_concurrency=8, routing_mix={"llama-3-8b": 0.5, "llama-3-70b": 0.5})
    cfg2 = ServingConfig(config_id="cfg-2", max_concurrency=16, routing_mix={"llama-3-8b": 0.8, "llama-3-70b": 0.2})

    sim.apply_config(cfg1)
    assert sim.active_config.config_id == "cfg-1"

    sim.apply_config(cfg2)
    assert sim.active_config.config_id == "cfg-2"

    sim.rollback()
    assert sim.active_config.config_id == "cfg-1"

