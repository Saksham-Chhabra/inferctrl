"""Automated 3-5 Minute Hackathon Presentation Demo Script for INFERCTRL."""

import time

from inferctrl.controller.state_machine import ControllerAgent
from inferctrl.policy.engine import PolicyEngine
from inferctrl.quality.golden_dataset import QualityEvaluator
from inferctrl.schemas import ControllerState, DecisionOutcome, ServingConfig
from inferctrl.serving.simulator import InferenceSimulator
from inferctrl.telemetry.generator import TraceGenerator


def print_header(title: str):
    print(f"\n{'='*75}")
    print(f"  >>> {title}")
    print(f"{'='*75}\n")


def run_demo():
    print_header("INFERCTRL: Autonomous LLM Inference Optimizer — Hackathon Demo")

    # 1. Start in SIMULATED mode and show the baseline
    print("[Step 1] Initializing Serving Infrastructure (SIMULATED mode)...")
    sim = InferenceSimulator()
    policy = PolicyEngine(target_p95_latency_ms=5000.0, max_error_rate=0.03, max_quality_regression_pct=1.0)
    quality = QualityEvaluator()

    baseline_config = ServingConfig(
        config_id="cfg-baseline-v1",
        max_concurrency=8,
        max_batch_size=16,
        batch_wait_ms=15.0,
        routing_mix={"llama-3-8b": 0.70, "llama-3-70b": 0.30},
    )
    controller = ControllerAgent(sim, policy, quality, baseline_config)
    print(f"  Active Baseline: {baseline_config.config_id}")
    print(f"  Model Routing  : 70% Llama-3-8B | 30% Llama-3-70B")
    print(f"  Batching Knobs : max_batch_size={baseline_config.max_batch_size}, wait={baseline_config.batch_wait_ms}ms, conc={baseline_config.max_concurrency}")

    # 2. Replay steady traffic
    print("\n[Step 2] Replaying Steady Traffic (Nominal Load)...")
    steady_traces = TraceGenerator.generate_trace(scenario="steady_short", request_count=80, random_seed=42)
    steady_bench = sim.run_benchmark(baseline_config, steady_traces, warmup_count=15)
    print(f"  Steady P95 Latency : {steady_bench.latency.total_p95_ms:.1f} ms")
    print(f"  Steady Throughput  : {steady_bench.throughput_req_per_sec:.1f} req/sec")
    print(f"  Steady Cost / 1K   : ${steady_bench.estimated_cost_per_1k_tokens:.4f}")

    # 3. Trigger a burst and a shift toward longer prompts
    print_header("[Step 3] INJECTING WORKLOAD SURGE (+2.1x Burst & Heavy Reasoning)")
    burst_traces = TraceGenerator.generate_trace(scenario="bursty_spike", request_count=100, random_seed=42)
    print(f"  Simulating 100 burst requests with arrival spike...")

    # 4. Show Workload Analyst evidence and the generated optimization hypothesis
    print("\n[Step 4] Workload Analyst Agent: Analyzing Sliding Window...")
    loop_result = controller.run_closed_loop(burst_traces, current_queue_depth=18)
    report = loop_result.workload_report
    print(f"  Classified Regime : {report.regime.upper()} (Confidence: {report.confidence*100:.0f}%)")
    print(f"  Drift Detected    : {report.drift_detected}")
    for ev in report.evidence:
        print(f"    - Evidence: {ev}")
    print(f"  Focus Knobs       : {', '.join(report.recommended_experiment_focus)}")

    # 5. Candidate Experiments
    print_header("[Step 5] Benchmarking Proposed Candidates Against Same Replay Trace")
    baseline_res = loop_result.baseline_result
    print(f"  Baseline under Burst: P95 = {baseline_res.latency.total_p95_ms:.1f} ms | Cost/1K = ${baseline_res.estimated_cost_per_1k_tokens:.4f}\n")

    print(f"  {'Candidate ID':<35} | {'P95 Latency':<12} | {'Quality':<8} | {'Cost/1K':<9} | {'Guardrail Status'}")
    print(f"  {'-'*85}")

    for prop, b_res, q_res, passed, viol in loop_result.candidate_results:
        status_str = "PASSED [Valid]" if passed else f"REJECTED: {viol[0][:35]}..."
        print(f"  {prop.candidate_id:<35} | {b_res.latency.total_p95_ms:>8.1f} ms | {q_res.overall_score:>7.3f} | ${b_res.estimated_cost_per_1k_tokens:>7.4f} | {status_str}")

    # 6. Show Best Valid Candidate
    best = loop_result.recommended_candidate
    print_header("[Step 6] Controller Decision & Pareto Optimum")
    print(f"  Selected Candidate : {best.candidate_id}")
    print(f"  Decision Rationale : {loop_result.decision.rationale}")
    print(f"  Targeted Knobs     : {best.targeted_knob}")
    print(f"  Measured Deltas    : P95 Latency {loop_result.decision.latency_delta_pct}% | Cost {loop_result.decision.cost_delta_pct}%")

    # 7. Promote to 5% Canary
    print_header("[Step 7] Promoting to 5% Live Canary Lane")
    canary = controller.start_canary(best, traffic_pct=5.0)
    print(f"  Canary Session ID : {canary.canary_id}")
    print(f"  Traffic Split     : {canary.traffic_pct}% Canary / 95% Baseline")
    print(f"  Controller State  : {controller.state.value.upper()}")

    # 8. Inject Regression and Automatic Rollback
    print_header("[Step 8] FAULT INJECTION: Simulating GPU Throttle & Latency Spike")
    print(f"  Canary latency surges to 6,200 ms (breaching target SLO: 5,000 ms)...")
    healthy, reason = controller.check_canary_health(measured_p95_ms=6200.0, measured_error_rate=0.0)
    print(f"  Safety Watchdog Health Check: Passed = {healthy}")
    print(f"  Breach Reason: {reason}")

    print("\n[Step 9] Instant Automated Rollback...")
    controller.trigger_rollback(reason)
    print(f"  Controller State  : {controller.state.value.upper()}")
    print(f"  Canary Status     : {canary.status.upper()}")
    print(f"  Restored Config   : {controller.active_config.config_id}")

    # 10. Audit Trail
    print_header("[Step 10] Immutable Audit Log Trail")
    for event in controller.audit_events[-5:]:
        print(f"  [{event.timestamp.strftime('%H:%M:%S')}] {event.actor.upper():<16} -> {event.action:<20} | Result: {event.result:<12} | Target: {event.target}")

    print_header("DEMO COMPLETE: Zero-GPU Autonomous Optimization Loop Verified!")


if __name__ == "__main__":
    run_demo()
