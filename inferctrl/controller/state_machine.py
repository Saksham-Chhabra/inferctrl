"""Autonomous Controller Agent and Closed-Loop State Machine."""

import uuid
from typing import Dict, List, Optional, Tuple
from inferctrl.agents.benchmark import BenchmarkAgent
from inferctrl.agents.scheduler import SchedulerAgent
from inferctrl.agents.workload import WorkloadAnalystAgent
from inferctrl.policy.engine import PolicyEngine
from inferctrl.quality.golden_dataset import QualityEvaluator
from inferctrl.schemas import (
    AuditEvent,
    BenchmarkResult,
    CanarySession,
    CandidateProposal,
    ControllerState,
    DecisionOutcome,
    DecisionRecord,
    QualityScore,
    RequestTrace,
    ServingConfig,
    WorkloadReport,
)
from inferctrl.serving.base import ServingAdapter


class OptimizationLoopResult:
    def __init__(
        self,
        run_id: str,
        workload_report: WorkloadReport,
        baseline_result: BenchmarkResult,
        candidate_results: List[Tuple[CandidateProposal, BenchmarkResult, QualityScore, bool, List[str]]],
        recommended_candidate: Optional[CandidateProposal],
        decision: DecisionRecord,
    ):
        self.run_id = run_id
        self.workload_report = workload_report
        self.baseline_result = baseline_result
        self.candidate_results = candidate_results
        self.recommended_candidate = recommended_candidate
        self.decision = decision


class ControllerAgent:
    """Coordinates closed-loop LLM inference optimization, safety enforcement, and rollbacks."""

    def __init__(
        self,
        serving_adapter: ServingAdapter,
        policy_engine: PolicyEngine,
        quality_evaluator: QualityEvaluator,
        baseline_config: ServingConfig,
    ):
        self.adapter = serving_adapter
        self.policy = policy_engine
        self.quality_evaluator = quality_evaluator
        self.active_config = baseline_config
        self.last_known_good_config = baseline_config
        self.state = ControllerState.IDLE
        self.audit_events: List[AuditEvent] = []
        self.active_canary: Optional[CanarySession] = None

        # Apply baseline config initially
        self.adapter.apply_config(self.active_config)
        self._record_audit("controller", "init", "baseline_config", "success", {"config_id": baseline_config.config_id})

    def _record_audit(self, actor: str, action: str, target: str, result: str, details: Optional[Dict[str, str]] = None):
        event = AuditEvent(
            event_id=f"audit-{uuid.uuid4().hex[:8]}",
            actor=actor,
            action=action,
            target=target,
            result=result,
            details=details or {},
        )
        self.audit_events.append(event)
        return event

    def run_closed_loop(
        self,
        traces: List[RequestTrace],
        window_seconds: float = 60.0,
        current_queue_depth: int = 0,
        run_id: Optional[str] = None,
    ) -> OptimizationLoopResult:
        run_id = run_id or f"run-{uuid.uuid4().hex[:6]}"
        self.state = ControllerState.OBSERVING
        self._record_audit("controller", "state_transition", "state", "OBSERVING", {"run_id": run_id})

        # 1. Observe Workload
        analyst = WorkloadAnalystAgent()
        workload_report = analyst.analyze_window(traces, window_seconds, current_queue_depth)
        self._record_audit(
            "workload_analyst",
            "classify_regime",
            workload_report.regime,
            "success",
            {"drift": str(workload_report.drift_detected), "confidence": str(workload_report.confidence)},
        )

        # 2. Benchmark Baseline
        self.state = ControllerState.BENCHMARKING
        bench_agent = BenchmarkAgent(self.adapter)
        baseline_proposal = CandidateProposal(
            candidate_id=f"{self.active_config.config_id}-base",
            run_id=run_id,
            parent_config_id=self.active_config.config_id,
            config=self.active_config,
            hypothesis="Baseline performance benchmark under current workload.",
            targeted_knob="baseline",
            expected_tradeoff="None",
            validation_passed=True,
        )
        baseline_result = bench_agent.run_candidate_benchmark(baseline_proposal, traces)
        baseline_quality = self.quality_evaluator.evaluate_config(self.active_config)

        # 3. Hypothesize Candidates
        self.state = ControllerState.HYPOTHESIZING
        scheduler = SchedulerAgent(run_id=run_id)
        proposals = scheduler.generate_candidates(self.active_config, workload_report)

        # 4. Validate & Benchmark Candidates
        self.state = ControllerState.BENCHMARKING
        candidate_evaluations = []
        valid_passed_candidates = []

        for prop in proposals:
            # Deterministic proposal validation
            is_valid, validation_reasons = self.policy.validate_proposal(prop)
            prop.validation_passed = is_valid
            prop.rejection_reasons = validation_reasons

            if not is_valid:
                continue

            # Run trial
            b_res = bench_agent.run_candidate_benchmark(prop, traces)
            # Evaluate quality
            q_res = self.quality_evaluator.evaluate_config(
                prop.config,
                baseline_score=baseline_quality.overall_score,
                max_allowed_regression_pct=self.policy.max_quality_regression_pct,
            )

            # Evaluate guardrails
            passed_guardrails, violations = self.policy.evaluate_guardrails(b_res, q_res, baseline_result)

            candidate_evaluations.append((prop, b_res, q_res, passed_guardrails, violations))

            if passed_guardrails:
                obj_score = self.policy.compute_objective_score(b_res, baseline_result)
                valid_passed_candidates.append((prop, b_res, q_res, obj_score))

        # 5. Evaluate & Decide
        self.state = ControllerState.EVALUATING
        best_candidate: Optional[CandidateProposal] = None
        decision_record: DecisionRecord

        if valid_passed_candidates:
            # Sort by lowest objective score (best trade-off)
            valid_passed_candidates.sort(key=lambda x: x[3])
            best_cand, best_bench, best_qual, best_score = valid_passed_candidates[0]
            best_candidate = best_cand

            lat_delta = round(
                ((best_bench.latency.total_p95_ms - baseline_result.latency.total_p95_ms) / baseline_result.latency.total_p95_ms) * 100.0,
                1,
            )
            cost_delta = round(
                ((best_bench.estimated_cost_per_request - baseline_result.estimated_cost_per_request) / baseline_result.estimated_cost_per_request)
                * 100.0,
                1,
            )
            qual_delta = round((best_qual.overall_score - baseline_quality.overall_score) * 100.0, 1)

            self.state = ControllerState.RECOMMENDING
            decision_record = DecisionRecord(
                decision_id=f"dec-{uuid.uuid4().hex[:8]}",
                run_id=run_id,
                candidate_id=best_candidate.candidate_id,
                outcome=DecisionOutcome.APPROVE_RECOMMENDATION,
                rationale=(
                    f"Candidate {best_candidate.candidate_id} improves P95 latency by {abs(lat_delta):.1f}% "
                    f"and cost by {abs(cost_delta):.1f}% with quality at {best_qual.overall_score:.3f} "
                    f"(guardrails satisfied)."
                ),
                evidence=[
                    f"Objective score: {best_score:.3f}",
                    f"Latency P95: {best_bench.latency.total_p95_ms:.1f}ms (SLO: {self.policy.target_p95_latency_ms}ms)",
                    f"Error rate: {best_bench.error_rate:.3f}",
                ],
                latency_delta_pct=lat_delta,
                cost_delta_pct=cost_delta,
                quality_delta_pct=qual_delta,
            )
            self._record_audit(
                "controller", "recommend_candidate", best_candidate.candidate_id, "RECOMMENDED", {"score": str(best_score)}
            )
        else:
            self.state = ControllerState.REJECTED
            decision_record = DecisionRecord(
                decision_id=f"dec-{uuid.uuid4().hex[:8]}",
                run_id=run_id,
                candidate_id=None,
                outcome=DecisionOutcome.REJECT_RECOMMENDATION,
                rationale="All proposed candidates violated hard latency, error, or quality guardrails.",
                evidence=["No candidate passed the deterministic safety policy checks."],
                latency_delta_pct=0.0,
                cost_delta_pct=0.0,
                quality_delta_pct=0.0,
            )
            self._record_audit("controller", "reject_all_candidates", run_id, "REJECTED")

        return OptimizationLoopResult(
            run_id=run_id,
            workload_report=workload_report,
            baseline_result=baseline_result,
            candidate_results=candidate_evaluations,
            recommended_candidate=best_candidate,
            decision=decision_record,
        )

    def start_canary(self, candidate_proposal: CandidateProposal, traffic_pct: float = 5.0) -> CanarySession:
        """Promote an approved candidate to canary."""
        self.state = ControllerState.CANARYING
        canary = CanarySession(
            canary_id=f"canary-{uuid.uuid4().hex[:8]}",
            candidate_id=candidate_proposal.candidate_id,
            traffic_pct=traffic_pct,
            status="active",
        )
        self.active_canary = canary
        self._record_audit(
            "operator", "start_canary", candidate_proposal.candidate_id, "CANARYING", {"traffic_pct": str(traffic_pct)}
        )
        return canary

    def check_canary_health(self, measured_p95_ms: float, measured_error_rate: float) -> Tuple[bool, Optional[str]]:
        """Watchdog: verifies if live canary is within SLOs. Returns (is_healthy, rollback_reason)."""
        if not self.active_canary:
            return (True, None)

        self.active_canary.current_p95_ms = measured_p95_ms
        self.active_canary.current_error_rate = measured_error_rate

        if measured_p95_ms > self.policy.target_p95_latency_ms:
            reason = f"Canary P95 latency {measured_p95_ms:.1f}ms breached SLO ({self.policy.target_p95_latency_ms:.1f}ms)"
            return (False, reason)

        if measured_error_rate > self.policy.max_error_rate:
            reason = f"Canary error rate {measured_error_rate:.3f} breached threshold ({self.policy.max_error_rate:.3f})"
            return (False, reason)

        return (True, None)

    def trigger_rollback(self, reason: str) -> bool:
        """Trigger instant automated rollback to last known good configuration."""
        self.state = ControllerState.ROLLING_BACK
        self.adapter.rollback()
        self.state = ControllerState.ROLLED_BACK
        if self.active_canary:
            self.active_canary.status = "rolled_back"
            self.active_canary.rollback_reason = reason

        self._record_audit("safety_watchdog", "rollback", self.active_config.config_id, "ROLLED_BACK", {"reason": reason})
        return True

    def promote_canary(self) -> bool:
        """Fully promote canary to active configuration."""
        if not self.active_canary:
            return False
        self.state = ControllerState.PROMOTED
        self.active_canary.status = "promoted"
        self._record_audit("controller", "promote_config", self.active_canary.candidate_id, "PROMOTED")
        self.active_canary = None
        return True

