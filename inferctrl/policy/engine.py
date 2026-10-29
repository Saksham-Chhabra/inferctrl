"""Deterministic Safety & Policy Engine."""

from typing import Dict, List, Optional, Tuple
from inferctrl.schemas import (
    BenchmarkResult,
    CandidateProposal,
    QualityScore,
    ServingConfig,
)


class PolicyEngine:
    """Deterministic authority for configuration validation, SLO checks, and Pareto frontier ranking."""

    def __init__(
        self,
        target_p95_latency_ms: float = 1000.0,
        max_error_rate: float = 0.02,
        max_quality_regression_pct: float = 1.0,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.target_p95_latency_ms = target_p95_latency_ms
        self.max_error_rate = max_error_rate
        self.max_quality_regression_pct = max_quality_regression_pct
        self.weights = weights or {
            "latency": 0.40,
            "cost": 0.40,
            "error": 0.10,
            "memory": 0.10,
        }

    def validate_proposal(self, proposal: CandidateProposal) -> Tuple[bool, List[str]]:
        """Validate candidate configuration bounds prior to running experiments."""
        reasons = []
        cfg = proposal.config

        if cfg.max_concurrency < 1 or cfg.max_concurrency > 64:
            reasons.append(f"max_concurrency {cfg.max_concurrency} out of bounds [1, 64]")
        if cfg.max_batch_size < 1 or cfg.max_batch_size > 128:
            reasons.append(f"max_batch_size {cfg.max_batch_size} out of bounds [1, 128]")
        if cfg.batch_wait_ms < 0.0 or cfg.batch_wait_ms > 100.0:
            reasons.append(f"batch_wait_ms {cfg.batch_wait_ms} out of bounds [0.0, 100.0]")

        return (len(reasons) == 0, reasons)

    def evaluate_guardrails(
        self,
        candidate_result: BenchmarkResult,
        quality_score: QualityScore,
        baseline_result: Optional[BenchmarkResult] = None,
    ) -> Tuple[bool, List[str]]:
        """Enforce hard SLO constraints on benchmarked results."""
        violations: List[str] = []

        # 1. P95 Latency SLO
        if candidate_result.latency.total_p95_ms > self.target_p95_latency_ms:
            violations.append(
                f"P95 latency {candidate_result.latency.total_p95_ms:.1f}ms exceeds target SLO "
                f"({self.target_p95_latency_ms:.1f}ms)"
            )

        # 2. Error Rate Guardrail
        if candidate_result.error_rate > self.max_error_rate:
            violations.append(
                f"Error rate {candidate_result.error_rate:.3f} exceeds maximum threshold ({self.max_error_rate:.3f})"
            )

        # 3. Quality Regression Guardrail
        if not quality_score.passed_guardrails:
            violations.append(f"Quality guardrail failed: {quality_score.rejection_reason}")

        # 4. Baseline regression comparisons if baseline provided
        if baseline_result is not None:
            if candidate_result.error_rate > (baseline_result.error_rate + 0.01):
                violations.append("Error rate increased by > 1.0% compared to baseline")

        passed = len(violations) == 0
        return (passed, violations)

    def compute_objective_score(
        self,
        candidate_result: BenchmarkResult,
        baseline_result: BenchmarkResult,
    ) -> float:
        """Compute normalized soft objective score (lower is better)."""
        norm_lat = candidate_result.latency.total_p95_ms / max(1.0, baseline_result.latency.total_p95_ms)
        norm_cost = candidate_result.estimated_cost_per_request / max(1e-6, baseline_result.estimated_cost_per_request)
        norm_err = (1.0 + candidate_result.error_rate) / (1.0 + baseline_result.error_rate)
        norm_mem = candidate_result.gpu_utilization_pct / 100.0

        score = (
            self.weights["latency"] * norm_lat
            + self.weights["cost"] * norm_cost
            + self.weights["error"] * norm_err
            + self.weights["memory"] * norm_mem
        )
        return round(score, 4)

    @staticmethod
    def extract_pareto_frontier(
        candidates: List[Tuple[BenchmarkResult, QualityScore]],
    ) -> List[Tuple[BenchmarkResult, QualityScore]]:
        """Extract non-dominated candidates optimizing: min Latency, min Cost, max Quality."""
        pareto_set: List[Tuple[BenchmarkResult, QualityScore]] = []

        for candidate in candidates:
            c_res, c_qual = candidate
            c_lat = c_res.latency.total_p95_ms
            c_cost = c_res.estimated_cost_per_1k_tokens
            c_quality = c_qual.overall_score

            is_dominated = False
            for other in candidates:
                if other == candidate:
                    continue
                o_res, o_qual = other
                o_lat = o_res.latency.total_p95_ms
                o_cost = o_res.estimated_cost_per_1k_tokens
                o_quality = o_qual.overall_score

                # other dominates candidate if other is at least as good in all and strictly better in at least one
                if (o_lat <= c_lat and o_cost <= c_cost and o_quality >= c_quality) and (
                    o_lat < c_lat or o_cost < c_cost or o_quality > c_quality
                ):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_set.append(candidate)

        return pareto_set

