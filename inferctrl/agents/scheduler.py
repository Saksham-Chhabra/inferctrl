"""Scheduler Agent: Serving configuration candidate proposal engine."""

from typing import List
from inferctrl.schemas import (
    CandidateProposal,
    ServingConfig,
    WorkloadReport,
)


class SchedulerAgent:
    """Agent responsible for proposing serving configurations tailored to detected workload shifts."""

    def __init__(self, run_id: str = "run-default"):
        self.run_id = run_id

    def generate_candidates(
        self,
        base_config: ServingConfig,
        workload_report: WorkloadReport,
    ) -> List[CandidateProposal]:
        candidates: List[CandidateProposal] = []
        regime = workload_report.regime

        # Candidate 1: Concurrency & dynamic batching scale-up
        cfg_1 = base_config.model_copy(
            update={
                "config_id": f"{self.run_id}-cand-concurrency",
                "max_concurrency": min(32, base_config.max_concurrency * 2),
                "max_batch_size": min(64, base_config.max_batch_size * 2),
                "batch_wait_ms": 25.0,
            }
        )
        candidates.append(
            CandidateProposal(
                candidate_id=cfg_1.config_id,
                run_id=self.run_id,
                parent_config_id=base_config.config_id,
                config=cfg_1,
                hypothesis="Expanding concurrency and batch capacity allows draining burst arrivals before queue times breach SLO.",
                targeted_knob="max_concurrency,max_batch_size",
                expected_tradeoff="Higher throughput and lower queue time, slight ITL contention.",
            )
        )

        # Candidate 2: Cost-aware balanced routing
        cfg_2 = base_config.model_copy(
            update={
                "config_id": f"{self.run_id}-cand-cost-optimized",
                "max_concurrency": 12,
                "max_batch_size": 24,
                "batch_wait_ms": 20.0,
                "routing_mix": {"llama-3-8b": 0.80, "llama-3-70b": 0.20},
            }
        )
        candidates.append(
            CandidateProposal(
                candidate_id=cfg_2.config_id,
                run_id=self.run_id,
                parent_config_id=base_config.config_id,
                config=cfg_2,
                hypothesis="Shifting 10% more traffic to Llama-3-8B reduces cost per request by ~30% with < 0.5% quality regression.",
                targeted_knob="routing_mix",
                expected_tradeoff="Significantly lower cost and latency; slight quality decrease on deep reasoning.",
            )
        )

        # Candidate 3: Fast TTFT / Low-Wait candidate
        cfg_3 = base_config.model_copy(
            update={
                "config_id": f"{self.run_id}-cand-low-latency",
                "max_concurrency": 10,
                "max_batch_size": 16,
                "batch_wait_ms": 5.0,
                "routing_mix": {"llama-3-8b": 0.70, "llama-3-70b": 0.30},
            }
        )
        candidates.append(
            CandidateProposal(
                candidate_id=cfg_3.config_id,
                run_id=self.run_id,
                parent_config_id=base_config.config_id,
                config=cfg_3,
                hypothesis="Reducing batch wait to 5ms improves time-to-first-token (TTFT) for interactive chat.",
                targeted_knob="batch_wait_ms",
                expected_tradeoff="Lower TTFT, slightly reduced batch throughput packing.",
            )
        )

        # Candidate 4: Aggressive candidate designed to test guardrail rejection
        cfg_reject = base_config.model_copy(
            update={
                "config_id": f"{self.run_id}-cand-extreme-8b",
                "max_concurrency": 16,
                "max_batch_size": 32,
                "batch_wait_ms": 10.0,
                "routing_mix": {"llama-3-8b": 0.98, "llama-3-70b": 0.02},  # Drops quality heavily!
            }
        )
        candidates.append(
            CandidateProposal(
                candidate_id=cfg_reject.config_id,
                run_id=self.run_id,
                parent_config_id=base_config.config_id,
                config=cfg_reject,
                hypothesis="Diverting nearly all traffic to 8B minimizes cost, testing quality guardrail enforcement.",
                targeted_knob="routing_mix",
                expected_tradeoff="Lowest cost but high probability of failing quality guardrail.",
            )
        )

        return candidates

