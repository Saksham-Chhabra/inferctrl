"""Benchmark Agent: Orchestrates controlled experiments and metrics aggregation."""

from typing import List, Tuple
from inferctrl.schemas import (
    BenchmarkResult,
    CandidateProposal,
    RequestTrace,
    ServingConfig,
)
from inferctrl.serving.base import ServingAdapter


class BenchmarkAgent:
    """Coordinates repeated benchmark trials against serving configurations."""

    def __init__(self, adapter: ServingAdapter, warmup_count: int = 20, random_seed: int = 42):
        self.adapter = adapter
        self.warmup_count = warmup_count
        self.random_seed = random_seed

    def run_candidate_benchmark(
        self,
        proposal: CandidateProposal,
        traces: List[RequestTrace],
    ) -> BenchmarkResult:
        """Execute benchmark for a candidate proposal and attach candidate metadata."""
        result = self.adapter.run_benchmark(
            config=proposal.config,
            traces=traces,
            warmup_count=self.warmup_count,
            random_seed=self.random_seed,
        )
        # Link result with candidate proposal IDs
        return result.model_copy(
            update={
                "candidate_id": proposal.candidate_id,
                "run_id": proposal.run_id,
            }
        )

