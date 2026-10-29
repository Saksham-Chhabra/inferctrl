"""Schemas for benchmarks, trials, and metrics."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class TrialStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALID = "invalid"


class LatencyMetrics(BaseModel):
    """Granular latency percentiles in milliseconds."""
    queue_time_p50_ms: float
    queue_time_p95_ms: float
    ttft_p50_ms: float      # Time to first token
    ttft_p95_ms: float
    itl_p50_ms: float       # Inter-token latency
    itl_p95_ms: float
    total_p50_ms: float
    total_p95_ms: float
    total_p99_ms: float


class BenchmarkResult(BaseModel):
    """Aggregated result across trials for a single candidate."""
    candidate_id: str
    run_id: str
    trace_id: str
    trials_count: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    latency: LatencyMetrics
    throughput_req_per_sec: float
    tokens_per_second: float
    estimated_cost_per_request: float
    estimated_cost_per_1k_tokens: float
    gpu_utilization_pct: float
    is_valid: bool = True
    notes: Optional[str] = None


class QualityScore(BaseModel):
    """Subgroup and overall quality evaluation results."""
    candidate_id: str
    overall_score: float = Field(ge=0.0, le=1.0)
    baseline_score: float = Field(ge=0.0, le=1.0)
    score_delta: float
    task_scores: Dict[str, float] = Field(default_factory=dict)
    subgroup_regressions: Dict[str, float] = Field(default_factory=dict)
    schema_valid_fraction: float = 1.0
    is_conclusive: bool = True
    passed_guardrails: bool = True
    rejection_reason: Optional[str] = None

