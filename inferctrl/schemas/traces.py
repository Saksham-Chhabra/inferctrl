"""Schemas for request traces and telemetry."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    CHAT = "chat"
    CODE = "code"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    STRUCTURED_JSON = "structured_json"
    OTHER = "other"


class RequestTrace(BaseModel):
    """Single request trace metadata."""
    request_id: str
    arrival_time_ms: float
    task_type: TaskType = TaskType.CHAT
    prompt_tokens: int = Field(ge=1)
    output_tokens: int = Field(ge=1)
    model_id: Optional[str] = None
    endpoint: str = "default"
    tenant_id: str = "default"


class WorkloadFeatures(BaseModel):
    """Summary of workload telemetry over a sliding window."""
    window_seconds: float
    total_requests: int
    requests_per_second: float
    p50_prompt_tokens: float
    p95_prompt_tokens: float
    p50_output_tokens: float
    p95_output_tokens: float
    burstiness_index: float
    current_queue_depth: int
    error_rate: float
    task_distribution: Dict[str, float] = Field(default_factory=dict)


class WorkloadReport(BaseModel):
    """Output contract from the Workload Analyst Agent."""
    regime: str  # e.g., "steady_short", "bursty", "long_context", "mixed", "unknown"
    confidence: float = Field(ge=0.0, le=1.0)
    window_seconds: float
    features: WorkloadFeatures
    drift_detected: bool
    evidence: List[str] = Field(default_factory=list)
    recommended_experiment_focus: List[str] = Field(default_factory=list)

