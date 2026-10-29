"""Schemas for controller decisions, canary rollouts, and audit events."""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ControllerState(str, Enum):
    IDLE = "idle"
    OBSERVING = "observing"
    HYPOTHESIZING = "hypothesizing"
    BENCHMARKING = "benchmarking"
    EVALUATING = "evaluating"
    RECOMMENDING = "recommending"
    CANARYING = "canarying"
    PROMOTED = "promoted"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"
    REJECTED = "rejected"


class DecisionOutcome(str, Enum):
    APPROVE_RECOMMENDATION = "approve_recommendation"
    REJECT_RECOMMENDATION = "reject_recommendation"
    START_CANARY = "start_canary"
    ROLLBACK_CANARY = "rollback_canary"
    KEEP_CURRENT = "keep_current"
    INCONCLUSIVE = "inconclusive"


class DecisionRecord(BaseModel):
    """Immutable audit record of a controller decision."""
    decision_id: str
    run_id: str
    candidate_id: Optional[str] = None
    outcome: DecisionOutcome
    rationale: str
    evidence: List[str] = Field(default_factory=list)
    latency_delta_pct: float
    cost_delta_pct: float
    quality_delta_pct: float
    policy_version: str = "v1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CanarySession(BaseModel):
    """Canary deployment tracking record."""
    canary_id: str
    candidate_id: str
    traffic_pct: float = 5.0
    status: str = "active"  # active, completed, rolled_back
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_seconds: int = 60
    current_p95_ms: float = 0.0
    current_error_rate: float = 0.0
    rollback_reason: Optional[str] = None


class AuditEvent(BaseModel):
    """Single immutable audit log item."""
    event_id: str
    actor: str  # e.g., "workload_agent", "policy_engine", "operator", "controller"
    action: str
    target: str
    result: str
    details: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

