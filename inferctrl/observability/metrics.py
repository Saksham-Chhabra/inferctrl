"""Request Observability and Metrics Registry."""

from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from inferctrl.providers.base import MeasurementSource


class RequestLogRecord(BaseModel):
    """Complete, privacy-conscious audit record for an LLM inference request."""
    request_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    app_id: str = "default"
    requested_model: str
    selected_provider: str
    selected_model: str
    routing_strategy: str
    decision_reason: str
    ttft_ms: float
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    provider_reported_cost: Optional[float] = None
    status: str = "success"  # success, error, fallback
    error_category: str = "none"  # none, timeout, provider_error, auth_error
    measurement_source: MeasurementSource
    quality_score: Optional[float] = None
    prompt_preview: str = Field(default="[redacted for privacy]")


class ObservabilityRegistry:
    """Thread-safe in-memory ring buffer for request telemetry and performance aggregates."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.records: deque[RequestLogRecord] = deque(maxlen=capacity)

    def log_request(self, record: RequestLogRecord):
        self.records.append(record)

    def get_recent(
        self,
        limit: int = 50,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[RequestLogRecord]:
        result = list(self.records)
        if provider:
            result = [r for r in result if r.selected_provider.lower() == provider.lower()]
        if model:
            result = [r for r in result if r.selected_model.lower() == model.lower()]
        if status:
            result = [r for r in result if r.status.lower() == status.lower()]
        return list(reversed(result))[:limit]

    def compute_summary(self, window_seconds: float = 300.0) -> Dict[str, any]:
        if not self.records:
            return {
                "total_requests": 0,
                "requests_per_sec": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "error_rate": 0.0,
                "total_cost_usd": 0.0,
                "provider_breakdown": {},
                "model_breakdown": {},
            }

        now = datetime.now(timezone.utc)
        recent = [
            r for r in self.records
            if (now - r.timestamp).total_seconds() <= window_seconds
        ] or list(self.records)

        latencies = [r.total_latency_ms for r in recent]
        costs = [r.estimated_cost_usd for r in recent]
        errors = [r for r in recent if r.status != "success"]

        provider_counts: Dict[str, int] = {}
        model_counts: Dict[str, int] = {}
        for r in recent:
            provider_counts[r.selected_provider] = provider_counts.get(r.selected_provider, 0) + 1
            model_counts[r.selected_model] = model_counts.get(r.selected_model, 0) + 1

        effective_sec = max(1.0, window_seconds)
        rps = round(len(recent) / effective_sec, 2)

        return {
            "total_requests": len(recent),
            "requests_per_sec": rps,
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 1) if latencies else 0.0,
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else 0.0,
            "error_rate": round(len(errors) / max(1, len(recent)), 4),
            "total_cost_usd": round(sum(costs), 6),
            "provider_breakdown": provider_counts,
            "model_breakdown": model_counts,
        }

