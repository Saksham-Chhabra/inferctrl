"""OpenAI-compatible request and response data contracts with INFERCTRL telemetry."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="auto", description="Model ID or 'auto' for autonomous routing")
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = False
    routing_strategy: Optional[str] = Field(default=None, description="Override routing strategy: lowest_cost, lowest_latency, quality_first, balanced, round_robin")


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageMetadata(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class InferctrlMetadata(BaseModel):
    request_id: str
    provider: str
    selected_model: str
    routing_strategy: str
    decision_reason: str
    ttft_ms: float
    total_latency_ms: float
    estimated_cost_usd: float
    provider_reported_cost: Optional[float] = None
    measurement_source: str
    quality_verified: bool
    eligible_models: List[str] = Field(default_factory=list)


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: UsageMetadata
    inferctrl: InferctrlMetadata

