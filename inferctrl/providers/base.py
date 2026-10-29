"""Base interfaces and contracts for LLM Provider Adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MeasurementSource(str, Enum):
    ACTUAL = "actual"
    PROVIDER_REPORTED = "provider_reported"
    ESTIMATED = "estimated"
    SIMULATED = "simulated"


@dataclass
class ModelMetadata:
    id: str
    name: str
    provider: str
    context_length: int = 8192
    is_available: bool = True
    task_specialties: List[str] = field(default_factory=lambda: ["chat"])
    quality_score: Optional[float] = None
    quality_verified: bool = False


@dataclass
class ProviderHealth:
    healthy: bool
    provider: str
    latency_ms: float = 0.0
    message: str = "OK"
    available_models: List[str] = field(default_factory=list)


@dataclass
class ProviderResponse:
    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_latency_ms: float
    measurement_source: MeasurementSource
    provider_reported_cost: Optional[float] = None
    provider_metadata: Dict[str, Any] = field(default_factory=dict)


class BaseProviderAdapter(ABC):
    """Abstract interface implemented by Gemini, Ollama, and Simulator adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider credentials/endpoints are reachable and ready."""
        pass

    @abstractmethod
    def list_models(self) -> List[ModelMetadata]:
        """List models supported by this provider."""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> ProviderResponse:
        """Execute chat completion and return standardized ProviderResponse."""
        pass

    @abstractmethod
    async def health_check(self) -> ProviderHealth:
        """Perform a live ping/health check."""
        pass

