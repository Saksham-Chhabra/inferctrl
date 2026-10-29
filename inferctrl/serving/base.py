"""Base protocol and abstract interface for serving adapters."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from inferctrl.schemas import ServingConfig, BenchmarkResult, RequestTrace


class ServingAdapter(ABC):
    """Abstract interface for all inference serving backends (Simulator, vLLM)."""

    @abstractmethod
    def validate_config(self, config: ServingConfig) -> Tuple[bool, List[str]]:
        """Validate whether the configuration is supported and safe for this backend."""
        pass

    @abstractmethod
    def apply_config(self, config: ServingConfig) -> bool:
        """Apply the configuration to the serving lane."""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, any]:
        """Verify endpoint connectivity, worker status, and error rate."""
        pass

    @abstractmethod
    def rollback(self) -> bool:
        """Roll back to the last known-good configuration."""
        pass

    @abstractmethod
    def run_benchmark(
        self,
        config: ServingConfig,
        traces: List[RequestTrace],
        warmup_count: int = 20,
        random_seed: int = 42,
    ) -> BenchmarkResult:
        """Execute a controlled benchmark against the provided request trace."""
        pass

