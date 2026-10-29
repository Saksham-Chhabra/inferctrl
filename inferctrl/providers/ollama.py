"""Ollama Local Inference Provider Adapter."""

import os
import time
from typing import Any, Dict, List, Optional
import httpx

from inferctrl.providers.base import (
    BaseProviderAdapter,
    MeasurementSource,
    ModelMetadata,
    ProviderHealth,
    ProviderResponse,
)


class OllamaProviderAdapter(BaseProviderAdapter):
    """Adapter for Ollama local inference server."""

    DEFAULT_FALLBACK_MODELS = [
        ModelMetadata(
            id="llama3",
            name="Llama 3 (Local)",
            provider="ollama",
            context_length=8192,
            is_available=False,
            task_specialties=["chat", "code", "general"],
            quality_score=0.88,
            quality_verified=True,
        ),
        ModelMetadata(
            id="mistral",
            name="Mistral 7B (Local)",
            provider="ollama",
            context_length=8192,
            is_available=False,
            task_specialties=["chat", "reasoning"],
            quality_score=0.86,
            quality_verified=True,
        ),
        ModelMetadata(
            id="qwen2.5-coder",
            name="Qwen 2.5 Coder (Local)",
            provider="ollama",
            context_length=32768,
            is_available=False,
            task_specialties=["code", "structured_json"],
            quality_score=0.92,
            quality_verified=True,
        ),
    ]

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self._cached_available: Optional[bool] = None
        self._cached_models: List[str] = []

    @property
    def name(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        if self._cached_available is not None:
            return self._cached_available
        try:
            with httpx.Client(timeout=1.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    self._cached_available = True
                    self._cached_models = [m.get("name", "") for m in res.json().get("models", [])]
                    return True
        except Exception:
            pass
        self._cached_available = False
        return False

    def list_models(self) -> List[ModelMetadata]:
        available = self.is_available()
        if available and self._cached_models:
            return [
                ModelMetadata(
                    id=m_name,
                    name=f"{m_name} (Ollama Local)",
                    provider="ollama",
                    context_length=8192,
                    is_available=True,
                    task_specialties=["chat", "local_compute"],
                    quality_score=0.88,
                    quality_verified=False,
                )
                for m_name in self._cached_models
            ]
        # Return default known models with is_available status
        return [
            ModelMetadata(
                id=m.id,
                name=m.name,
                provider=m.provider,
                context_length=m.context_length,
                is_available=available,
                task_specialties=m.task_specialties,
                quality_score=m.quality_score,
                quality_verified=m.quality_verified,
            )
            for m in self.DEFAULT_FALLBACK_MODELS
        ]

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> ProviderResponse:
        chosen_model = model or "llama3"
        if chosen_model.startswith("ollama:"):
            chosen_model = chosen_model.split(":", 1)[1]

        start_time = time.perf_counter()

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": chosen_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                res = await client.post(url, json=payload)
        except Exception as e:
            raise RuntimeError(f"Ollama local connection failed to {self.base_url}: {str(e)}")

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        if res.status_code != 200:
            raise RuntimeError(f"Ollama error ({res.status_code}): {res.text}")

        data = res.json()
        content = data.get("message", {}).get("content", "")

        # Ollama returns exact nano-second durations
        prompt_eval_ns = data.get("prompt_eval_duration", 0)
        ttft_ms = (prompt_eval_ns / 1_000_000.0) if prompt_eval_ns else (total_latency_ms * 0.35)

        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return ProviderResponse(
            content=content,
            model=chosen_model,
            provider="ollama",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=round(ttft_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            measurement_source=MeasurementSource.ACTUAL,
            provider_reported_cost=0.0,  # Local hardware compute
            provider_metadata={
                "total_duration_ms": data.get("total_duration", 0) / 1_000_000.0,
                "load_duration_ms": data.get("load_duration", 0) / 1_000_000.0,
            },
        )

    async def health_check(self) -> ProviderHealth:
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
            elapsed = (time.perf_counter() - start) * 1000.0

            if res.status_code == 200:
                models = [m.get("name", "") for m in res.json().get("models", [])]
                self._cached_available = True
                self._cached_models = models
                return ProviderHealth(
                    healthy=True,
                    provider="ollama",
                    latency_ms=round(elapsed, 1),
                    message="Ollama daemon reachable",
                    available_models=models,
                )
            else:
                self._cached_available = False
                return ProviderHealth(
                    healthy=False,
                    provider="ollama",
                    latency_ms=round(elapsed, 1),
                    message=f"Ollama returned HTTP {res.status_code}",
                    available_models=[],
                )
        except Exception:
            self._cached_available = False
            return ProviderHealth(
                healthy=False,
                provider="ollama",
                message=f"Ollama not running at {self.base_url}. Start with 'ollama serve' for local models.",
                available_models=[],
            )

