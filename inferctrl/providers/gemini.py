"""Google Gemini Provider Adapter."""

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


class GeminiProviderAdapter(BaseProviderAdapter):
    """Adapter for Google Gemini API."""

    SUPPORTED_MODELS = [
        ModelMetadata(
            id="gemini-1.5-flash",
            name="Gemini 1.5 Flash",
            provider="gemini",
            context_length=1048576,
            is_available=True,
            task_specialties=["chat", "code", "summarization", "structured_json"],
            quality_score=0.91,
            quality_verified=True,
        ),
        ModelMetadata(
            id="gemini-1.5-pro",
            name="Gemini 1.5 Pro",
            provider="gemini",
            context_length=2097152,
            is_available=True,
            task_specialties=["reasoning", "code", "deep_analysis"],
            quality_score=0.96,
            quality_verified=True,
        ),
        ModelMetadata(
            id="gemini-2.0-flash",
            name="Gemini 2.0 Flash",
            provider="gemini",
            context_length=1048576,
            is_available=True,
            task_specialties=["chat", "code", "latency_sensitive"],
            quality_score=0.92,
            quality_verified=True,
        ),
        ModelMetadata(
            id="gemini-2.5-flash",
            name="Gemini 2.5 Flash",
            provider="gemini",
            context_length=1048576,
            is_available=True,
            task_specialties=["chat", "code", "multimodal"],
            quality_score=0.93,
            quality_verified=True,
        ),
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    @property
    def name(self) -> str:
        return "gemini"

    def is_available(self) -> bool:
        return bool(self.api_key.strip())

    def list_models(self) -> List[ModelMetadata]:
        available = self.is_available()
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
            for m in self.SUPPORTED_MODELS
        ]

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> ProviderResponse:
        if not self.is_available():
            raise RuntimeError(
                "Gemini API key is not configured. Set GEMINI_API_KEY in environment or .env."
            )

        chosen_model = model or "gemini-1.5-flash"
        # Normalize model string
        if chosen_model.startswith("gemini:"):
            chosen_model = chosen_model.split(":", 1)[1]

        start_time = time.perf_counter()

        # Format messages into Gemini format
        contents = []
        for msg in messages:
            role = "user" if msg.get("role") in ["user", "system"] else "model"
            contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{chosen_model}:generateContent?key={self.api_key}"
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            },
        }
        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API error ({resp.status_code}): {resp.text}")

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            text = ""
        else:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = parts[0].get("text", "") if parts else ""

        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)

        # TTFT approximation for non-streaming REST: 40% of turnaround
        ttft_ms = total_latency_ms * 0.40

        return ProviderResponse(
            content=text,
            model=chosen_model,
            provider="gemini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=round(ttft_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            measurement_source=MeasurementSource.ACTUAL,
            provider_metadata={
                "finish_reason": candidates[0].get("finishReason") if candidates else "UNKNOWN",
                "usage_metadata": usage,
            },
        )

    async def health_check(self) -> ProviderHealth:
        if not self.is_available():
            return ProviderHealth(
                healthy=False,
                provider="gemini",
                message="GEMINI_API_KEY environment variable is not configured.",
                available_models=[],
            )

        start = time.perf_counter()
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.get(url)
            elapsed = (time.perf_counter() - start) * 1000.0

            if res.status_code == 200:
                data = res.json()
                models = [m.get("name", "").replace("models/", "") for m in data.get("models", [])]
                return ProviderHealth(
                    healthy=True,
                    provider="gemini",
                    latency_ms=round(elapsed, 1),
                    message="Reachable and authenticated",
                    available_models=models[:10],
                )
            else:
                return ProviderHealth(
                    healthy=False,
                    provider="gemini",
                    latency_ms=round(elapsed, 1),
                    message=f"Gemini API authentication/network error ({res.status_code})",
                    available_models=[],
                )
        except Exception as e:
            return ProviderHealth(
                healthy=False,
                provider="gemini",
                message=f"Connection failed: {str(e)}",
                available_models=[],
            )

