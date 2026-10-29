"""Deterministic Simulator Provider Adapter."""

import asyncio
import random
import time
from typing import Dict, List, Optional

from inferctrl.providers.base import (
    BaseProviderAdapter,
    MeasurementSource,
    ModelMetadata,
    ProviderHealth,
    ProviderResponse,
)


class SimulatorProviderAdapter(BaseProviderAdapter):
    """Zero-GPU deterministic provider adapter for development, CI, and live benchmarks."""

    MODELS = [
        ModelMetadata(
            id="sim-llama-3-8b",
            name="Llama 3 8B (Simulated)",
            provider="simulator",
            context_length=8192,
            is_available=True,
            task_specialties=["chat", "code", "fast_response"],
            quality_score=0.88,
            quality_verified=True,
        ),
        ModelMetadata(
            id="sim-llama-3-70b",
            name="Llama 3 70B (Simulated)",
            provider="simulator",
            context_length=8192,
            is_available=True,
            task_specialties=["reasoning", "complex_code", "deep_analysis"],
            quality_score=0.96,
            quality_verified=True,
        ),
        ModelMetadata(
            id="sim-gemini-flash",
            name="Gemini 1.5 Flash (Simulated)",
            provider="simulator",
            context_length=1048576,
            is_available=True,
            task_specialties=["chat", "summarization", "balanced"],
            quality_score=0.92,
            quality_verified=True,
        ),
    ]

    def __init__(self, random_seed: int = 42):
        self.rng = random.Random(random_seed)

    @property
    def name(self) -> str:
        return "simulator"

    def is_available(self) -> bool:
        return True

    def list_models(self) -> List[ModelMetadata]:
        return list(self.MODELS)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> ProviderResponse:
        chosen_model = model or "sim-llama-3-8b"
        if chosen_model.startswith("simulator:"):
            chosen_model = chosen_model.split(":", 1)[1]

        user_content = ""
        for m in messages:
            if m.get("role") == "user":
                user_content += m.get("content", "") + " "
        user_content = user_content.strip()

        # Generate realistic output based on prompt
        content = self._generate_simulated_answer(user_content, chosen_model)

        prompt_tokens = max(5, int(len(user_content.split()) * 1.3))
        completion_tokens = max(10, int(len(content.split()) * 1.3))

        # Model latency physics
        if "70b" in chosen_model:
            ttft_ms = 85.0 + prompt_tokens * 0.12
            itl_ms = 28.0
        elif "flash" in chosen_model:
            ttft_ms = 45.0 + prompt_tokens * 0.05
            itl_ms = 12.0
        else:  # 8b
            ttft_ms = 35.0 + prompt_tokens * 0.04
            itl_ms = 9.0

        total_latency_ms = ttft_ms + (completion_tokens * itl_ms)

        # Small async pause (scaled down 10x so UI feels snappy during demos)
        sleep_sec = min(0.15, total_latency_ms / 10000.0)
        await asyncio.sleep(sleep_sec)

        return ProviderResponse(
            content=content,
            model=chosen_model,
            provider="simulator",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=round(ttft_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            measurement_source=MeasurementSource.SIMULATED,
            provider_reported_cost=None,
            provider_metadata={
                "simulation_seed": 42,
                "note": "Deterministic simulated output with realistic token physics.",
            },
        )

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            healthy=True,
            provider="simulator",
            latency_ms=0.5,
            message="Deterministic physics simulator active (100% reproducible)",
            available_models=[m.id for m in self.MODELS],
        )

    def _generate_simulated_answer(self, prompt: str, model: str) -> str:
        p_lower = prompt.lower()
        if "redis" in p_lower or "cache" in p_lower:
            return (
                "Redis is an open-source, in-memory data structure store used as a database, "
                "cache, streaming engine, and message broker. By storing data in RAM rather than "
                "disk, it provides sub-millisecond response times for frequent reads and session management."
            )
        elif "code" in p_lower or "python" in p_lower or "def " in p_lower:
            return (
                "```python\ndef optimize_inference(batch_size: int, wait_ms: float):\n"
                "    \"\"\"Calculates optimal queue dispatch window.\"\"\"\n"
                "    effective_throughput = batch_size / (wait_ms / 1000.0)\n"
                "    return round(effective_throughput, 2)\n```"
            )
        elif "why" in p_lower or "reason" in p_lower or "how" in p_lower:
            return (
                f"From an infrastructure perspective, {prompt.rstrip('?')} depends on hardware concurrency "
                f"and batch formation. Under heavy traffic, dynamic routing prevents head-of-line blocking."
            )
        else:
            return (
                f"[Processed by {model}]: INFERCTRL successfully handled your request under current policy. "
                f"Input length: {len(prompt)} characters. Serving metrics and cost accounting recorded."
            )

