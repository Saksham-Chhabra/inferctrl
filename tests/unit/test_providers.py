"""Unit tests for Provider Adapters (Gemini, Ollama, Simulator)."""

import pytest
from inferctrl.providers.base import MeasurementSource
from inferctrl.providers.gemini import GeminiProviderAdapter
from inferctrl.providers.ollama import OllamaProviderAdapter
from inferctrl.providers.simulator import SimulatorProviderAdapter


@pytest.mark.asyncio
async def test_simulator_provider_generation():
    sim = SimulatorProviderAdapter(random_seed=42)
    assert sim.is_available() is True

    models = sim.list_models()
    assert len(models) >= 3
    assert any(m.id == "sim-llama-3-8b" for m in models)

    messages = [{"role": "user", "content": "Explain Redis caching mechanism"}]
    resp = await sim.generate(messages, model="sim-llama-3-8b")

    assert resp.provider == "simulator"
    assert resp.model == "sim-llama-3-8b"
    assert "Redis" in resp.content
    assert resp.prompt_tokens > 0
    assert resp.completion_tokens > 0
    assert resp.ttft_ms > 0
    assert resp.total_latency_ms >= resp.ttft_ms
    assert resp.measurement_source == MeasurementSource.SIMULATED


@pytest.mark.asyncio
async def test_gemini_provider_unconfigured():
    gemini = GeminiProviderAdapter(api_key="")
    assert gemini.is_available() is False

    health = await gemini.health_check()
    assert health.healthy is False
    assert "not configured" in health.message

    with pytest.raises(RuntimeError, match="not configured"):
        await gemini.generate([{"role": "user", "content": "hello"}])


@pytest.mark.asyncio
async def test_ollama_provider_unreachable():
    ollama = OllamaProviderAdapter(base_url="http://localhost:19999")  # Non-existent port
    assert ollama.is_available() is False

    health = await ollama.health_check()
    assert health.healthy is False
    assert "not running" in health.message or "failed" in health.message

