"""Provider-aware cost accounting and pricing registry."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelPricing:
    provider: str
    model: str
    input_rate_per_million: float
    output_rate_per_million: float
    currency: str = "USD"
    effective_date: str = "2026-01-01"


DEFAULT_PRICING: Dict[str, ModelPricing] = {
    # Google Gemini Models
    "gemini:gemini-1.5-flash": ModelPricing("gemini", "gemini-1.5-flash", 0.075, 0.30),
    "gemini:gemini-1.5-pro": ModelPricing("gemini", "gemini-1.5-pro", 3.50, 10.50),
    "gemini:gemini-2.0-flash": ModelPricing("gemini", "gemini-2.0-flash", 0.10, 0.40),
    "gemini:gemini-2.5-flash": ModelPricing("gemini", "gemini-2.5-flash", 0.075, 0.30),

    # Ollama Local Models ($0.00 compute)
    "ollama:llama3": ModelPricing("ollama", "llama3", 0.0, 0.0),
    "ollama:llama3:8b": ModelPricing("ollama", "llama3:8b", 0.0, 0.0),
    "ollama:llama3:70b": ModelPricing("ollama", "llama3:70b", 0.0, 0.0),
    "ollama:mistral": ModelPricing("ollama", "mistral", 0.0, 0.0),
    "ollama:qwen2.5-coder": ModelPricing("ollama", "qwen2.5-coder", 0.0, 0.0),

    # Simulator Models
    "simulator:sim-llama-3-8b": ModelPricing("simulator", "sim-llama-3-8b", 0.50, 0.50),
    "simulator:sim-llama-3-70b": ModelPricing("simulator", "sim-llama-3-70b", 4.00, 4.00),
    "simulator:sim-gemini-flash": ModelPricing("simulator", "sim-gemini-flash", 0.075, 0.30),
}


class CostCalculator:
    """Calculates provider-aware inference costs based on token usage and pricing tables."""

    def __init__(self, custom_pricing: Optional[Dict[str, ModelPricing]] = None):
        self.pricing: Dict[str, ModelPricing] = custom_pricing or dict(DEFAULT_PRICING)

    def get_pricing(self, provider: str, model: str) -> Optional[ModelPricing]:
        key = f"{provider.lower()}:{model.lower()}"
        if key in self.pricing:
            return self.pricing[key]
        # Wildcard fallback for provider
        wildcard_key = f"{provider.lower()}:default"
        return self.pricing.get(wildcard_key)

    def calculate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate estimated cost:
        (input_tokens * input_rate + output_tokens * output_rate) / 1,000,000
        """
        pricing = self.get_pricing(provider, model)
        if pricing is None:
            # Fallback default: $1.00 / 1M tokens
            return round(((input_tokens + output_tokens) * 1.0) / 1_000_000.0, 8)

        cost = (
            input_tokens * pricing.input_rate_per_million
            + output_tokens * pricing.output_rate_per_million
        ) / 1_000_000.0

        return round(cost, 8)

