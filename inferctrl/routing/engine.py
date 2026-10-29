"""Routing Engine with configurable policies and constraint enforcement."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from inferctrl.cost.calculator import CostCalculator
from inferctrl.observability.metrics import ObservabilityRegistry
from inferctrl.providers.base import BaseProviderAdapter, ModelMetadata


class RoutingStrategy(str, Enum):
    FIXED = "fixed"
    ROUND_ROBIN = "round_robin"
    LOWEST_COST = "lowest_cost"
    LOWEST_LATENCY = "lowest_latency"
    QUALITY_FIRST = "quality_first"
    BALANCED = "balanced"


@dataclass
class RoutingConstraints:
    max_latency_ms: float = 5000.0
    max_estimated_cost_per_request: float = 0.05
    min_quality_score: float = 0.80
    fallback_enabled: bool = True


@dataclass
class RoutingDecision:
    provider_name: str
    model_id: str
    strategy: RoutingStrategy
    decision_reason: str
    eligible_candidates: List[str]
    rejected_candidates: Dict[str, str] = field(default_factory=dict)
    quality_verified: bool = False
    estimated_cost_per_1k: float = 0.0


class RoutingEngine:
    """Selects the optimal provider and model for each inference request."""

    def __init__(
        self,
        providers: Dict[str, BaseProviderAdapter],
        cost_calculator: CostCalculator,
        observability: ObservabilityRegistry,
        default_strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        constraints: Optional[RoutingConstraints] = None,
    ):
        self.providers = providers
        self.cost_calculator = cost_calculator
        self.observability = observability
        self.default_strategy = default_strategy
        self.constraints = constraints or RoutingConstraints()
        self._round_robin_idx = 0

    def select_route(
        self,
        requested_model: Optional[str] = None,
        strategy: Optional[RoutingStrategy] = None,
        task_type: str = "chat",
        prompt_token_estimate: int = 100,
        output_token_estimate: int = 150,
    ) -> RoutingDecision:
        active_strategy = strategy or self.default_strategy

        # Gather all models from all providers
        all_models: List[Tuple[str, ModelMetadata]] = []
        for p_name, adapter in self.providers.items():
            if adapter.is_available():
                for m in adapter.list_models():
                    if m.is_available:
                        all_models.append((p_name, m))

        if not all_models:
            # Fallback to simulator if available
            sim = self.providers.get("simulator")
            if sim:
                m = sim.list_models()[0]
                return RoutingDecision(
                    provider_name="simulator",
                    model_id=m.id,
                    strategy=active_strategy,
                    decision_reason="Fallback: No live external providers available; routed to deterministic simulator.",
                    eligible_candidates=[m.id],
                    quality_verified=True,
                )
            raise RuntimeError("No available LLM providers configured.")

        # If user explicitly requested a specific model (and not 'auto' / 'inferctrl-auto')
        if requested_model and requested_model not in ["auto", "inferctrl-auto", "default"]:
            req_clean = requested_model.lower()
            for p_name, m in all_models:
                if m.id.lower() == req_clean or f"{p_name}:{m.id}".lower() == req_clean:
                    cost_1k = self.cost_calculator.calculate_cost(p_name, m.id, 500, 500) * 1000.0
                    return RoutingDecision(
                        provider_name=p_name,
                        model_id=m.id,
                        strategy=RoutingStrategy.FIXED,
                        decision_reason=f"Explicitly requested model '{requested_model}'.",
                        eligible_candidates=[m.id],
                        quality_verified=m.quality_verified,
                        estimated_cost_per_1k=cost_1k,
                    )

        # Filter candidates by hard constraints
        eligible: List[Tuple[str, ModelMetadata]] = []
        rejected: Dict[str, str] = {}

        summary = self.observability.compute_summary(window_seconds=600.0)

        for p_name, m in all_models:
            # Cost check
            cost = self.cost_calculator.calculate_cost(
                p_name, m.id, prompt_token_estimate, output_token_estimate
            )
            if cost > self.constraints.max_estimated_cost_per_request:
                rejected[f"{p_name}:{m.id}"] = (
                    f"Estimated cost ${cost:.4f} exceeds limit (${self.constraints.max_estimated_cost_per_request:.4f})"
                )
                continue

            # Quality check
            if m.quality_score is not None and m.quality_score < self.constraints.min_quality_score:
                rejected[f"{p_name}:{m.id}"] = (
                    f"Quality score {m.quality_score:.2f} below required {self.constraints.min_quality_score:.2f}"
                )
                continue

            eligible.append((p_name, m))

        # If all candidates violated constraints and fallback enabled, relax constraints
        if not eligible:
            if self.constraints.fallback_enabled:
                eligible = list(all_models)
                fallback_note = " (Constraints relaxed via fallback policy)"
            else:
                raise RuntimeError(
                    f"No eligible models satisfied routing constraints. Rejected: {list(rejected.keys())}"
                )
        else:
            fallback_note = ""

        # Apply active routing strategy
        if active_strategy == RoutingStrategy.ROUND_ROBIN:
            p_name, m = eligible[self._round_robin_idx % len(eligible)]
            self._round_robin_idx += 1
            reason = f"Round-robin turn {self._round_robin_idx}{fallback_note}."

        elif active_strategy == RoutingStrategy.LOWEST_COST:
            # Sort by lowest estimated cost
            eligible.sort(
                key=lambda x: self.cost_calculator.calculate_cost(
                    x[0], x[1].id, prompt_token_estimate, output_token_estimate
                )
            )
            p_name, m = eligible[0]
            c = self.cost_calculator.calculate_cost(p_name, m.id, prompt_token_estimate, output_token_estimate)
            reason = f"Lowest estimated cost: ${c:.6f} per request{fallback_note}."

        elif active_strategy == RoutingStrategy.QUALITY_FIRST:
            # Sort by highest quality (default 0.85 if unverified)
            eligible.sort(key=lambda x: (x[1].quality_score or 0.85), reverse=True)
            p_name, m = eligible[0]
            q_val = m.quality_score or 0.85
            q_status = "verified" if m.quality_verified else "unverified estimate"
            reason = f"Highest quality score: {q_val:.2f} ({q_status}){fallback_note}."

        elif active_strategy == RoutingStrategy.LOWEST_LATENCY:
            # Prefer fast models (flash, 8b, or local ollama)
            def latency_rank(item):
                p, mod = item
                if "flash" in mod.id or "8b" in mod.id:
                    return 1
                if p == "ollama":
                    return 2
                return 3

            eligible.sort(key=latency_rank)
            p_name, m = eligible[0]
            reason = f"Fastest recent profile for low-latency turnaround{fallback_note}."

        else:  # BALANCED
            def balanced_score(item):
                p, mod = item
                cost = self.cost_calculator.calculate_cost(p, mod.id, 500, 500) * 1000.0  # cost per 1k
                # Lower score is better
                lat_weight = 0.4
                cost_weight = 0.4
                qual_weight = 0.2

                norm_cost = min(1.0, cost / 0.005)
                norm_lat = 0.3 if ("flash" in mod.id or "8b" in mod.id) else 0.8
                norm_qual = 1.0 - (mod.quality_score or 0.85)

                return (cost_weight * norm_cost) + (lat_weight * norm_lat) + (qual_weight * norm_qual)

            eligible.sort(key=balanced_score)
            p_name, m = eligible[0]
            reason = f"Optimal balanced Pareto trade-off between cost, latency, and verified quality{fallback_note}."

        cost_1k = self.cost_calculator.calculate_cost(p_name, m.id, 500, 500) * 1000.0

        return RoutingDecision(
            provider_name=p_name,
            model_id=m.id,
            strategy=active_strategy,
            decision_reason=reason,
            eligible_candidates=[f"{p}:{mod.id}" for p, mod in eligible],
            rejected_candidates=rejected,
            quality_verified=m.quality_verified,
            estimated_cost_per_1k=cost_1k,
        )
