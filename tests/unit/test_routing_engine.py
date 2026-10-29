"""Unit tests for the Routing Engine, policies, and constraints."""

import pytest
from inferctrl.cost.calculator import CostCalculator
from inferctrl.observability.metrics import ObservabilityRegistry
from inferctrl.providers.simulator import SimulatorProviderAdapter
from inferctrl.routing.engine import (
    RoutingConstraints,
    RoutingDecision,
    RoutingEngine,
    RoutingStrategy,
)


@pytest.fixture
def routing_setup():
    sim = SimulatorProviderAdapter()
    cost_calc = CostCalculator()
    obs = ObservabilityRegistry()
    providers = {"simulator": sim}
    engine = RoutingEngine(
        providers=providers,
        cost_calculator=cost_calc,
        observability=obs,
        default_strategy=RoutingStrategy.BALANCED,
    )
    return engine


def test_lowest_cost_routing(routing_setup):
    engine = routing_setup
    decision = engine.select_route(
        requested_model="auto",
        strategy=RoutingStrategy.LOWEST_COST,
        prompt_token_estimate=100,
        output_token_estimate=100,
    )
    # Llama 3 8B or Gemini Flash should be chosen over 70B
    assert decision.model_id in ["sim-llama-3-8b", "sim-gemini-flash"]
    assert "Lowest estimated cost" in decision.decision_reason


def test_quality_first_routing(routing_setup):
    engine = routing_setup
    decision = engine.select_route(
        requested_model="auto",
        strategy=RoutingStrategy.QUALITY_FIRST,
    )
    # Llama 3 70B has the highest quality score (0.96)
    assert decision.model_id == "sim-llama-3-70b"
    assert "Highest quality score" in decision.decision_reason
    assert decision.quality_verified is True


def test_round_robin_routing(routing_setup):
    engine = routing_setup
    d1 = engine.select_route(strategy=RoutingStrategy.ROUND_ROBIN)
    d2 = engine.select_route(strategy=RoutingStrategy.ROUND_ROBIN)
    d3 = engine.select_route(strategy=RoutingStrategy.ROUND_ROBIN)

    # Must cycle through models
    assert d1.model_id != d2.model_id or d2.model_id != d3.model_id


def test_hard_constraint_enforcement(routing_setup):
    engine = routing_setup
    # Strict constraint: max cost $0.0000001 (all models exceed)
    # When fallback is disabled, should raise RuntimeError
    engine.constraints = RoutingConstraints(
        max_estimated_cost_per_request=0.000001,
        fallback_enabled=False,
    )
    with pytest.raises(RuntimeError, match="No eligible models satisfied routing constraints"):
        engine.select_route(strategy=RoutingStrategy.BALANCED)

    # When fallback is enabled, constraints are relaxed and routing succeeds with fallback note
    engine.constraints.fallback_enabled = True
    decision = engine.select_route(strategy=RoutingStrategy.BALANCED)
    assert len(decision.rejected_candidates) > 0
    assert "relaxed via fallback" in decision.decision_reason
