"""Gateway Service: Dispatches chat completions through routing, providers, and observability."""

import time
import uuid
from typing import Dict, List, Optional

from inferctrl.cost.calculator import CostCalculator
from inferctrl.gateway.models import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    InferctrlMetadata,
    UsageMetadata,
)
from inferctrl.observability.metrics import ObservabilityRegistry, RequestLogRecord
from inferctrl.providers.base import BaseProviderAdapter, MeasurementSource, ModelMetadata
from inferctrl.routing.engine import RoutingDecision, RoutingEngine, RoutingStrategy


class GatewayService:
    """Core integration gateway for AI applications."""

    def __init__(
        self,
        providers: Dict[str, BaseProviderAdapter],
        routing_engine: RoutingEngine,
        cost_calculator: CostCalculator,
        observability: ObservabilityRegistry,
    ):
        self.providers = providers
        self.router = routing_engine
        self.cost_calculator = cost_calculator
        self.observability = observability

    async def execute_chat_completion(
        self,
        req: ChatCompletionRequest,
        app_id: str = "default",
    ) -> ChatCompletionResponse:
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        monotonic_start = time.perf_counter()

        # Parse routing strategy override if provided
        chosen_strategy = None
        if req.routing_strategy:
            try:
                chosen_strategy = RoutingStrategy(req.routing_strategy.lower())
            except ValueError:
                pass

        # Estimate prompt tokens
        raw_prompt = " ".join([m.content for m in req.messages])
        prompt_token_est = max(10, int(len(raw_prompt.split()) * 1.3))

        # 1. Routing Decision
        route_decision: RoutingDecision = self.router.select_route(
            requested_model=req.model,
            strategy=chosen_strategy,
            prompt_token_estimate=prompt_token_est,
        )

        selected_provider = self.providers.get(route_decision.provider_name)
        if not selected_provider or not selected_provider.is_available():
            # Trigger fallback to simulator if live provider failed
            sim_provider = self.providers.get("simulator")
            if sim_provider:
                selected_provider = sim_provider
                route_decision.provider_name = "simulator"
                route_decision.model_id = sim_provider.list_models()[0].id
                route_decision.decision_reason += " (Fallback: Selected provider unavailable)"
            else:
                raise RuntimeError(f"Provider '{route_decision.provider_name}' is not available.")

        # 2. Dispatch Generation to Provider
        formatted_messages = [{"role": m.role, "content": m.content} for m in req.messages]
        try:
            prov_resp = await selected_provider.generate(
                messages=formatted_messages,
                model=route_decision.model_id,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
            status = "success"
            error_category = "none"
        except Exception as e:
            # Automatic fallback to simulator on provider exception
            sim_provider = self.providers.get("simulator")
            if sim_provider and selected_provider != sim_provider:
                selected_provider = sim_provider
                prov_resp = await sim_provider.generate(
                    messages=formatted_messages,
                    model="sim-llama-3-8b",
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                )
                status = "fallback"
                error_category = f"provider_error: {str(e)[:40]}"
                route_decision.provider_name = "simulator"
                route_decision.model_id = "sim-llama-3-8b"
                route_decision.decision_reason += f" (Auto-recovered after provider error: {str(e)[:40]})"
            else:
                raise e

        total_elapsed_ms = (time.perf_counter() - monotonic_start) * 1000.0

        # 3. Cost Calculation
        estimated_cost = self.cost_calculator.calculate_cost(
            provider=prov_resp.provider,
            model=prov_resp.model,
            input_tokens=prov_resp.prompt_tokens,
            output_tokens=prov_resp.completion_tokens,
        )

        # 4. Observability Logging (privacy-conscious: preview prompt only)
        prompt_preview = raw_prompt[:40] + "..." if len(raw_prompt) > 40 else raw_prompt
        log_record = RequestLogRecord(
            request_id=request_id,
            app_id=app_id,
            requested_model=req.model,
            selected_provider=prov_resp.provider,
            selected_model=prov_resp.model,
            routing_strategy=route_decision.strategy.value,
            decision_reason=route_decision.decision_reason,
            ttft_ms=prov_resp.ttft_ms,
            total_latency_ms=round(total_elapsed_ms, 2),
            prompt_tokens=prov_resp.prompt_tokens,
            completion_tokens=prov_resp.completion_tokens,
            total_tokens=prov_resp.prompt_tokens + prov_resp.completion_tokens,
            estimated_cost_usd=estimated_cost,
            provider_reported_cost=prov_resp.provider_reported_cost,
            status=status,
            error_category=error_category,
            measurement_source=prov_resp.measurement_source,
            quality_score=0.92,
            prompt_preview=prompt_preview,
        )
        self.observability.log_request(log_record)

        # 5. Format OpenAI-compatible response with INFERCTRL metadata
        return ChatCompletionResponse(
            id=request_id,
            object="chat.completion",
            created=int(time.time()),
            model=prov_resp.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=prov_resp.content),
                    finish_reason="stop",
                )
            ],
            usage=UsageMetadata(
                prompt_tokens=prov_resp.prompt_tokens,
                completion_tokens=prov_resp.completion_tokens,
                total_tokens=prov_resp.prompt_tokens + prov_resp.completion_tokens,
            ),
            inferctrl=InferctrlMetadata(
                request_id=request_id,
                provider=prov_resp.provider,
                selected_model=prov_resp.model,
                routing_strategy=route_decision.strategy.value,
                decision_reason=route_decision.decision_reason,
                ttft_ms=prov_resp.ttft_ms,
                total_latency_ms=round(total_elapsed_ms, 2),
                estimated_cost_usd=estimated_cost,
                provider_reported_cost=prov_resp.provider_reported_cost,
                measurement_source=prov_resp.measurement_source.value,
                quality_verified=route_decision.quality_verified,
                eligible_models=route_decision.eligible_candidates,
            ),
        )

    def list_all_models(self) -> List[dict]:
        models = []
        for p_name, adapter in self.providers.items():
            for m in adapter.list_models():
                pricing = self.cost_calculator.get_pricing(p_name, m.id)
                cost_1k = (
                    (pricing.input_rate_per_million + pricing.output_rate_per_million) / 2.0 / 1000.0
                    if pricing
                    else 0.001
                )
                models.append(
                    {
                        "id": f"{p_name}:{m.id}",
                        "model": m.id,
                        "provider": p_name,
                        "name": m.name,
                        "context_length": m.context_length,
                        "is_available": m.is_available,
                        "quality_score": m.quality_score,
                        "quality_verified": m.quality_verified,
                        "estimated_cost_per_1k": round(cost_1k, 6),
                    }
                )
        return models

