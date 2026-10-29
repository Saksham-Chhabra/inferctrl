"""Deterministic Discrete-Event LLM Inference Serving Simulator."""

import heapq
import random
from typing import Dict, List, Optional, Tuple
import numpy as np

from inferctrl.schemas import (
    BenchmarkResult,
    LatencyMetrics,
    QualityScore,
    RequestTrace,
    ServingConfig,
    TaskType,
)
from inferctrl.serving.base import ServingAdapter


class ModelProfile:
    def __init__(
        self,
        model_id: str,
        ttft_base_ms: float,
        prefill_rate: float,
        decode_rate: float,
        cost_per_1k: float,
        base_quality: float,
        task_quality: Dict[str, float],
    ):
        self.model_id = model_id
        self.ttft_base_ms = ttft_base_ms
        self.prefill_rate = prefill_rate
        self.decode_rate = decode_rate
        self.cost_per_1k = cost_per_1k
        self.base_quality = base_quality
        self.task_quality = task_quality


DEFAULT_PROFILES = {
    "llama-3-8b": ModelProfile(
        model_id="llama-3-8b",
        ttft_base_ms=35.0,
        prefill_rate=8500.0,
        decode_rate=160.0,
        cost_per_1k=0.0005,
        base_quality=0.88,
        task_quality={
            TaskType.CHAT: 0.91,
            TaskType.CODE: 0.85,
            TaskType.REASONING: 0.82,
            TaskType.SUMMARIZATION: 0.90,
            TaskType.STRUCTURED_JSON: 0.87,
        },
    ),
    "llama-3-70b": ModelProfile(
        model_id="llama-3-70b",
        ttft_base_ms=90.0,
        prefill_rate=2800.0,
        decode_rate=65.0,
        cost_per_1k=0.0040,
        base_quality=0.96,
        task_quality={
            TaskType.CHAT: 0.97,
            TaskType.CODE: 0.95,
            TaskType.REASONING: 0.94,
            TaskType.SUMMARIZATION: 0.96,
            TaskType.STRUCTURED_JSON: 0.96,
        },
    ),
}


class InferenceSimulator(ServingAdapter):
    """High-fidelity, deterministic discrete-event simulator for LLM inference serving."""

    def __init__(self, profiles: Optional[Dict[str, ModelProfile]] = None):
        self.profiles = profiles or DEFAULT_PROFILES
        self.active_config: Optional[ServingConfig] = None
        self.last_known_good_config: Optional[ServingConfig] = None
        self.is_healthy = True

    def validate_config(self, config: ServingConfig) -> Tuple[bool, List[str]]:
        reasons = []
        if config.max_concurrency < 1 or config.max_concurrency > 64:
            reasons.append(f"max_concurrency {config.max_concurrency} outside [1, 64]")
        if config.max_batch_size < 1 or config.max_batch_size > 128:
            reasons.append(f"max_batch_size {config.max_batch_size} outside [1, 128]")
        if config.batch_wait_ms < 0.0 or config.batch_wait_ms > 100.0:
            reasons.append(f"batch_wait_ms {config.batch_wait_ms} outside [0.0, 100.0]")
        for model_id in config.routing_mix.keys():
            if model_id not in self.profiles:
                reasons.append(f"Unknown model_id '{model_id}' in routing_mix")
        return (len(reasons) == 0, reasons)

    def apply_config(self, config: ServingConfig) -> bool:
        valid, _ = self.validate_config(config)
        if not valid:
            return False
        if self.active_config is not None:
            self.last_known_good_config = self.active_config
        self.active_config = config
        return True

    def health_check(self) -> Dict[str, any]:
        return {
            "status": "healthy" if self.is_healthy else "unhealthy",
            "active_config_id": self.active_config.config_id if self.active_config else None,
            "backend": "deterministic_simulator",
        }

    def rollback(self) -> bool:
        if self.last_known_good_config:
            self.active_config = self.last_known_good_config
            return True
        return False

    def run_benchmark(
        self,
        config: ServingConfig,
        traces: List[RequestTrace],
        warmup_count: int = 20,
        random_seed: int = 42,
    ) -> BenchmarkResult:
        rng = random.Random(random_seed)

        models = list(config.routing_mix.keys())
        weights = [config.routing_mix[m] for m in models]

        # Total worker concurrency lanes across the GPU cluster
        num_workers = max(1, config.replica_count * max(1, config.max_concurrency // 4))
        # Min-heap of available worker finish times (in ms)
        worker_available_times = [0.0] * num_workers
        heapq.heapify(worker_available_times)

        queue_times: List[float] = []
        ttfts: List[float] = []
        itls: List[float] = []
        total_latencies: List[float] = []
        total_cost = 0.0
        total_tokens = 0
        failed_count = 0

        sorted_traces = sorted(traces, key=lambda t: t.arrival_time_ms)
        trace_idx = 0
        total_trace_count = len(sorted_traces)

        while trace_idx < total_trace_count:
            # Form a batch of up to max_batch_size requests
            batch_traces: List[RequestTrace] = []
            first_trace = sorted_traces[trace_idx]
            batch_traces.append(first_trace)
            trace_idx += 1

            batch_cutoff_time = first_trace.arrival_time_ms + config.batch_wait_ms

            while (
                trace_idx < total_trace_count
                and len(batch_traces) < config.max_batch_size
                and sorted_traces[trace_idx].arrival_time_ms <= batch_cutoff_time
            ):
                batch_traces.append(sorted_traces[trace_idx])
                trace_idx += 1

            # Dispatch batch to earliest available worker
            earliest_worker_free_time = heapq.heappop(worker_available_times)

            batch_dispatch_time = max(batch_cutoff_time, earliest_worker_free_time)
            batch_size = len(batch_traces)

            # Assign models for requests in batch
            batch_models = [rng.choices(models, weights=weights, k=1)[0] for _ in batch_traces]

            # Batch execution physics:
            # Weighted throughput across models in batch
            total_prompt_tok = sum(t.prompt_tokens for t in batch_traces)
            max_out_tok = max(t.output_tokens for t in batch_traces)

            # Contention scales with batch size beyond 1
            contention = 1.0 + 0.008 * (batch_size - 1)

            # Execution time on worker
            avg_prefill_rate = np.mean([self.profiles[m].prefill_rate for m in batch_models])
            avg_decode_rate = np.mean([self.profiles[m].decode_rate for m in batch_models]) / contention
            avg_base_ttft = np.mean([self.profiles[m].ttft_base_ms for m in batch_models])

            batch_prefill_ms = avg_base_ttft + (total_prompt_tok / (avg_prefill_rate * 1.5)) * 1000.0
            batch_decode_ms = (max_out_tok / max(1.0, avg_decode_rate)) * 1000.0
            batch_finish_time = batch_dispatch_time + batch_prefill_ms + batch_decode_ms

            heapq.heappush(worker_available_times, batch_finish_time)

            # Record metrics for each request in the batch
            for idx_in_batch, (req, assigned_m) in enumerate(zip(batch_traces, batch_models)):
                global_idx = (trace_idx - batch_size) + idx_in_batch
                profile = self.profiles[assigned_m]

                req_queue_time = max(0.0, batch_dispatch_time - req.arrival_time_ms)
                if req_queue_time > 15000.0:
                    failed_count += 1
                    continue

                req_ttft = req_queue_time + batch_prefill_ms
                req_itl = batch_decode_ms / max(1, req.output_tokens)
                req_total = req_ttft + batch_decode_ms

                req_tokens = req.prompt_tokens + req.output_tokens
                req_cost = (req_tokens / 1000.0) * profile.cost_per_1k

                if global_idx >= warmup_count:
                    queue_times.append(req_queue_time)
                    ttfts.append(req_ttft)
                    itls.append(req_itl)
                    total_latencies.append(req_total)
                    total_cost += req_cost
                    total_tokens += req_tokens

        measured_count = len(total_latencies)
        if measured_count == 0:
            raise ValueError("No requests remained after warmup.")

        # Compute duration of measured window
        start_ms = sorted_traces[warmup_count].arrival_time_ms
        end_ms = sorted_traces[-1].arrival_time_ms
        window_sec = max(0.001, (end_ms - start_ms) / 1000.0)

        successful_requests = measured_count
        error_rate = failed_count / max(1, len(sorted_traces))

        # Percentile calculations
        latency_metrics = LatencyMetrics(
            queue_time_p50_ms=float(np.percentile(queue_times, 50)),
            queue_time_p95_ms=float(np.percentile(queue_times, 95)),
            ttft_p50_ms=float(np.percentile(ttfts, 50)),
            ttft_p95_ms=float(np.percentile(ttfts, 95)),
            itl_p50_ms=float(np.percentile(itls, 50)),
            itl_p95_ms=float(np.percentile(itls, 95)),
            total_p50_ms=float(np.percentile(total_latencies, 50)),
            total_p95_ms=float(np.percentile(total_latencies, 95)),
            total_p99_ms=float(np.percentile(total_latencies, 99)),
        )

        # GPU utilization approximation
        avg_concurrency_load = min(1.0, (measured_count / window_sec) / (config.replica_count * config.max_concurrency * 1.5))
        gpu_util = float(round(avg_concurrency_load * 100.0, 1))

        return BenchmarkResult(
            candidate_id=config.config_id,
            run_id="run-benchmark",
            trace_id="synthetic-trace",
            trials_count=1,
            total_requests=measured_count + failed_count,
            successful_requests=successful_requests,
            failed_requests=failed_count,
            error_rate=error_rate,
            latency=latency_metrics,
            throughput_req_per_sec=float(round(successful_requests / window_sec, 2)),
            tokens_per_second=float(round(total_tokens / window_sec, 2)),
            estimated_cost_per_request=float(round(total_cost / max(1, successful_requests), 6)),
            estimated_cost_per_1k_tokens=float(round((total_cost / max(1, total_tokens)) * 1000.0, 6)),
            gpu_utilization_pct=gpu_util,
            is_valid=True,
        )
