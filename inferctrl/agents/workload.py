"""Workload Analyst Agent: Statistical drift detection and regime classification."""

from typing import List, Optional
import numpy as np

from inferctrl.schemas import RequestTrace, WorkloadFeatures, WorkloadReport


class WorkloadAnalystAgent:
    """Agent that ingests recent request traces and characterizes workload regimes and drift."""

    def __init__(self, baseline_rps: float = 25.0, baseline_prompt_p95: float = 400.0):
        self.baseline_rps = baseline_rps
        self.baseline_prompt_p95 = baseline_prompt_p95

    def analyze_window(
        self,
        traces: List[RequestTrace],
        window_seconds: Optional[float] = None,
        current_queue_depth: int = 0,
    ) -> WorkloadReport:
        if not traces:
            effective_window = window_seconds or 60.0
            return WorkloadReport(
                regime="unknown",
                confidence=0.0,
                window_seconds=effective_window,
                features=WorkloadFeatures(
                    window_seconds=effective_window,
                    total_requests=0,
                    requests_per_second=0.0,
                    p50_prompt_tokens=0.0,
                    p95_prompt_tokens=0.0,
                    p50_output_tokens=0.0,
                    p95_output_tokens=0.0,
                    burstiness_index=1.0,
                    current_queue_depth=current_queue_depth,
                    error_rate=0.0,
                ),
                drift_detected=False,
                evidence=["No requests in window."],
                recommended_experiment_focus=[],
            )

        total_requests = len(traces)
        if window_seconds is not None:
            effective_window = window_seconds
        elif len(traces) > 1:
            effective_window = max(1.0, (traces[-1].arrival_time_ms - traces[0].arrival_time_ms) / 1000.0)
        else:
            effective_window = 1.0

        actual_rps = total_requests / max(0.5, effective_window)

        prompt_tokens = [t.prompt_tokens for t in traces]
        output_tokens = [t.output_tokens for t in traces]

        # Calculate arrival inter-times for burstiness index
        arrival_times = [t.arrival_time_ms for t in traces]
        if len(arrival_times) > 2:
            inter_arrivals = np.diff(arrival_times)
            mean_ia = np.mean(inter_arrivals)
            std_ia = np.std(inter_arrivals)
            burstiness = float(round(std_ia / max(1e-4, mean_ia), 2))
        else:
            burstiness = 1.0

        p50_p = float(np.percentile(prompt_tokens, 50))
        p95_p = float(np.percentile(prompt_tokens, 95))
        p50_o = float(np.percentile(output_tokens, 50))
        p95_o = float(np.percentile(output_tokens, 95))

        # Task distribution
        task_counts = {}
        for t in traces:
            task_counts[t.task_type.value] = task_counts.get(t.task_type.value, 0) + 1
        task_dist = {k: round(v / total_requests, 3) for k, v in task_counts.items()}

        features = WorkloadFeatures(
            window_seconds=round(effective_window, 2),
            total_requests=total_requests,
            requests_per_second=round(actual_rps, 2),
            p50_prompt_tokens=round(p50_p, 1),
            p95_prompt_tokens=round(p95_p, 1),
            p50_output_tokens=round(p50_o, 1),
            p95_output_tokens=round(p95_o, 1),
            burstiness_index=burstiness,
            current_queue_depth=current_queue_depth,
            error_rate=0.0,
            task_distribution=task_dist,
        )

        evidence: List[str] = []
        drift_detected = False
        recommended_focus: List[str] = []

        # Drift detection rules
        rps_ratio = actual_rps / max(1.0, self.baseline_rps)
        if rps_ratio >= 1.5 or burstiness > 1.4:
            drift_detected = True
            evidence.append(f"Arrival rate / burstiness surged over rolling baseline ({actual_rps:.1f} RPS, burstiness {burstiness})")
            recommended_focus.extend(["max_concurrency", "max_batch_size", "batch_wait_ms"])

        if p95_p >= (self.baseline_prompt_p95 * 1.5):
            drift_detected = True
            evidence.append(f"Prompt length P95 increased to {p95_p:.0f} tokens (>1.5x baseline {self.baseline_prompt_p95:.0f})")
            recommended_focus.extend(["routing_mix", "batch_wait_ms"])

        if current_queue_depth > 15:
            drift_detected = True
            evidence.append(f"Queue depth elevated at {current_queue_depth} requests")
            if "max_concurrency" not in recommended_focus:
                recommended_focus.append("max_concurrency")

        # Classify regime
        if rps_ratio >= 1.5 or burstiness > 1.4:
            regime = "bursty"
            confidence = 0.92
        elif p95_p > 2000.0:
            regime = "long_context"
            confidence = 0.89
        elif actual_rps < (self.baseline_rps * 0.4):
            regime = "low_traffic"
            confidence = 0.85
        else:
            regime = "steady_short"
            confidence = 0.95

        return WorkloadReport(
            regime=regime,
            confidence=confidence,
            window_seconds=round(effective_window, 2),
            features=features,
            drift_detected=drift_detected,
            evidence=evidence if evidence else ["Workload is within nominal operating range."],
            recommended_experiment_focus=list(set(recommended_focus)),
        )
