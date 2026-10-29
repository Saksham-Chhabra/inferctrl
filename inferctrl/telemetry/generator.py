"""Synthetic workload and request trace generator."""

import random
from typing import List
from inferctrl.schemas import RequestTrace, TaskType


class TraceGenerator:
    """Generates deterministic request traces for benchmarking and simulation."""

    @staticmethod
    def generate_trace(
        scenario: str = "steady_short",
        request_count: int = 150,
        random_seed: int = 42,
    ) -> List[RequestTrace]:
        rng = random.Random(random_seed)
        traces: List[RequestTrace] = []
        current_time_ms = 0.0

        task_types = [
            (TaskType.CHAT, 0.35),
            (TaskType.CODE, 0.20),
            (TaskType.REASONING, 0.25),
            (TaskType.SUMMARIZATION, 0.10),
            (TaskType.STRUCTURED_JSON, 0.10),
        ]
        tasks, task_weights = zip(*task_types)

        if scenario == "steady_short":
            base_rps = 8.0
            prompt_range = (150, 350)
            output_range = (50, 150)
        elif scenario == "bursty_spike":
            base_rps = 18.0
            prompt_range = (200, 500)
            output_range = (60, 180)
        elif scenario == "long_context":
            base_rps = 5.0
            prompt_range = (1500, 3000)
            output_range = (120, 250)
        else:  # mixed
            base_rps = 12.0
            prompt_range = (180, 800)
            output_range = (60, 180)

        for i in range(request_count):
            inter_arrival_sec = rng.expovariate(base_rps)
            if scenario == "bursty_spike" and (30 <= i <= 70):
                inter_arrival_sec *= 0.40  # Burst window

            current_time_ms += inter_arrival_sec * 1000.0

            chosen_task = rng.choices(tasks, weights=task_weights, k=1)[0]
            
            # Task-specific adjustments
            if chosen_task == TaskType.REASONING:
                p_tokens = rng.randint(prompt_range[0], int(prompt_range[1] * 1.5))
                o_tokens = rng.randint(int(output_range[0] * 1.5), int(output_range[1] * 2.0))
            elif chosen_task == TaskType.SUMMARIZATION:
                p_tokens = rng.randint(int(prompt_range[0] * 2.0), int(prompt_range[1] * 2.5))
                o_tokens = rng.randint(output_range[0], output_range[1])
            else:
                p_tokens = rng.randint(prompt_range[0], prompt_range[1])
                o_tokens = rng.randint(output_range[0], output_range[1])

            traces.append(
                RequestTrace(
                    request_id=f"req-{i+1:04d}",
                    arrival_time_ms=round(current_time_ms, 2),
                    task_type=chosen_task,
                    prompt_tokens=p_tokens,
                    output_tokens=o_tokens,
                    endpoint="cluster-primary",
                )
            )

        return traces
