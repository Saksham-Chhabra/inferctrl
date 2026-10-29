"""Golden Quality Dataset and Evaluation Suite."""

from typing import Dict, List, Optional
from inferctrl.schemas import QualityScore, ServingConfig, TaskType


class GoldenTestCase:
    def __init__(self, case_id: str, task_type: TaskType, prompt: str, reference_solution: str):
        self.case_id = case_id
        self.task_type = task_type
        self.prompt = prompt
        self.reference_solution = reference_solution


# Representative golden dataset covering critical LLM task categories
GOLDEN_DATASET: List[GoldenTestCase] = [
    GoldenTestCase(
        "code-01",
        TaskType.CODE,
        "Write a Python function to compute the longest palindromic substring.",
        "def longest_palindrome(s): ...",
    ),
    GoldenTestCase(
        "reason-01",
        TaskType.REASONING,
        "A train leaves Station A at 60 mph. Another leaves Station B at 80 mph. They are 280 miles apart. When do they meet?",
        "2 hours.",
    ),
    GoldenTestCase(
        "json-01",
        TaskType.STRUCTURED_JSON,
        "Extract invoice details into JSON conforming to schema: {vendor: str, amount: float, currency: str}.",
        '{"vendor": "Acme Corp", "amount": 1250.50, "currency": "USD"}',
    ),
    GoldenTestCase(
        "summary-01",
        TaskType.SUMMARIZATION,
        "Summarize the executive summary of the quarterly financial statement in 2 sentences.",
        "Revenue increased by 14% year-over-year. Operating margin expanded to 28%.",
    ),
    GoldenTestCase(
        "chat-01",
        TaskType.CHAT,
        "Explain quantum entanglement to a high school student.",
        "Two particles become connected so measuring one instantly dictates the state of the other...",
    ),
]


class QualityEvaluator:
    """Evaluates output quality across task classes and identifies regressions."""

    def __init__(self, model_task_quality: Optional[Dict[str, Dict[TaskType, float]]] = None):
        # Base task qualities per model
        self.model_task_quality = model_task_quality or {
            "llama-3-8b": {
                TaskType.CHAT: 0.91,
                TaskType.CODE: 0.85,
                TaskType.REASONING: 0.82,
                TaskType.SUMMARIZATION: 0.90,
                TaskType.STRUCTURED_JSON: 0.87,
            },
            "llama-3-70b": {
                TaskType.CHAT: 0.97,
                TaskType.CODE: 0.95,
                TaskType.REASONING: 0.94,
                TaskType.SUMMARIZATION: 0.96,
                TaskType.STRUCTURED_JSON: 0.96,
            },
        }

    def evaluate_config(
        self,
        config: ServingConfig,
        baseline_score: float = 0.93,
        max_allowed_regression_pct: float = 1.0,
    ) -> QualityScore:
        """Calculate quality scores across all task classes given the routing mix."""
        task_scores: Dict[str, float] = {}
        subgroup_regressions: Dict[str, float] = {}

        tasks = [TaskType.CHAT, TaskType.CODE, TaskType.REASONING, TaskType.SUMMARIZATION, TaskType.STRUCTURED_JSON]

        for task in tasks:
            weighted_score = 0.0
            for model_id, weight in config.routing_mix.items():
                m_scores = self.model_task_quality.get(model_id, {})
                score = m_scores.get(task, 0.85)
                weighted_score += score * weight

            task_scores[task.value] = round(weighted_score, 4)

        overall = round(sum(task_scores.values()) / len(task_scores), 4)
        overall_delta = round((overall - baseline_score) * 100.0, 2)  # In percentage points

        passed_guardrails = True
        rejection_reason = None

        # Check for regression beyond allowed budget
        regression_budget_abs = max_allowed_regression_pct / 100.0
        if (baseline_score - overall) > regression_budget_abs:
            passed_guardrails = False
            rejection_reason = (
                f"Overall quality score {overall:.3f} regressed by "
                f"{abs(overall_delta):.2f}% vs baseline {baseline_score:.3f} (max allowed: {max_allowed_regression_pct}%)"
            )

        # Check for individual subgroup severe regressions (> 2.0%)
        for task_name, score in task_scores.items():
            if score < 0.84:
                subgroup_regressions[task_name] = score
                if passed_guardrails:
                    passed_guardrails = False
                    rejection_reason = f"Subgroup task '{task_name}' quality regressed to {score:.3f} (< 0.84 threshold)"

        return QualityScore(
            candidate_id=config.config_id,
            overall_score=overall,
            baseline_score=baseline_score,
            score_delta=overall_delta,
            task_scores=task_scores,
            subgroup_regressions=subgroup_regressions,
            schema_valid_fraction=1.0,
            is_conclusive=True,
            passed_guardrails=passed_guardrails,
            rejection_reason=rejection_reason,
        )

