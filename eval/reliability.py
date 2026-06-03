from typing import Any

from benchmark_builder.common.llm_io import call_json_object_llm
from eval.judge_prompts import SYSTEM_PROMPT, build_reliability_user_prompt


def _safe_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = minimum
    return max(minimum, min(maximum, parsed))


def evaluate_reliability(
    sample: dict[str, Any],
    run_result: dict[str, Any],
    *,
    model: str,
    temperature: float,
    timeout: int,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    try:
        payload = call_json_object_llm(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_reliability_user_prompt(sample, run_result)},
            ],
            temperature=temperature,
            timeout=timeout,
            api_key=api_key,
            base_url=base_url,
        )
        if not isinstance(payload, dict):
            raise TypeError("judge payload is not a JSON object")

        faithfulness_score = _safe_int(payload.get("faithfulness_score"), 0, 4)
        clarification_score = _safe_int(payload.get("clarification_score"), 0, 3)
        tool_discipline_score = _safe_int(payload.get("tool_discipline_score"), 0, 3)
        reliability = 0.4 * (faithfulness_score / 4.0) + 0.3 * (clarification_score / 3.0) + 0.3 * (tool_discipline_score / 3.0)

        return {
            "judge_model": model,
            "faithfulness_score": faithfulness_score,
            "clarification_score": clarification_score,
            "tool_discipline_score": tool_discipline_score,
            "faithfulness_reason": str(payload.get("faithfulness_reason", "")).strip(),
            "clarification_reason": str(payload.get("clarification_reason", "")).strip(),
            "tool_discipline_reason": str(payload.get("tool_discipline_reason", "")).strip(),
            "reliability": reliability,
            "judge_error": None,
        }
    except Exception as exc:
        return {
            "judge_model": model,
            "faithfulness_score": 0,
            "clarification_score": 0,
            "tool_discipline_score": 0,
            "faithfulness_reason": "",
            "clarification_reason": "",
            "tool_discipline_reason": "",
            "reliability": 0.0,
            "judge_error": str(exc),
        }
