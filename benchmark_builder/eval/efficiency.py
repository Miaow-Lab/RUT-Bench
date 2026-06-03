from typing import Any


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def evaluate_efficiency(sample: dict[str, Any], run_result: dict[str, Any], completion_passed: bool) -> dict[str, Any]:
    actual_actions = run_result.get("actual_actions", []) if isinstance(run_result.get("actual_actions"), list) else []
    transcript = run_result.get("transcript", []) if isinstance(run_result.get("transcript"), list) else []

    user_turn_count = sum(1 for item in transcript if item.get("role") == "user")
    assistant_response_turn_count = sum(1 for action in actual_actions if action.get("kind") == "respond")
    assistant_tool_call_count = sum(1 for action in actual_actions if action.get("kind") == "tool")
    total_action_turns = user_turn_count + assistant_response_turn_count + assistant_tool_call_count

    min_required_action_turns = _safe_int(sample.get("min_required_action_turns"), 0)
    if total_action_turns <= 0 or min_required_action_turns <= 0:
        efficiency = 0.0
    else:
        efficiency = min(1.0, min_required_action_turns / float(total_action_turns))

    success = bool(completion_passed)
    sr = 1.0 if success else 0.0
    task_score = sr * efficiency

    return {
        "min_required_action_turns": min_required_action_turns,
        "actual_user_turn_count": user_turn_count,
        "actual_assistant_response_turn_count": assistant_response_turn_count,
        "actual_assistant_tool_call_count": assistant_tool_call_count,
        "actual_total_action_turns": total_action_turns,
        "success": success,
        "sr": sr,
        "efficiency": efficiency,
        "task_score": task_score,
    }
