import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_builder.common.env_runtime import build_env_runtime, stable_json_dumps, summarize_state_population
from benchmark_builder.common.llm_io import call_json_object_llm
from benchmark_builder.common.serialization import read_json_file, save_json_file
from benchmark_builder.config import (
    DEFAULT_INIT_STATE_BANK,
    DEFAULT_STAGE2_ENV_MAX_STEPS,
    DEFAULT_STAGE2_MAX_ATTEMPTS_PER_TASK,
    DEFAULT_STAGE2_MAX_ENVS,
    DEFAULT_STAGE2_MAX_WORKERS,
    DEFAULT_STAGE2_MIN_BLUEPRINTS_PER_ENV,
    DEFAULT_STAGE2_MIN_DIFFICULTY_BUCKETS,
    DEFAULT_STAGE2_MAX_STATES_PER_ENV,
    DEFAULT_STAGE2_MODEL,
    DEFAULT_STAGE2_REQUIRED_DIALOGUE_MODES,
    DEFAULT_STAGE2_TASKS_PER_STATE,
    DEFAULT_STAGE2_TEMPERATURE,
    DEFAULT_STAGE2_TIMEOUT,
    DEFAULT_TASK_BLUEPRINTS,
    STAGE2_PROMPT_STATE_CHAR_LIMIT,
)
from utils.auto_env import get_state_diff


SYSTEM_PROMPT = """
You are an expert benchmark task designer for an agent tool-calling environment.

Design exactly one executable task blueprint and one oracle tool plan.

Requirements:
1. Use only entities that actually exist in the provided init_config.
2. Prefer tasks with at least one objective state change on structured fields.
3. The user-facing request must not mention internal IDs, tool names, code details, or hidden state keys.
4. The tool plan must be directly executable from the provided initial state, and every tool step must return success during replay.
5. Standard execute tasks should use between 1 and 6 tool calls. A rare 0-tool plan is allowed only when task_type is clarify_then_execute or refuse_or_scope.
6. Set agent_action_budget to an integer no larger than 50.
7. If a later step needs a newly created entity, explicitly choose its ID in the creation step so the plan stays deterministic.
8. Prefer realistic tasks that require discovery before mutation, but prefer concrete verification tools over high-level helper tools when the state hints already expose the relevant candidate IDs.
9. Avoid plans that depend on unavailable equipment types, nonexistent future slots, or helper tools whose success is uncertain from the current state.
10. If dialogue_mode is multi_turn, provide 2-3 coherent turn-level tasks.
11. Set task_type to one of execute, clarify_then_execute, or refuse_or_scope. Prefer execute unless the prompt explicitly calls for a clarification-heavy or refusal-heavy task.
12. If task_type is clarify_then_execute or refuse_or_scope, include the necessary clarification_requirements or refusal_requirements and any forbidden_actions.
13. If the prompt gives an explicit coverage target for this attempt, satisfy it exactly.
14. Output JSON only.

Return an object with this schema:
{
  "goal_summary": "short benchmark goal",
  "user_request": "natural user-facing request without IDs",
  "dialogue_mode": "single_turn or multi_turn",
  "difficulty_bucket": "easy or medium or hard",
    "task_type": "execute or clarify_then_execute or refuse_or_scope",
  "tool_call_budget": 2,
  "agent_action_budget": 12,
  "expected_phases": ["phase 1", "phase 2"],
  "expected_final_outcome": "objective expected outcome",
  "relevant_entities": ["entity ids or values central to the task"],
  "quality_checks": "why the task is benchmark-worthy",
  "unique_answer_rationale": "why there is one correct answer before conflict",
  "canonical_answer": "exact expected end result",
  "fallback_answer_hint": "backup answer if natural, else none",
  "fallback_inferior_reason": "why fallback is worse, else none",
    "clarification_requirements": ["required clarification or missing fact, else []"],
    "refusal_requirements": ["required refusal or scope boundary, else []"],
    "forbidden_actions": ["tool/action the agent must not do, else []"],
  "multi_turn_tasks": [
        {
            "task_summary": "turn goal",
            "expected_outcome": "turn outcome",
            "task_type": "execute or clarify_then_execute or refuse_or_scope",
            "required_entities": ["optional entity ids or values"]
        }
  ],
  "tool_plan": [
    {
      "tool_name": "exact tool name",
      "arguments": {"arg": "value"},
      "purpose": "why this tool is called"
    }
  ]
}
""".strip()


VALID_TASK_TYPES = {"execute", "clarify_then_execute", "refuse_or_scope"}

FORMAL_BLUEPRINT_SLOT_TEMPLATES = [
    {
        "slot_name": "T1",
        "dialogue_mode": "single_turn",
        "difficulty_bucket": "easy",
        "tool_call_range": {"min": 1, "max": 2},
        "task_type": "execute",
    },
    {
        "slot_name": "T2",
        "dialogue_mode": "single_turn",
        "difficulty_bucket": "medium",
        "tool_call_range": {"min": 2, "max": 4},
        "task_type": "execute",
    },
    {
        "slot_name": "T3",
        "dialogue_mode": "multi_turn",
        "difficulty_bucket": "medium",
        "tool_call_range": {"min": 3, "max": 5},
        "task_type": "execute",
    },
    {
        "slot_name": "T4",
        "dialogue_mode": "multi_turn",
        "difficulty_bucket": "hard",
        "tool_call_range": {"min": 4, "max": 6},
        "task_type": "execute",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate smoke-scoped task blueprints and validated gold traces from an init state bank.")
    parser.add_argument("--input-file", default=str(DEFAULT_INIT_STATE_BANK), help="Path to init_state_bank.json")
    parser.add_argument("--output-file", default=str(DEFAULT_TASK_BLUEPRINTS), help="Path to task_blueprints.json")
    parser.add_argument("--model", default=DEFAULT_STAGE2_MODEL, help="LLM model used for blueprint generation")
    parser.add_argument("--temperature", type=float, default=DEFAULT_STAGE2_TEMPERATURE, help="LLM temperature")
    parser.add_argument("--tasks-per-state", type=int, default=DEFAULT_STAGE2_TASKS_PER_STATE, help="Accepted blueprints per init state")
    parser.add_argument("--max-attempts-per-task", type=int, default=DEFAULT_STAGE2_MAX_ATTEMPTS_PER_TASK, help="LLM attempts per accepted blueprint")
    parser.add_argument("--max-envs", type=int, default=DEFAULT_STAGE2_MAX_ENVS, help="Optional limit for smoke runs. 0 means all envs.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_STAGE2_MAX_WORKERS, help="Worker count across environments")
    parser.add_argument("--max-states-per-env", type=int, default=DEFAULT_STAGE2_MAX_STATES_PER_ENV, help="Limit state_bank entries per env")
    parser.add_argument("--min-blueprints-per-env", type=int, default=DEFAULT_STAGE2_MIN_BLUEPRINTS_PER_ENV, help="Optional env-level minimum accepted blueprint count")
    parser.add_argument("--required-dialogue-modes", default=DEFAULT_STAGE2_REQUIRED_DIALOGUE_MODES, help="Comma-separated env-level required dialogue modes, e.g. single_turn,multi_turn")
    parser.add_argument("--min-difficulty-buckets", type=int, default=DEFAULT_STAGE2_MIN_DIFFICULTY_BUCKETS, help="Optional env-level minimum count of distinct difficulty buckets")
    parser.add_argument("--env-max-steps", type=int, default=DEFAULT_STAGE2_ENV_MAX_STEPS, help="Max env steps during gold-trace replay")
    parser.add_argument("--timeout", type=int, default=DEFAULT_STAGE2_TIMEOUT, help="LLM timeout in seconds")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key override for Stage2")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"), help="OpenAI base URL override for Stage2")
    parser.add_argument("--progress-interval", type=int, default=30, help="Seconds between Stage2 progress heartbeat logs. 0 disables heartbeat logs.")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds_int = max(0, int(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds_part:02d}"
    return f"{minutes:02d}:{seconds_part:02d}"


def _print_stage2_progress(
    *,
    completed_count: int,
    total_count: int,
    start_time: float,
    blueprint_count: int,
    failure_count: int,
    latest_env_id: str = "",
    latest_blueprint_count: int | None = None,
    latest_failure_count: int | None = None,
    latest_coverage_passed: bool | None = None,
) -> None:
    elapsed = max(time.monotonic() - start_time, 0.0)
    if completed_count > 0:
        avg_seconds_per_env = elapsed / completed_count
        eta_seconds = avg_seconds_per_env * max(total_count - completed_count, 0)
    else:
        eta_seconds = None

    parts = [
        f"[Stage2 progress] envs={completed_count}/{total_count}",
        f"blueprints={blueprint_count}",
        f"failures={failure_count}",
        f"elapsed={_format_duration(elapsed)}",
        f"eta={_format_duration(eta_seconds)}",
    ]
    if latest_env_id:
        latest_parts = [f"latest_env={latest_env_id}"]
        if latest_blueprint_count is not None:
            latest_parts.append(f"latest_blueprints={latest_blueprint_count}")
        if latest_failure_count is not None:
            latest_parts.append(f"latest_failures={latest_failure_count}")
        if latest_coverage_passed is not None:
            latest_parts.append(f"latest_coverage_passed={latest_coverage_passed}")
        parts.append(" ".join(latest_parts))
    print(" | ".join(parts), flush=True)


def _state_bank_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return [item for item in payload["items"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise TypeError("State bank payload must be a list or a dict with an items array")


def _compact_json(data: Any, max_chars: int) -> str:
    text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "\n...<truncated>..."
    return text


def _build_tool_list(tools: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for tool in tools:
        params = tool.get("parameters", {}) if isinstance(tool.get("parameters"), dict) else {}
        props = params.get("properties", {}) if isinstance(params.get("properties"), dict) else {}
        required = set(params.get("required", [])) if isinstance(params.get("required"), list) else set()
        param_text = ", ".join(f"{name}{'*' if name in required else ''}" for name in props) or "(none)"
        lines.append(f"- {tool.get('name', '')}({param_text}): {tool.get('description', '')}")
    return "\n".join(lines)


def _looks_like_identifier(value: str) -> bool:
    token = str(value or "").strip()
    if not token:
        return False
    return any(char in token for char in ("_", "-")) or any(char.isdigit() for char in token)


def _detect_query_leaks(user_request: str, tools: list[dict[str, Any]], relevant_entities: list[Any]) -> dict[str, list[str]]:
    text = (user_request or "").lower()
    tool_hits = [tool.get("name", "") for tool in tools if tool.get("name") and tool["name"].lower() in text]

    entity_hits: list[str] = []
    for entity in relevant_entities:
        token = str(entity).strip()
        if token and _looks_like_identifier(token) and token.lower() in text:
            entity_hits.append(token)

    return {
        "tool_hits": tool_hits,
        "entity_hits": entity_hits,
    }


def _build_user_prompt(
    state_item: dict[str, Any],
    state_entry: dict[str, Any],
    accepted_summaries: list[str],
    failure_reasons: list[str],
    coverage_target: dict[str, Any],
) -> str:
    accepted_block = "\n".join(f"- {summary}" for summary in accepted_summaries[-3:]) or "- none"
    failure_block = "\n".join(f"- {reason}" for reason in failure_reasons[-4:]) or "- none"
    target_required_mode = coverage_target.get("required_dialogue_mode") or "none"
    target_required_bucket = coverage_target.get("required_difficulty_bucket") or "none"
    missing_modes = coverage_target.get("missing_required_dialogue_modes", [])
    present_modes = coverage_target.get("present_dialogue_modes", [])
    present_buckets = coverage_target.get("present_difficulty_buckets", [])
    require_new_bucket = bool(coverage_target.get("require_new_difficulty_bucket"))
    slot_name = coverage_target.get("slot_name") or "none"
    tool_call_range = coverage_target.get("target_tool_call_range") or {}
    min_tool_calls = tool_call_range.get("min", "none")
    max_tool_calls = tool_call_range.get("max", "none")
    preferred_task_type = coverage_target.get("preferred_task_type") or "execute"
    return f"""
Environment ID: {state_item.get('env_id', '')}
Environment summary:
{state_item.get('environment_summary', '')}

Environment introduction:
{state_item.get('environment_introduction', '')}

Constraint rules:
{state_item.get('constraints_rules_text', '')}

Available tools:
{_build_tool_list(state_item.get('tools', []))}

Current execution hints:
{_build_state_hints(state_entry.get('init_config', {}))}

Initial state profile:
{state_entry.get('state_profile', '')}

Initial state snapshot:
{_compact_json(state_entry.get('init_config', {}), STAGE2_PROMPT_STATE_CHAR_LIMIT)}

Already accepted task summaries for this state:
{accepted_block}

Recent rejected attempts:
{failure_block}

Current env-level coverage target for this attempt:
- accepted blueprint count so far: {coverage_target.get('accepted_blueprint_count', 0)}
- remaining blueprint slots across this env: {coverage_target.get('remaining_env_slots', 0)}
- formal slot target: {slot_name}
- present dialogue modes: {', '.join(present_modes) or 'none'}
- missing required dialogue modes: {', '.join(missing_modes) or 'none'}
- required dialogue_mode for this attempt: {target_required_mode}
- present difficulty buckets: {', '.join(present_buckets) or 'none'}
- must use a new difficulty bucket this attempt: {'yes' if require_new_bucket else 'no'}
- required difficulty bucket for this attempt: {target_required_bucket}
- target tool call range for this attempt: {min_tool_calls}-{max_tool_calls}
- preferred task_type for this attempt: {preferred_task_type}

Design one executable benchmark task blueprint now.
""".strip()


def _build_state_hints(init_config: dict[str, Any]) -> str:
    providers = init_config.get("providers", {}) if isinstance(init_config.get("providers"), dict) else {}
    patients = init_config.get("patients", {}) if isinstance(init_config.get("patients"), dict) else {}
    appointments = init_config.get("appointments", {}) if isinstance(init_config.get("appointments"), dict) else {}
    timeslots = init_config.get("timeslots", {}) if isinstance(init_config.get("timeslots"), dict) else {}
    equipment = init_config.get("equipment", {}) if isinstance(init_config.get("equipment"), dict) else {}

    active_appointments: list[str] = []
    for appointment_id, appointment in sorted(appointments.items()):
        if not isinstance(appointment, dict):
            continue
        status = str(appointment.get("status", "")).strip().lower()
        if status in {"cancelled", "completed"}:
            continue
        active_appointments.append(
            f"- {appointment_id}: patient={appointment.get('patient_id')} provider={appointment.get('provider_id')} type={appointment.get('appointment_type')} date={appointment.get('date')} time={appointment.get('time')}"
        )

    available_slots: list[str] = []
    for slot_id, slot in sorted(timeslots.items()):
        if not isinstance(slot, dict) or not slot.get("is_available"):
            continue
        available_slots.append(
            f"- {slot_id}: provider={slot.get('provider_id')} date={slot.get('date')} {slot.get('start_time')}-{slot.get('end_time')}"
        )

    available_equipment: list[str] = []
    for equipment_id, item in sorted(equipment.items()):
        if not isinstance(item, dict):
            continue
        if not item.get("availability_status"):
            continue
        if str(item.get("maintenance_status", "")).strip().lower() != "ok":
            continue
        available_equipment.append(f"- {equipment_id}: type={item.get('type')} location={item.get('location')}")

    return "\n".join(
        [
            f"- provider_ids: {', '.join(sorted(providers.keys())[:8]) or 'none'}",
            f"- patient_ids: {', '.join(sorted(patients.keys())[:8]) or 'none'}",
            "- active appointments:",
            *(active_appointments[:6] or ["- none"]),
            "- available timeslots:",
            *(available_slots[:8] or ["- none"]),
            "- currently available equipment:",
            *(available_equipment[:8] or ["- none"]),
        ]
    )


def _normalize_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _normalize_list_of_strings(value: Any, *, min_items: int = 0) -> list[str]:
    if not isinstance(value, list):
        return []
    result = [_normalize_string(item) for item in value if _normalize_string(item)]
    if len(result) < min_items:
        return []
    return result


def _normalize_micro_task_list(value: Any, default_task_type: str, default_entities: list[Any]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in value[:3]:
        if not isinstance(item, dict):
            continue
        task_summary = _normalize_string(item.get("task_summary"))
        expected_outcome = _normalize_string(item.get("expected_outcome"))
        task_type = _normalize_string(item.get("task_type"), default_task_type)
        if task_type not in VALID_TASK_TYPES:
            task_type = default_task_type
        required_entities = item.get("required_entities") if isinstance(item.get("required_entities"), list) else default_entities
        normalized_entities = [entity for entity in required_entities if str(entity).strip()]
        if task_summary and expected_outcome:
            normalized.append(
                {
                    "task_summary": task_summary,
                    "expected_outcome": expected_outcome,
                    "task_type": task_type,
                    "required_entities": normalized_entities,
                }
            )
    return normalized


def _parse_required_dialogue_modes(raw_value: str) -> list[str]:
    ordered_modes = ["single_turn", "multi_turn"]
    requested = {part.strip() for part in str(raw_value or "").split(",") if part.strip()}
    return [mode for mode in ordered_modes if mode in requested]


def _new_env_generation_progress() -> dict[str, Any]:
    return {
        "accepted_blueprint_count": 0,
        "accepted_summaries": [],
        "dialogue_modes_present": set(),
        "difficulty_buckets_present": set(),
        "seen_fingerprints": set(),
        "accepted_task_ids": [],
    }


def _next_preferred_difficulty_bucket(present_buckets: set[str]) -> str | None:
    for bucket in ("easy", "medium", "hard"):
        if bucket not in present_buckets:
            return bucket
    return None


def _select_slot_template(accepted_blueprint_count: int, total_target_count: int) -> dict[str, Any] | None:
    if total_target_count < 4:
        return None
    if accepted_blueprint_count < len(FORMAL_BLUEPRINT_SLOT_TEMPLATES):
        return FORMAL_BLUEPRINT_SLOT_TEMPLATES[accepted_blueprint_count]
    return None


def _build_coverage_target(progress: dict[str, Any], args: argparse.Namespace, remaining_env_slots: int) -> dict[str, Any]:
    accepted_blueprint_count = int(progress.get("accepted_blueprint_count", 0))
    total_target_count = accepted_blueprint_count + remaining_env_slots
    slot_template = _select_slot_template(accepted_blueprint_count, total_target_count)

    present_dialogue_modes = set(progress.get("dialogue_modes_present", set()))
    required_dialogue_modes = _parse_required_dialogue_modes(args.required_dialogue_modes)
    missing_required_dialogue_modes = [mode for mode in required_dialogue_modes if mode not in present_dialogue_modes]

    present_difficulty_buckets = set(progress.get("difficulty_buckets_present", set()))
    missing_difficulty_bucket_count = max(0, args.min_difficulty_buckets - len(present_difficulty_buckets))
    require_new_difficulty_bucket = missing_difficulty_bucket_count > 0 and remaining_env_slots <= missing_difficulty_bucket_count

    required_dialogue_mode = slot_template.get("dialogue_mode") if slot_template else None
    if required_dialogue_mode is None and missing_required_dialogue_modes:
        required_dialogue_mode = missing_required_dialogue_modes[0]

    required_difficulty_bucket = slot_template.get("difficulty_bucket") if slot_template else None
    if required_difficulty_bucket is None and require_new_difficulty_bucket:
        required_difficulty_bucket = _next_preferred_difficulty_bucket(present_difficulty_buckets)

    return {
        "accepted_blueprint_count": accepted_blueprint_count,
        "remaining_env_slots": remaining_env_slots,
        "slot_name": slot_template.get("slot_name") if slot_template else None,
        "present_dialogue_modes": sorted(present_dialogue_modes),
        "missing_required_dialogue_modes": missing_required_dialogue_modes,
        "required_dialogue_mode": required_dialogue_mode,
        "present_difficulty_buckets": sorted(present_difficulty_buckets),
        "require_new_difficulty_bucket": require_new_difficulty_bucket,
        "required_difficulty_bucket": required_difficulty_bucket,
        "preferred_difficulty_bucket": _next_preferred_difficulty_bucket(present_difficulty_buckets),
        "target_tool_call_range": _json_safe_snapshot(slot_template.get("tool_call_range", {})) if slot_template else None,
        "preferred_task_type": slot_template.get("task_type", "execute") if slot_template else "execute",
    }


def _validate_candidate_against_coverage_target(candidate: dict[str, Any], coverage_target: dict[str, Any]) -> str | None:
    required_dialogue_mode = coverage_target.get("required_dialogue_mode")
    if required_dialogue_mode and candidate.get("dialogue_mode") != required_dialogue_mode:
        return f"candidate dialogue_mode must be '{required_dialogue_mode}' for current coverage target"

    required_difficulty_bucket = coverage_target.get("required_difficulty_bucket")
    if required_difficulty_bucket and candidate.get("difficulty_bucket") != required_difficulty_bucket:
        return f"candidate difficulty_bucket must be '{required_difficulty_bucket}' for current coverage target"

    if coverage_target.get("require_new_difficulty_bucket"):
        present_buckets = set(coverage_target.get("present_difficulty_buckets", []))
        if candidate.get("difficulty_bucket") in present_buckets:
            return "candidate must introduce a new difficulty bucket for current coverage target"

    tool_call_range = coverage_target.get("target_tool_call_range")
    if isinstance(tool_call_range, dict):
        min_calls = int(tool_call_range.get("min", 0) or 0)
        max_calls = int(tool_call_range.get("max", 0) or 0)
        actual_calls = len(candidate.get("tool_plan", [])) if isinstance(candidate.get("tool_plan"), list) else 0
        if min_calls > 0 and actual_calls < min_calls:
            return f"candidate tool_plan length {actual_calls} is below required min {min_calls}"
        if max_calls > 0 and actual_calls > max_calls:
            return f"candidate tool_plan length {actual_calls} exceeds required max {max_calls}"

    preferred_task_type = _normalize_string(coverage_target.get("preferred_task_type"))
    if preferred_task_type == "execute" and candidate.get("task_type") != "execute":
        return "candidate task_type must remain execute for the current formal slot target"

    return None


def _build_final_state_assertions(value: Any, path: str = "") -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []

    if set(value.keys()) == {"changed"} and isinstance(value.get("changed"), dict):
        return [
            {
                "path": path,
                "assertion_type": "changed",
                "old_value": _json_safe_snapshot(value["changed"].get("old")),
                "new_value": _json_safe_snapshot(value["changed"].get("new")),
            }
        ]
    if set(value.keys()) == {"added"}:
        return [{"path": path, "assertion_type": "added", "value": _json_safe_snapshot(value.get("added"))}]
    if set(value.keys()) == {"removed"}:
        return [{"path": path, "assertion_type": "removed", "value": _json_safe_snapshot(value.get("removed"))}]

    assertions: list[dict[str, Any]] = []
    for key, nested in value.items():
        nested_path = f"{path}.{key}" if path else str(key)
        assertions.extend(_build_final_state_assertions(nested, nested_path))
    return assertions


def _build_micro_tasks(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    task_type = candidate.get("task_type", "execute")
    relevant_entities = candidate.get("relevant_entities", []) if isinstance(candidate.get("relevant_entities"), list) else []
    multi_turn_tasks = candidate.get("multi_turn_tasks", []) if isinstance(candidate.get("multi_turn_tasks"), list) else []

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(multi_turn_tasks, start=1):
        if not isinstance(item, dict):
            continue
        task_summary = _normalize_string(item.get("task_summary"))
        expected_outcome = _normalize_string(item.get("expected_outcome"))
        item_task_type = _normalize_string(item.get("task_type"), task_type)
        if item_task_type not in VALID_TASK_TYPES:
            item_task_type = task_type
        required_entities = item.get("required_entities") if isinstance(item.get("required_entities"), list) else relevant_entities
        normalized_entities = [entity for entity in required_entities if str(entity).strip()]
        if task_summary and expected_outcome:
            normalized.append(
                {
                    "task_id": f"subtask_{index}",
                    "user_goal": task_summary,
                    "task_type": item_task_type,
                    "required_entities": normalized_entities,
                    "expected_outcome": expected_outcome,
                }
            )

    if normalized:
        return normalized

    return [
        {
            "task_id": "subtask_1",
            "user_goal": _normalize_string(candidate.get("goal_summary")),
            "task_type": task_type,
            "required_entities": [entity for entity in relevant_entities if str(entity).strip()],
            "expected_outcome": _normalize_string(candidate.get("expected_final_outcome")),
        }
    ]


def _build_canonical_action_graph(gold_trace: list[dict[str, Any]], candidate: dict[str, Any]) -> dict[str, Any]:
    required_read_nodes: list[dict[str, Any]] = []
    required_read_equivalence_sets: list[dict[str, Any]] = []
    required_write_nodes: list[dict[str, Any]] = []
    precedence_edges: list[dict[str, int]] = []
    previous_step_index: int | None = None

    for step in gold_trace:
        if not isinstance(step, dict):
            continue
        step_index = int(step.get("step_index", 0) or 0)
        node = {
            "step_index": step_index,
            "tool_name": _normalize_string(step.get("tool_name")),
            "arguments": _json_safe_snapshot(step.get("arguments", {})),
            "purpose": _normalize_string(step.get("purpose")),
        }
        state_diff = step.get("state_diff", {}) if isinstance(step.get("state_diff"), dict) else {}
        if state_diff:
            node["expected_state_diff"] = _json_safe_snapshot(state_diff)
            required_write_nodes.append(node)
        else:
            required_read_nodes.append(node)
            required_read_equivalence_sets.append(
                {
                    "step_index": step_index,
                    "equivalent_nodes": [node],
                }
            )
        if previous_step_index is not None and step_index > 0:
            precedence_edges.append({"before_step_index": previous_step_index, "after_step_index": step_index})
        if step_index > 0:
            previous_step_index = step_index

    clarify_nodes = [
        {"requirement": requirement, "required_before_write": True}
        for requirement in candidate.get("clarification_requirements", [])
        if _normalize_string(requirement)
    ]
    refuse_nodes = [
        {"requirement": requirement, "required": True}
        for requirement in candidate.get("refusal_requirements", [])
        if _normalize_string(requirement)
    ]

    return {
        "graph_type": "linear_action_graph_v2",
        "task_type": candidate.get("task_type", "execute"),
        "required_read_nodes": required_read_nodes,
        "required_read_equivalence_sets": required_read_equivalence_sets,
        "required_write_nodes": required_write_nodes,
        "clarify_nodes": clarify_nodes,
        "refuse_nodes": refuse_nodes,
        "precedence_edges": precedence_edges,
        "forbidden_actions": _json_safe_snapshot(candidate.get("forbidden_actions", [])),
    }


def _infer_min_required_action_turns(candidate: dict[str, Any], gold_trace: list[dict[str, Any]]) -> int:
    multi_turn_tasks = candidate.get("multi_turn_tasks", []) if isinstance(candidate.get("multi_turn_tasks"), list) else []
    user_turn_count = len(multi_turn_tasks) if candidate.get("dialogue_mode") == "multi_turn" else 1
    if user_turn_count <= 0:
        user_turn_count = 1
    return user_turn_count + len(gold_trace) + 1


def _validate_candidate_schema(candidate: Any, state_item: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(candidate, dict):
        return None, "LLM response is not a JSON object"

    goal_summary = _normalize_string(candidate.get("goal_summary"))
    user_request = _normalize_string(candidate.get("user_request"))
    expected_final_outcome = _normalize_string(candidate.get("expected_final_outcome"))
    if not goal_summary or not user_request or not expected_final_outcome:
        return None, "candidate is missing goal_summary, user_request, or expected_final_outcome"

    dialogue_mode = _normalize_string(candidate.get("dialogue_mode"), "single_turn")
    if dialogue_mode not in {"single_turn", "multi_turn"}:
        return None, "dialogue_mode must be single_turn or multi_turn"

    difficulty_bucket = _normalize_string(candidate.get("difficulty_bucket"), "medium")
    if difficulty_bucket not in {"easy", "medium", "hard"}:
        return None, "difficulty_bucket must be easy, medium, or hard"

    task_type = _normalize_string(candidate.get("task_type"), "execute")
    if task_type not in VALID_TASK_TYPES:
        return None, "task_type must be execute, clarify_then_execute, or refuse_or_scope"

    try:
        tool_call_budget = int(candidate.get("tool_call_budget", 0))
    except (TypeError, ValueError):
        return None, "tool_call_budget must be an integer"
    if tool_call_budget < 0 or tool_call_budget > 6:
        return None, "tool_call_budget must be within 0-6"

    try:
        agent_action_budget = int(candidate.get("agent_action_budget", 0))
    except (TypeError, ValueError):
        return None, "agent_action_budget must be an integer"
    if agent_action_budget < 1 or agent_action_budget > 50:
        return None, "agent_action_budget must be within 1-50"

    expected_phases = _normalize_list_of_strings(candidate.get("expected_phases"), min_items=1)
    if not expected_phases:
        return None, "expected_phases must contain at least one non-empty item"

    relevant_entities = []
    if isinstance(candidate.get("relevant_entities"), list):
        relevant_entities = [item for item in candidate.get("relevant_entities", []) if str(item).strip()]

    clarification_requirements = _normalize_list_of_strings(candidate.get("clarification_requirements"))
    refusal_requirements = _normalize_list_of_strings(candidate.get("refusal_requirements"))
    forbidden_actions = _normalize_list_of_strings(candidate.get("forbidden_actions"))

    if dialogue_mode == "multi_turn":
        raw_multi_turn_tasks = candidate.get("multi_turn_tasks")
        if not isinstance(raw_multi_turn_tasks, list):
            return None, "multi_turn_tasks must be a list for multi_turn tasks"
        normalized_multi_turn = _normalize_micro_task_list(raw_multi_turn_tasks, task_type, relevant_entities)
        if len(normalized_multi_turn) < 2:
            return None, "multi_turn tasks need at least two coherent turn entries"
    else:
        normalized_multi_turn = [
            {
                "task_summary": goal_summary,
                "expected_outcome": expected_final_outcome,
                "task_type": task_type,
                "required_entities": relevant_entities,
            }
        ]

    tool_plan = candidate.get("tool_plan")
    if not isinstance(tool_plan, list):
        return None, "tool_plan must be a list"
    if len(tool_plan) < 0 or len(tool_plan) > 6:
        return None, "tool_plan length must be within 0-6"
    if len(tool_plan) == 0 and task_type == "execute":
        return None, "execute tasks must contain at least one tool step"
    if len(tool_plan) == 0 and task_type == "clarify_then_execute" and not clarification_requirements:
        return None, "clarify_then_execute tasks without tool steps must include clarification_requirements"
    if len(tool_plan) == 0 and task_type == "refuse_or_scope" and not refusal_requirements:
        return None, "refuse_or_scope tasks without tool steps must include refusal_requirements"

    known_tools = {
        tool.get("name", ""): tool
        for tool in state_item.get("tools", [])
        if isinstance(tool, dict) and tool.get("name")
    }
    normalized_plan: list[dict[str, Any]] = []
    for step_index, step in enumerate(tool_plan, 1):
        if not isinstance(step, dict):
            return None, f"tool_plan step {step_index} is not an object"
        tool_name = _normalize_string(step.get("tool_name"))
        if tool_name not in known_tools:
            return None, f"tool_plan step {step_index} uses unknown tool '{tool_name}'"
        arguments = step.get("arguments", {})
        if not isinstance(arguments, dict):
            return None, f"tool_plan step {step_index} arguments must be an object"
        purpose = _normalize_string(step.get("purpose"), f"step {step_index}")
        normalized_plan.append(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "purpose": purpose,
            }
        )

    if tool_call_budget < len(normalized_plan):
        return None, "tool_call_budget is smaller than the proposed tool plan length"

    leaks = _detect_query_leaks(user_request, state_item.get("tools", []), relevant_entities)
    if leaks["tool_hits"] or leaks["entity_hits"]:
        return None, f"user_request leaks hidden tool names or identifiers: {leaks}"

    normalized = {
        "goal_summary": goal_summary,
        "user_request": user_request,
        "dialogue_mode": dialogue_mode,
        "difficulty_bucket": difficulty_bucket,
        "task_type": task_type,
        "tool_call_budget": tool_call_budget,
        "tool_call_range": {"min": len(normalized_plan), "max": tool_call_budget},
        "agent_action_budget": agent_action_budget,
        "expected_phases": expected_phases,
        "expected_final_outcome": expected_final_outcome,
        "relevant_entities": relevant_entities,
        "quality_checks": _normalize_string(candidate.get("quality_checks"), "not provided"),
        "unique_answer_rationale": _normalize_string(candidate.get("unique_answer_rationale"), "not provided"),
        "canonical_answer": _normalize_string(candidate.get("canonical_answer"), expected_final_outcome),
        "fallback_answer_hint": _normalize_string(candidate.get("fallback_answer_hint"), "none"),
        "fallback_inferior_reason": _normalize_string(candidate.get("fallback_inferior_reason"), "none"),
        "clarification_requirements": clarification_requirements,
        "refusal_requirements": refusal_requirements,
        "forbidden_actions": forbidden_actions,
        "multi_turn_tasks": normalized_multi_turn,
        "tool_plan": normalized_plan,
    }
    return normalized, None


def _observation_failed(observation: Any) -> bool:
    if isinstance(observation, dict):
        if "error" in observation:
            return True
        if observation.get("success") is False:
            return True
    return False


def _json_safe_snapshot(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _replay_candidate(
    state_item: dict[str, Any],
    state_entry: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_steps: int,
) -> tuple[dict[str, Any] | None, str | None]:
    env = build_env_runtime(state_item, max_steps=max_steps)
    init_config = deepcopy(state_entry.get("init_config", {}))
    env.env_init(init_config=init_config)
    initial_state = env.get_state_info()

    if not candidate["tool_plan"]:
        replay_validation = {
            "is_valid": True,
            "executed_tool_count": 0,
            "mutating_tool_count": 0,
            "final_state_changed": False,
            "triggered_dynamic_events": env.get_dynamic_event_summary(),
        }
        return {
            "gold_trace": [],
            "final_state_diff": {},
            "final_state_assertions": [],
            "canonical_action_graph": _build_canonical_action_graph([], candidate),
            "micro_tasks": _build_micro_tasks(candidate),
            "min_required_action_turns": _infer_min_required_action_turns(candidate, []),
            "replay_validation": replay_validation,
        }, None

    gold_trace: list[dict[str, Any]] = []
    mutating_steps = 0

    for step_index, step in enumerate(candidate["tool_plan"], 1):
        action = {
            "name": step["tool_name"],
            "params": deepcopy(step["arguments"]),
        }
        observation, reward, terminated, truncated, info = env.env_step(action)
        step_record = env.trajectory[-1]
        state_diff = step_record.get("state_diff", {}) if isinstance(step_record, dict) else {}
        if state_diff:
            mutating_steps += 1

        gold_trace.append(
            {
                "step_index": step_index,
                "tool_name": step["tool_name"],
                "arguments": deepcopy(step["arguments"]),
                "purpose": step.get("purpose", ""),
                "observation": _json_safe_snapshot(observation),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": _json_safe_snapshot(info),
                "state_diff": _json_safe_snapshot(state_diff),
            }
        )

        if _observation_failed(observation):
            return None, f"tool step {step_index} failed: {observation}"
        if terminated or truncated:
            return None, f"environment terminated early at tool step {step_index}"

    final_state = env.get_state_info()
    final_state_diff = get_state_diff(initial_state, final_state)
    if not final_state_diff:
        return None, "replay produced no objective state change"
    if mutating_steps < 1:
        return None, "replay did not include any mutating tool step"

    replay_validation = {
        "is_valid": True,
        "executed_tool_count": len(gold_trace),
        "mutating_tool_count": mutating_steps,
        "final_state_changed": True,
        "triggered_dynamic_events": env.get_dynamic_event_summary(),
    }
    return {
        "gold_trace": gold_trace,
        "final_state_diff": final_state_diff,
        "final_state_assertions": _build_final_state_assertions(final_state_diff),
        "canonical_action_graph": _build_canonical_action_graph(gold_trace, candidate),
        "micro_tasks": _build_micro_tasks(candidate),
        "min_required_action_turns": _infer_min_required_action_turns(candidate, gold_trace),
        "replay_validation": replay_validation,
    }, None


def _candidate_fingerprint(candidate: dict[str, Any]) -> str:
    return stable_json_dumps(
        {
            "goal_summary": candidate.get("goal_summary", ""),
            "user_request": candidate.get("user_request", ""),
            "tool_plan": candidate.get("tool_plan", []),
        }
    )


def _state_item_output_base(state_item: dict[str, Any], state_entry: dict[str, Any]) -> dict[str, Any]:
    validation = state_entry.get("validation", {}) if isinstance(state_entry.get("validation"), dict) else {}
    state_stats = validation.get("state_stats", {}) if isinstance(validation.get("state_stats"), dict) else {}
    if not state_stats:
        state_stats = summarize_state_population(
            state_entry.get("init_config", {}),
            state_item.get("state_container_names", []),
        )

    return {
        "env_id": state_item.get("env_id", ""),
        "config_id": state_entry.get("config_id", ""),
        "environment_summary": state_item.get("environment_summary", ""),
        "environment_introduction": state_item.get("environment_introduction", ""),
        "env_class_name": state_item.get("env_class_name", ""),
        "env_class_code": state_item.get("env_class_code", ""),
        "constraints_rules_list": state_item.get("constraints_rules_list", []),
        "constraints_rules_text": state_item.get("constraints_rules_text", ""),
        "tools": state_item.get("tools", []),
        "tool_summary": state_item.get("tool_summary", []),
        "tool_summary_text": state_item.get("tool_summary_text", ""),
        "state_container_names": state_item.get("state_container_names", []),
        "method_names": state_item.get("method_names", []),
        "complexity": state_item.get("complexity", {}),
        "target_profile": state_entry.get("target_profile", ""),
        "state_profile": state_entry.get("state_profile", state_stats.get("state_profile", "unknown")),
        "state_stats": state_stats,
        "init_config": state_entry.get("init_config", {}),
        "requested_blueprint_count": 0,
        "task_blueprints": [],
        "task_generation_failures": [],
        "blueprint_coverage": {},
    }


def _build_state_blueprint_coverage(result: dict[str, Any]) -> dict[str, Any]:
    blueprints = result.get("task_blueprints", []) if isinstance(result.get("task_blueprints"), list) else []
    dialogue_modes_present = sorted(
        {
            blueprint.get("dialogue_mode", "")
            for blueprint in blueprints
            if isinstance(blueprint, dict) and blueprint.get("dialogue_mode")
        }
    )
    difficulty_buckets_present = sorted(
        {
            blueprint.get("difficulty_bucket", "")
            for blueprint in blueprints
            if isinstance(blueprint, dict) and blueprint.get("difficulty_bucket")
        }
    )
    requested_blueprint_count = int(result.get("requested_blueprint_count", 0))
    realized_blueprint_count = len(blueprints)
    coverage_failures: list[str] = []
    if requested_blueprint_count > realized_blueprint_count:
        coverage_failures.append(
            f"realized_blueprint_count {realized_blueprint_count} is below requested_blueprint_count {requested_blueprint_count}"
        )

    return {
        "requested_blueprint_count": requested_blueprint_count,
        "realized_blueprint_count": realized_blueprint_count,
        "dialogue_modes_present": dialogue_modes_present,
        "difficulty_buckets_present": difficulty_buckets_present,
        "coverage_failures": coverage_failures,
        "coverage_passed": not coverage_failures,
    }


def _update_env_generation_progress(progress: dict[str, Any], blueprint: dict[str, Any], fingerprint: str) -> None:
    progress["accepted_blueprint_count"] = int(progress.get("accepted_blueprint_count", 0)) + 1
    progress.setdefault("accepted_summaries", []).append(blueprint.get("goal_summary", ""))
    progress.setdefault("dialogue_modes_present", set()).add(blueprint.get("dialogue_mode", "single_turn"))
    progress.setdefault("difficulty_buckets_present", set()).add(blueprint.get("difficulty_bucket", "medium"))
    progress.setdefault("seen_fingerprints", set()).add(fingerprint)


def _build_env_coverage_summary(
    env_id: str,
    env_results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    requested_blueprint_count = sum(
        int(result.get("requested_blueprint_count", 0))
        for result in env_results
        if isinstance(result, dict)
    )
    realized_blueprints: list[dict[str, Any]] = []
    for result in env_results:
        if not isinstance(result, dict):
            continue
        blueprints = result.get("task_blueprints", []) if isinstance(result.get("task_blueprints"), list) else []
        realized_blueprints.extend(blueprints)

    realized_blueprint_count = len(realized_blueprints)
    required_dialogue_modes = _parse_required_dialogue_modes(args.required_dialogue_modes)
    dialogue_modes_present = sorted(
        {
            blueprint.get("dialogue_mode", "")
            for blueprint in realized_blueprints
            if isinstance(blueprint, dict) and blueprint.get("dialogue_mode")
        }
    )
    missing_dialogue_modes = [mode for mode in required_dialogue_modes if mode not in set(dialogue_modes_present)]
    difficulty_buckets_present = sorted(
        {
            blueprint.get("difficulty_bucket", "")
            for blueprint in realized_blueprints
            if isinstance(blueprint, dict) and blueprint.get("difficulty_bucket")
        }
    )

    coverage_failures: list[str] = []
    if args.min_blueprints_per_env > 0 and realized_blueprint_count < args.min_blueprints_per_env:
        coverage_failures.append(
            f"realized_blueprint_count {realized_blueprint_count} is below min_blueprints_per_env {args.min_blueprints_per_env}"
        )
    if missing_dialogue_modes:
        coverage_failures.append(f"missing required dialogue modes: {missing_dialogue_modes}")
    if args.min_difficulty_buckets > 0 and len(difficulty_buckets_present) < args.min_difficulty_buckets:
        coverage_failures.append(
            f"distinct difficulty bucket count {len(difficulty_buckets_present)} is below min_difficulty_buckets {args.min_difficulty_buckets}"
        )

    return {
        "env_id": env_id,
        "requested_blueprint_count": requested_blueprint_count,
        "realized_blueprint_count": realized_blueprint_count,
        "dialogue_modes_present": dialogue_modes_present,
        "missing_dialogue_modes": missing_dialogue_modes,
        "difficulty_buckets_present": difficulty_buckets_present,
        "min_required_blueprints": args.min_blueprints_per_env,
        "required_dialogue_modes": required_dialogue_modes,
        "min_required_difficulty_buckets": args.min_difficulty_buckets,
        "coverage_failures": coverage_failures,
        "coverage_passed": not coverage_failures,
    }


def _process_state_entry(
    state_item: dict[str, Any],
    state_entry: dict[str, Any],
    env_progress: dict[str, Any],
    remaining_env_slots_after_state: int,
    args: argparse.Namespace,
    target_blueprint_count: int | None = None,
    existing_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = existing_result if isinstance(existing_result, dict) else _state_item_output_base(state_item, state_entry)
    failure_reasons = [
        str(item.get("reason", ""))
        for item in result.get("task_generation_failures", [])
        if isinstance(item, dict) and item.get("reason")
    ]
    requested_blueprint_count = max(0, target_blueprint_count if target_blueprint_count is not None else args.tasks_per_state)
    result["requested_blueprint_count"] = requested_blueprint_count

    while len(result["task_blueprints"]) < requested_blueprint_count:
        target_index = len(result["task_blueprints"])
        accepted = False
        remaining_env_slots = remaining_env_slots_after_state + (requested_blueprint_count - len(result["task_blueprints"]))
        coverage_target = _build_coverage_target(env_progress, args, remaining_env_slots)

        for attempt in range(1, args.max_attempts_per_task + 1):
            try:
                payload = call_json_object_llm(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _build_user_prompt(
                                state_item,
                                state_entry,
                                env_progress.get("accepted_summaries", []),
                                failure_reasons,
                                coverage_target,
                            ),
                        },
                    ],
                    temperature=args.temperature,
                    timeout=args.timeout,
                    api_key=getattr(args, "api_key", None),
                    base_url=getattr(args, "base_url", None),
                )
                candidate, schema_error = _validate_candidate_schema(payload, state_item)
                if schema_error:
                    failure_reasons.append(schema_error)
                    result["task_generation_failures"].append(
                        {
                            "target_task_index": target_index + 1,
                            "attempt": attempt,
                            "reason": schema_error,
                        }
                    )
                    continue

                coverage_error = _validate_candidate_against_coverage_target(candidate, coverage_target)
                if coverage_error:
                    failure_reasons.append(coverage_error)
                    result["task_generation_failures"].append(
                        {
                            "target_task_index": target_index + 1,
                            "attempt": attempt,
                            "reason": coverage_error,
                            "goal_summary": candidate.get("goal_summary", ""),
                            "coverage_target": coverage_target,
                        }
                    )
                    continue

                fingerprint = _candidate_fingerprint(candidate)
                if fingerprint in env_progress.get("seen_fingerprints", set()):
                    reason = "duplicate blueprint candidate"
                    failure_reasons.append(reason)
                    result["task_generation_failures"].append(
                        {
                            "target_task_index": target_index + 1,
                            "attempt": attempt,
                            "reason": reason,
                            "goal_summary": candidate.get("goal_summary", ""),
                        }
                    )
                    continue

                replay_result, replay_error = _replay_candidate(
                    state_item,
                    state_entry,
                    candidate,
                    max_steps=args.env_max_steps,
                )
                if replay_error:
                    failure_reasons.append(replay_error)
                    result["task_generation_failures"].append(
                        {
                            "target_task_index": target_index + 1,
                            "attempt": attempt,
                            "reason": replay_error,
                            "goal_summary": candidate.get("goal_summary", ""),
                            "tool_plan": candidate.get("tool_plan", []),
                        }
                    )
                    continue

                task_id = f"{result['env_id']}_{result['config_id']}_task_{len(result['task_blueprints']) + 1}"
                result["task_blueprints"].append(
                    {
                        "task_id": task_id,
                        "generated_at": _now_iso(),
                        "llm_model": args.model,
                        "llm_temperature": args.temperature,
                        **candidate,
                        **replay_result,
                    }
                )
                _update_env_generation_progress(env_progress, candidate, fingerprint)
                accepted = True
                break
            except Exception as exc:
                reason = f"exception during task generation: {exc}"
                failure_reasons.append(reason)
                result["task_generation_failures"].append(
                    {
                        "target_task_index": target_index + 1,
                        "attempt": attempt,
                        "reason": reason,
                        "traceback": traceback.format_exc(limit=6),
                    }
                )

        if not accepted:
            break

    result["blueprint_coverage"] = _build_state_blueprint_coverage(result)
    return result


def _process_env_item(state_item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    state_bank = state_item.get("state_bank", []) if isinstance(state_item.get("state_bank"), list) else []
    if args.max_states_per_env > 0:
        state_bank = state_bank[: args.max_states_per_env]

    env_progress = _new_env_generation_progress()
    valid_state_entries: list[dict[str, Any]] = []
    env_results: list[dict[str, Any]] = []
    blueprint_count = 0
    failure_count = 0

    for state_index, state_entry in enumerate(state_bank):
        if not isinstance(state_entry, dict) or not isinstance(state_entry.get("init_config"), dict):
            continue
        valid_state_entries.append(state_entry)
        remaining_env_slots_after_state = (len(state_bank) - state_index - 1) * args.tasks_per_state
        result = _process_state_entry(state_item, state_entry, env_progress, remaining_env_slots_after_state, args)
        blueprint_count += len(result.get("task_blueprints", []))
        failure_count += len(result.get("task_generation_failures", []))
        env_results.append(result)

    if args.min_blueprints_per_env > 0 and blueprint_count < args.min_blueprints_per_env and env_results:
        while blueprint_count < args.min_blueprints_per_env:
            progress_made = False
            for state_index, state_entry in enumerate(valid_state_entries):
                if blueprint_count >= args.min_blueprints_per_env:
                    break
                existing_result = env_results[state_index]
                existing_blueprint_count = len(existing_result.get("task_blueprints", []))
                existing_failure_count = len(existing_result.get("task_generation_failures", []))
                updated_result = _process_state_entry(
                    state_item,
                    state_entry,
                    env_progress,
                    max(0, args.min_blueprints_per_env - int(env_progress.get("accepted_blueprint_count", 0)) - 1),
                    args,
                    target_blueprint_count=existing_blueprint_count + 1,
                    existing_result=existing_result,
                )
                env_results[state_index] = updated_result
                new_blueprints = len(updated_result.get("task_blueprints", [])) - existing_blueprint_count
                new_failures = len(updated_result.get("task_generation_failures", [])) - existing_failure_count
                if new_blueprints > 0:
                    blueprint_count += new_blueprints
                    progress_made = True
                if new_failures > 0:
                    failure_count += new_failures
            if not progress_made:
                break

    env_id = state_item.get("env_id", "")
    return {
        "env_id": env_id,
        "output_items": env_results,
        "env_coverage_summary": _build_env_coverage_summary(env_id, env_results, args),
        "blueprint_count": blueprint_count,
        "failure_count": failure_count,
    }


def main() -> None:
    args = parse_args()
    payload = read_json_file(args.input_file)
    items = _state_bank_items(payload)
    if args.max_envs > 0:
        items = items[: args.max_envs]

    output_items: list[dict[str, Any]] = []
    blueprint_count = 0
    failure_count = 0
    env_results_by_id: dict[str, dict[str, Any]] = {}
    completed_env_count = 0
    start_time = time.monotonic()

    print(
        "[Stage2 progress] "
        f"starting env_count={len(items)} max_workers={max(1, args.max_workers)} "
        f"tasks_per_state={args.tasks_per_state} max_states_per_env={args.max_states_per_env} "
        f"min_blueprints_per_env={args.min_blueprints_per_env} model={args.model}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {executor.submit(_process_env_item, state_item, args): state_item for state_item in items}
        pending_futures = set(futures)
        progress_interval = max(0, int(args.progress_interval or 0))
        while pending_futures:
            done_futures, pending_futures = wait(
                pending_futures,
                timeout=progress_interval if progress_interval > 0 else None,
                return_when=FIRST_COMPLETED,
            )
            if not done_futures:
                _print_stage2_progress(
                    completed_count=completed_env_count,
                    total_count=len(items),
                    start_time=start_time,
                    blueprint_count=blueprint_count,
                    failure_count=failure_count,
                )
                continue

            for future in done_futures:
                state_item = futures[future]
                env_id = str(state_item.get("env_id", ""))
                env_result = future.result()
                env_results_by_id[env_id] = env_result
                completed_env_count += 1
                latest_blueprint_count = int(env_result.get("blueprint_count", 0)) if isinstance(env_result, dict) else 0
                latest_failure_count = int(env_result.get("failure_count", 0)) if isinstance(env_result, dict) else 0
                blueprint_count += latest_blueprint_count
                failure_count += latest_failure_count
                coverage_summary = env_result.get("env_coverage_summary", {}) if isinstance(env_result, dict) else {}
                _print_stage2_progress(
                    completed_count=completed_env_count,
                    total_count=len(items),
                    start_time=start_time,
                    blueprint_count=blueprint_count,
                    failure_count=failure_count,
                    latest_env_id=env_id,
                    latest_blueprint_count=latest_blueprint_count,
                    latest_failure_count=latest_failure_count,
                    latest_coverage_passed=bool(coverage_summary.get("coverage_passed", False)) if isinstance(coverage_summary, dict) else None,
                )

    env_coverage_summary: list[dict[str, Any]] = []
    for state_item in items:
        env_id = str(state_item.get("env_id", ""))
        env_result = env_results_by_id.get(env_id)
        if not isinstance(env_result, dict):
            continue
        output_items.extend(env_result.get("output_items", []))
        if isinstance(env_result.get("env_coverage_summary"), dict):
            env_coverage_summary.append(env_result["env_coverage_summary"])

    env_coverage_failure_count = sum(1 for summary in env_coverage_summary if not summary.get("coverage_passed", False))
    env_coverage_pass_count = sum(1 for summary in env_coverage_summary if summary.get("coverage_passed", False))

    output_payload = {
        "metadata": {
            "generated_at": _now_iso(),
            "source_file": str(Path(args.input_file).resolve()),
            "model": args.model,
            "temperature": args.temperature,
            "tasks_per_state": args.tasks_per_state,
            "max_attempts_per_task": args.max_attempts_per_task,
            "max_workers": args.max_workers,
            "env_max_steps": args.env_max_steps,
            "min_blueprints_per_env": args.min_blueprints_per_env,
            "required_dialogue_modes": _parse_required_dialogue_modes(args.required_dialogue_modes),
            "min_difficulty_buckets": args.min_difficulty_buckets,
            "env_count": len(items),
            "state_item_count": len(output_items),
            "blueprint_count": blueprint_count,
            "generation_failure_count": failure_count,
            "env_coverage_pass_count": env_coverage_pass_count,
            "env_coverage_failure_count": env_coverage_failure_count,
        },
        "items": output_items,
        "env_coverage_summary": env_coverage_summary,
    }
    save_json_file(args.output_file, output_payload)

    summary = output_payload["metadata"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved Stage2 task blueprints to {args.output_file}")


if __name__ == "__main__":
    main()