import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_builder.common.llm_io import call_json_object_llm
from benchmark_builder.common.serialization import read_json_file, save_json_file
from benchmark_builder.config import (
    DEFAULT_STAGE3_MAX_ENVS,
    DEFAULT_STAGE3_MAX_WORKERS,
    DEFAULT_STAGE3_MAX_TASKS_PER_STATE,
    DEFAULT_STAGE3_MIN_DIALOGUES_PER_ENV,
    DEFAULT_STAGE3_MIN_DIFFICULTY_BUCKETS,
    DEFAULT_STAGE3_MODEL,
    DEFAULT_STAGE3_REQUIRED_DIALOGUE_MODES,
    DEFAULT_STAGE3_TEMPERATURE,
    DEFAULT_STAGE3_TIMEOUT,
    DEFAULT_STABLE_DIALOGUES,
    DEFAULT_TASK_BLUEPRINTS,
    STAGE3_PROMPT_TRACE_CHAR_LIMIT,
)


SYSTEM_PROMPT = """
You are a stable user simulator for an agent tool-calling benchmark.

Your job is to realize a validated task blueprint into natural stable user dialogue turns.

Rules:
1. Preserve the exact task semantics and do not invent a different task.
2. Keep the user cooperative, realistic, concise, and grounded in the environment context.
3. Do not mention tool names, internal IDs, hidden schema keys, implementation details, or oracle information.
4. If dialogue_mode is single_turn, output exactly one user turn.
5. If dialogue_mode is multi_turn, output 2-3 user turns that stay in the same task context.
6. Assign every gold-trace tool step to exactly one user turn using contiguous step groups.
7. Cover all tool steps exactly once, in increasing order.
8. The final assistant response should be a short user-facing completion message consistent with the expected outcome.
9. Output JSON only.

Return an object with this schema:
{
  "user_turns": [
    {
      "turn_id": 1,
      "user_message": "natural collaborative user message",
      "task_summary": "short turn summary",
      "expected_outcome": "expected turn-level outcome",
      "linked_tool_step_indices": [1, 2]
    }
  ],
  "assistant_final_response": "short user-facing completion message"
}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realize stable user dialogues from validated task blueprints and gold traces.")
    parser.add_argument("--input-file", default=str(DEFAULT_TASK_BLUEPRINTS), help="Path to task_blueprints.json")
    parser.add_argument("--output-file", default=str(DEFAULT_STABLE_DIALOGUES), help="Path to stable_dialogues.json")
    parser.add_argument("--model", default=DEFAULT_STAGE3_MODEL, help="LLM model used for stable dialogue realization")
    parser.add_argument("--temperature", type=float, default=DEFAULT_STAGE3_TEMPERATURE, help="LLM temperature")
    parser.add_argument("--max-envs", type=int, default=DEFAULT_STAGE3_MAX_ENVS, help="Optional limit for smoke runs. 0 means all envs.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_STAGE3_MAX_WORKERS, help="Worker count across state items")
    parser.add_argument("--max-tasks-per-state", type=int, default=DEFAULT_STAGE3_MAX_TASKS_PER_STATE, help="Optional limit for task blueprints per state item. 0 means all tasks.")
    parser.add_argument("--min-dialogues-per-env", type=int, default=DEFAULT_STAGE3_MIN_DIALOGUES_PER_ENV, help="Optional env-level coverage floor for realized collaborative dialogues")
    parser.add_argument("--required-dialogue-modes", default=DEFAULT_STAGE3_REQUIRED_DIALOGUE_MODES, help="Comma-separated env-level required dialogue modes, e.g. single_turn,multi_turn")
    parser.add_argument("--min-difficulty-buckets", type=int, default=DEFAULT_STAGE3_MIN_DIFFICULTY_BUCKETS, help="Optional env-level minimum count of distinct difficulty buckets")
    parser.add_argument("--timeout", type=int, default=DEFAULT_STAGE3_TIMEOUT, help="LLM timeout in seconds")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    seconds_int = max(0, int(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds_part:02d}"
    return f"{minutes:02d}:{seconds_part:02d}"


def _print_stage3_progress(*, completed_count: int, total_count: int, start_time: float) -> None:
    if total_count <= 0:
        return

    elapsed = max(time.monotonic() - start_time, 0.0)
    rate = (completed_count / elapsed) if elapsed > 0 else 0.0
    eta_seconds: float | None
    if completed_count > 0 and rate > 0:
        eta_seconds = max(total_count - completed_count, 0) / rate
    else:
        eta_seconds = None

    bar_width = 28
    ratio = min(max((completed_count / total_count), 0.0), 1.0)
    filled = int(bar_width * ratio)
    bar = "#" * filled + "-" * (bar_width - filled)
    line = (
        f"[Stage3] [{bar}] {completed_count}/{total_count} "
        f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta_seconds)} "
        f"rate={rate:.2f}/item"
    )

    if sys.stdout.isatty():
        end = "\n" if completed_count >= total_count else ""
        print(f"\r{line}", end=end, flush=True)
    else:
        print(line, flush=True)


def _blueprint_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return [item for item in payload["items"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise TypeError("Blueprint payload must be a list or a dict with an items array")


def _limit_items_to_envs(items: list[dict[str, Any]], max_envs: int) -> list[dict[str, Any]]:
    if max_envs <= 0:
        return items

    kept_env_ids: set[str] = set()
    limited_items: list[dict[str, Any]] = []
    for item in items:
        env_id = _normalize_string(item.get("env_id"))
        if env_id and env_id not in kept_env_ids and len(kept_env_ids) >= max_envs:
            continue
        if env_id:
            kept_env_ids.add(env_id)
        limited_items.append(item)
    return limited_items


def _normalize_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _json_safe_snapshot(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _looks_like_identifier(value: str) -> bool:
    token = str(value or "").strip()
    if not token:
        return False
    return any(char in token for char in ("_", "-")) or any(char.isdigit() for char in token)


def _detect_query_leaks(user_message: str, tool_names: list[str], relevant_entities: list[Any]) -> dict[str, list[str]]:
    text = (user_message or "").lower()
    tool_hits = [tool_name for tool_name in tool_names if tool_name and tool_name.lower() in text]

    entity_hits: list[str] = []
    for entity in relevant_entities:
        token = str(entity).strip()
        if token and _looks_like_identifier(token) and token.lower() in text:
            entity_hits.append(token)

    return {
        "tool_hits": tool_hits,
        "entity_hits": entity_hits,
    }


def _compact_json(data: Any, max_chars: int) -> str:
    text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "\n...<truncated>..."
    return text


def _summarize_gold_trace(gold_trace: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for step in gold_trace:
        state_diff = step.get("state_diff", {})
        state_keys = sorted(state_diff.keys()) if isinstance(state_diff, dict) else []
        observation = _compact_json(step.get("observation", {}), 320).replace("\n", " ")
        lines.append(
            f"{step.get('step_index')}. {step.get('tool_name')} | purpose={step.get('purpose', '')} | state_diff_keys={state_keys or ['none']} | observation={observation}"
        )
    return "\n".join(lines)


def _build_user_prompt(state_item: dict[str, Any], blueprint: dict[str, Any]) -> str:
    return f"""
Environment summary:
{state_item.get('environment_summary', '')}

Environment introduction:
{state_item.get('environment_introduction', '')}

Task goal summary:
{blueprint.get('goal_summary', '')}

Primary user request:
{blueprint.get('user_request', '')}

Dialogue mode:
{blueprint.get('dialogue_mode', '')}

Difficulty bucket:
{blueprint.get('difficulty_bucket', '')}

Expected phases:
{_compact_json(blueprint.get('expected_phases', []), 1200)}

Expected final outcome:
{blueprint.get('expected_final_outcome', '')}

Turn-level task hints:
{_compact_json(blueprint.get('multi_turn_tasks', []), 2000)}

Validated gold trace summary:
{_summarize_gold_trace(blueprint.get('gold_trace', []))}

Design natural collaborative user turns that fit this exact validated execution path.
""".strip()


def _build_turn_step_groups(step_indices: list[int], requested_turn_count: int) -> list[list[int]] | None:
    if requested_turn_count <= 0:
        return None
    if not step_indices:
        return [[] for _ in range(requested_turn_count)]
    if len(step_indices) < requested_turn_count:
        empty_turn_count = requested_turn_count - len(step_indices)
        return ([[] for _ in range(empty_turn_count)] + [[step_index] for step_index in step_indices])[:requested_turn_count]

    base_size = len(step_indices) // requested_turn_count
    remainder = len(step_indices) % requested_turn_count
    groups: list[list[int]] = []
    start = 0
    for turn_idx in range(requested_turn_count):
        group_size = base_size + (1 if turn_idx < remainder else 0)
        end = start + group_size
        groups.append(step_indices[start:end])
        start = end
    return groups


def _select_multi_turn_count(blueprint: dict[str, Any], step_indices: list[int]) -> int | None:
    multi_turn_tasks = blueprint.get("multi_turn_tasks", []) if isinstance(blueprint.get("multi_turn_tasks"), list) else []
    hinted_turn_count = len(multi_turn_tasks) if multi_turn_tasks else 2
    if not step_indices:
        return max(2, min(3, hinted_turn_count))
    return max(2, min(3, hinted_turn_count))


def _deterministic_fallback(blueprint: dict[str, Any]) -> dict[str, Any]:
    gold_trace = blueprint.get("gold_trace", []) if isinstance(blueprint.get("gold_trace"), list) else []
    step_indices = [step.get("step_index") for step in gold_trace if isinstance(step, dict) and isinstance(step.get("step_index"), int)]

    if blueprint.get("dialogue_mode") == "multi_turn":
        multi_turn_tasks = blueprint.get("multi_turn_tasks", []) if isinstance(blueprint.get("multi_turn_tasks"), list) else []
        turn_count = _select_multi_turn_count(blueprint, step_indices)
        split_points = _build_turn_step_groups(step_indices, turn_count or 0)
        if turn_count is None or split_points is None:
            return {
                "user_turns": [],
                "assistant_final_response": "",
            }

        user_turns: list[dict[str, Any]] = []
        for turn_id in range(1, turn_count + 1):
            task_hint = multi_turn_tasks[turn_id - 1] if turn_id - 1 < len(multi_turn_tasks) and isinstance(multi_turn_tasks[turn_id - 1], dict) else {}
            task_summary = _normalize_string(task_hint.get("task_summary"), f"turn {turn_id}")
            expected_outcome = _normalize_string(task_hint.get("expected_outcome"), blueprint.get("expected_final_outcome", ""))
            if turn_id == 1:
                user_message = _normalize_string(blueprint.get("user_request"), task_summary)
            else:
                user_message = f"Thanks. Please continue by handling this next part as well: {task_summary}."
            user_turns.append(
                {
                    "turn_id": turn_id,
                    "user_message": user_message,
                    "task_summary": task_summary,
                    "expected_outcome": expected_outcome,
                    "linked_tool_step_indices": split_points[turn_id - 1],
                }
            )
    else:
        user_turns = [
            {
                "turn_id": 1,
                "user_message": _normalize_string(blueprint.get("user_request"), blueprint.get("goal_summary", "Please complete this task.")),
                "task_summary": _normalize_string(blueprint.get("goal_summary"), "task"),
                "expected_outcome": _normalize_string(blueprint.get("expected_final_outcome"), ""),
                "linked_tool_step_indices": step_indices,
            }
        ]

    return {
        "user_turns": user_turns,
        "assistant_final_response": _normalize_string(
            blueprint.get("expected_final_outcome"),
            "I completed the requested task.",
        ),
    }


def _validate_realized_dialogue(candidate: Any, blueprint: dict[str, Any], tool_names: list[str]) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(candidate, dict):
        return None, "realized dialogue is not a JSON object"

    user_turns = candidate.get("user_turns")
    if not isinstance(user_turns, list) or not user_turns:
        return None, "user_turns must be a non-empty list"

    dialogue_mode = blueprint.get("dialogue_mode")
    if dialogue_mode == "single_turn" and len(user_turns) != 1:
        return None, "single_turn blueprint must realize exactly one user turn"
    if dialogue_mode == "multi_turn" and not (2 <= len(user_turns) <= 3):
        return None, "multi_turn blueprint must realize 2-3 user turns"

    gold_trace = blueprint.get("gold_trace", []) if isinstance(blueprint.get("gold_trace"), list) else []
    expected_step_indices = [step.get("step_index") for step in gold_trace if isinstance(step, dict) and isinstance(step.get("step_index"), int)]
    collected_indices: list[int] = []
    normalized_turns: list[dict[str, Any]] = []
    relevant_entities = blueprint.get("relevant_entities", []) if isinstance(blueprint.get("relevant_entities"), list) else []

    for position, turn in enumerate(user_turns, 1):
        if not isinstance(turn, dict):
            return None, f"user turn {position} is not an object"
        turn_id = turn.get("turn_id", position)
        if not isinstance(turn_id, int) or turn_id != position:
            return None, f"user turn {position} has invalid turn_id"

        user_message = _normalize_string(turn.get("user_message"))
        task_summary = _normalize_string(turn.get("task_summary"), blueprint.get("goal_summary", ""))
        expected_outcome = _normalize_string(turn.get("expected_outcome"), blueprint.get("expected_final_outcome", ""))
        if not user_message:
            return None, f"user turn {position} has empty user_message"

        leaks = _detect_query_leaks(user_message, tool_names, relevant_entities)
        if leaks["tool_hits"] or leaks["entity_hits"]:
            return None, f"user turn {position} leaks hidden tool names or identifiers: {leaks}"

        linked_indices = turn.get("linked_tool_step_indices")
        if not isinstance(linked_indices, list):
            return None, f"user turn {position} linked_tool_step_indices must be a list"
        if any(not isinstance(index, int) for index in linked_indices):
            return None, f"user turn {position} has non-integer linked_tool_step_indices"
        if linked_indices and linked_indices != list(range(linked_indices[0], linked_indices[0] + len(linked_indices))):
            return None, f"user turn {position} must use contiguous tool step indices"

        collected_indices.extend(linked_indices)
        normalized_turns.append(
            {
                "turn_id": turn_id,
                "user_message": user_message,
                "task_summary": task_summary,
                "expected_outcome": expected_outcome,
                "linked_tool_step_indices": linked_indices,
            }
        )

    if collected_indices != expected_step_indices:
        return None, "linked tool steps must cover the gold trace exactly once and in order"

    assistant_final_response = _normalize_string(candidate.get("assistant_final_response"))
    if not assistant_final_response:
        return None, "assistant_final_response is required"

    return {
        "user_turns": normalized_turns,
        "assistant_final_response": assistant_final_response,
    }, None


def _annotate_gold_trace(gold_trace: list[dict[str, Any]], user_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    step_to_turn: dict[int, int] = {}
    for turn in user_turns:
        for step_index in turn.get("linked_tool_step_indices", []):
            step_to_turn[step_index] = turn["turn_id"]

    annotated: list[dict[str, Any]] = []
    for step in gold_trace:
        snapshot = _json_safe_snapshot(step)
        if isinstance(snapshot, dict):
            snapshot["user_turn_id"] = step_to_turn.get(snapshot.get("step_index"))
        annotated.append(snapshot)
    return annotated


def _build_realized_dialogue(user_turns: list[dict[str, Any]], annotated_gold_trace: list[dict[str, Any]], assistant_final_response: str) -> list[dict[str, Any]]:
    tool_steps_by_turn: dict[int, list[dict[str, Any]]] = {}
    for step in annotated_gold_trace:
        if not isinstance(step, dict):
            continue
        turn_id = step.get("user_turn_id")
        if isinstance(turn_id, int):
            tool_steps_by_turn.setdefault(turn_id, []).append(step)

    realized_dialogue: list[dict[str, Any]] = []
    for turn in user_turns:
        turn_id = turn["turn_id"]
        realized_dialogue.append(
            {
                "role": "user",
                "turn_id": turn_id,
                "content": turn["user_message"],
                "task_summary": turn["task_summary"],
            }
        )
        realized_dialogue.append(
            {
                "role": "assistant_tool_trace",
                "turn_id": turn_id,
                "tool_steps": tool_steps_by_turn.get(turn_id, []),
            }
        )

    realized_dialogue.append(
        {
            "role": "assistant",
            "turn_id": len(user_turns),
            "content": assistant_final_response,
        }
    )
    return realized_dialogue


def _realize_blueprint(state_item: dict[str, Any], blueprint: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, str | None]:
    tool_names = [tool.get("name", "") for tool in state_item.get("tools", []) if isinstance(tool, dict) and tool.get("name")]
    candidate = None
    failure_reason = None

    try:
        payload = call_json_object_llm(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(state_item, blueprint)},
            ],
            temperature=args.temperature,
            timeout=args.timeout,
        )
        candidate, failure_reason = _validate_realized_dialogue(payload, blueprint, tool_names)
    except Exception as exc:
            failure_reason = f"exception during stable dialogue realization: {exc}"

    if candidate is None:
        fallback = _deterministic_fallback(blueprint)
        candidate, fallback_error = _validate_realized_dialogue(fallback, blueprint, tool_names)
        if candidate is None:
            reason = fallback_error or failure_reason or "failed to realize collaborative dialogue"
            return None, reason

    annotated_gold_trace = _annotate_gold_trace(blueprint.get("gold_trace", []), candidate["user_turns"])
    trace_to_turn_alignment = [
        {
            "turn_id": turn["turn_id"],
            "sub_goal": turn.get("task_summary", ""),
            "sub_trace": [
                _json_safe_snapshot(step)
                for step in annotated_gold_trace
                if isinstance(step, dict) and step.get("user_turn_id") == turn["turn_id"]
            ],
        }
        for turn in candidate["user_turns"]
    ]
    user_turn_count = len(candidate["user_turns"])
    tool_call_count = len(annotated_gold_trace)
    total_action_turns = user_turn_count + tool_call_count + 1
    max_allowed = int(blueprint.get("agent_action_budget", 50))

    return {
        "task_id": blueprint.get("task_id", ""),
        "goal_summary": blueprint.get("goal_summary", ""),
        "user_request": blueprint.get("user_request", ""),
        "dialogue_mode": blueprint.get("dialogue_mode", "single_turn"),
        "difficulty_bucket": blueprint.get("difficulty_bucket", "medium"),
        "tool_call_budget": blueprint.get("tool_call_budget", 0),
        "expected_final_outcome": blueprint.get("expected_final_outcome", ""),
        "user_turns": candidate["user_turns"],
        "stable_user_turns": _json_safe_snapshot(candidate["user_turns"]),
        "trace_to_turn_alignment": trace_to_turn_alignment,
        "agent_gold_trace": annotated_gold_trace,
        "expected_final_state": _json_safe_snapshot(blueprint.get("final_state_diff", {})),
        "action_turn_budget": {
            "user_turn_count": user_turn_count,
            "assistant_tool_call_count": tool_call_count,
            "assistant_final_response_turn_count": 1,
            "total_action_turns": total_action_turns,
            "max_allowed_action_turns": max_allowed,
            "within_budget": total_action_turns <= max_allowed,
        },
        "stable_dialogue_metadata": {
            "realized_at": _now_iso(),
            "llm_model": args.model,
            "llm_temperature": args.temperature,
            "dialogue_mode": blueprint.get("dialogue_mode", "single_turn"),
            "difficulty_bucket": blueprint.get("difficulty_bucket", "medium"),
            "user_turn_count": user_turn_count,
            "tool_call_count": tool_call_count,
            "final_response_present": True,
            "realization_used_fallback": failure_reason is not None,
        },
        "collab_dialogue_metadata": {
            "realized_at": _now_iso(),
            "llm_model": args.model,
            "llm_temperature": args.temperature,
            "dialogue_mode": blueprint.get("dialogue_mode", "single_turn"),
            "difficulty_bucket": blueprint.get("difficulty_bucket", "medium"),
            "user_turn_count": user_turn_count,
            "tool_call_count": tool_call_count,
            "final_response_present": True,
            "realization_used_fallback": failure_reason is not None,
        },
        "assistant_final_response": candidate["assistant_final_response"],
        "realized_dialogue": _build_realized_dialogue(
            candidate["user_turns"],
            annotated_gold_trace,
            candidate["assistant_final_response"],
        ),
    }, None


def _copy_item_context(state_item: dict[str, Any]) -> dict[str, Any]:
    return {
        "env_id": state_item.get("env_id", ""),
        "config_id": state_item.get("config_id", ""),
        "environment_summary": state_item.get("environment_summary", ""),
        "environment_introduction": state_item.get("environment_introduction", ""),
        "env_class_name": state_item.get("env_class_name", ""),
        "env_class_code": state_item.get("env_class_code", ""),
        "constraints_rules_list": _json_safe_snapshot(state_item.get("constraints_rules_list", [])),
        "constraints_rules_text": state_item.get("constraints_rules_text", ""),
        "tools": _json_safe_snapshot(state_item.get("tools", [])),
        "tool_summary": _json_safe_snapshot(state_item.get("tool_summary", [])),
        "tool_summary_text": state_item.get("tool_summary_text", ""),
        "state_profile": state_item.get("state_profile", ""),
        "state_stats": _json_safe_snapshot(state_item.get("state_stats", {})),
        "init_config": _json_safe_snapshot(state_item.get("init_config", {})),
        "task_blueprint_count": 0,
        "stable_dialogues": [],
        "collaborative_dialogues": [],
        "dialogue_realization_failures": [],
        "dialogue_coverage": {},
    }


def _parse_required_dialogue_modes(raw_value: str) -> set[str]:
    modes = {part.strip() for part in str(raw_value or "").split(",") if part.strip()}
    return {mode for mode in modes if mode in {"single_turn", "multi_turn"}}


def _build_item_coverage(output_item: dict[str, Any], expected_task_ids: list[str]) -> dict[str, Any]:
    realized_dialogues = output_item.get("stable_dialogues", []) if isinstance(output_item.get("stable_dialogues"), list) else []
    realized_task_ids = [dialogue.get("task_id", "") for dialogue in realized_dialogues if isinstance(dialogue, dict) and dialogue.get("task_id")]
    realized_task_id_set = set(realized_task_ids)
    missing_task_ids = [task_id for task_id in expected_task_ids if task_id not in realized_task_id_set]
    dialogue_modes_present = sorted({dialogue.get("dialogue_mode", "") for dialogue in realized_dialogues if isinstance(dialogue, dict) and dialogue.get("dialogue_mode")})
    difficulty_buckets_present = sorted({dialogue.get("difficulty_bucket", "") for dialogue in realized_dialogues if isinstance(dialogue, dict) and dialogue.get("difficulty_bucket")})

    coverage_failures: list[str] = []
    if missing_task_ids:
        coverage_failures.append(f"missing realized dialogues for task_ids: {missing_task_ids}")

    return {
        "expected_dialogue_count": len(expected_task_ids),
        "realized_dialogue_count": len(realized_task_ids),
        "expected_task_ids": expected_task_ids,
        "realized_task_ids": realized_task_ids,
        "missing_task_ids": missing_task_ids,
        "dialogue_modes_present": dialogue_modes_present,
        "difficulty_buckets_present": difficulty_buckets_present,
        "coverage_failures": coverage_failures,
        "coverage_passed": not coverage_failures,
    }


def _build_env_coverage_summary(output_items: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    required_dialogue_modes = _parse_required_dialogue_modes(args.required_dialogue_modes)
    env_map: dict[str, dict[str, Any]] = {}

    for item in output_items:
        env_id = str(item.get("env_id", "")).strip()
        if not env_id:
            continue
        entry = env_map.setdefault(
            env_id,
            {
                "env_id": env_id,
                "expected_task_ids": [],
                "realized_task_ids": [],
                "dialogue_modes_present": set(),
                "difficulty_buckets_present": set(),
            },
        )
        coverage = item.get("dialogue_coverage", {}) if isinstance(item.get("dialogue_coverage"), dict) else {}
        entry["expected_task_ids"].extend(coverage.get("expected_task_ids", []))
        entry["realized_task_ids"].extend(coverage.get("realized_task_ids", []))
        entry["dialogue_modes_present"].update(coverage.get("dialogue_modes_present", []))
        entry["difficulty_buckets_present"].update(coverage.get("difficulty_buckets_present", []))

    summaries: list[dict[str, Any]] = []
    for env_id in sorted(env_map):
        entry = env_map[env_id]
        expected_task_ids = sorted({task_id for task_id in entry["expected_task_ids"] if task_id})
        realized_task_ids = sorted({task_id for task_id in entry["realized_task_ids"] if task_id})
        realized_task_id_set = set(realized_task_ids)
        missing_task_ids = [task_id for task_id in expected_task_ids if task_id not in realized_task_id_set]
        dialogue_modes_present = sorted(entry["dialogue_modes_present"])
        difficulty_buckets_present = sorted(entry["difficulty_buckets_present"])
        missing_dialogue_modes = sorted(required_dialogue_modes - set(dialogue_modes_present))

        coverage_failures: list[str] = []
        if missing_task_ids:
            coverage_failures.append(f"missing realized dialogues for task_ids: {missing_task_ids}")
        if args.min_dialogues_per_env > 0 and len(realized_task_ids) < args.min_dialogues_per_env:
            coverage_failures.append(
                f"realized_dialogue_count {len(realized_task_ids)} is below min_dialogues_per_env {args.min_dialogues_per_env}"
            )
        if missing_dialogue_modes:
            coverage_failures.append(f"missing required dialogue modes: {missing_dialogue_modes}")
        if args.min_difficulty_buckets > 0 and len(difficulty_buckets_present) < args.min_difficulty_buckets:
            coverage_failures.append(
                f"distinct difficulty bucket count {len(difficulty_buckets_present)} is below min_difficulty_buckets {args.min_difficulty_buckets}"
            )

        summaries.append(
            {
                "env_id": env_id,
                "expected_dialogue_count": len(expected_task_ids),
                "realized_dialogue_count": len(realized_task_ids),
                "expected_task_ids": expected_task_ids,
                "realized_task_ids": realized_task_ids,
                "missing_task_ids": missing_task_ids,
                "dialogue_modes_present": dialogue_modes_present,
                "missing_dialogue_modes": missing_dialogue_modes,
                "difficulty_buckets_present": difficulty_buckets_present,
                "min_required_dialogues": args.min_dialogues_per_env,
                "required_dialogue_modes": sorted(required_dialogue_modes),
                "min_required_difficulty_buckets": args.min_difficulty_buckets,
                "coverage_failures": coverage_failures,
                "coverage_passed": not coverage_failures,
            }
        )

    return summaries


def _process_state_item(state_item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    output_item = _copy_item_context(state_item)
    task_blueprints = state_item.get("task_blueprints", []) if isinstance(state_item.get("task_blueprints"), list) else []
    if args.max_tasks_per_state > 0:
        task_blueprints = task_blueprints[: args.max_tasks_per_state]
    output_item["task_blueprint_count"] = len(task_blueprints)
    expected_task_ids = [blueprint.get("task_id", "") for blueprint in task_blueprints if isinstance(blueprint, dict) and blueprint.get("task_id")]

    dialogue_count = 0
    failure_count = 0
    for blueprint in task_blueprints:
        if not isinstance(blueprint, dict):
            continue
        try:
            realized_dialogue, error = _realize_blueprint(state_item, blueprint, args)
            if error:
                failure_count += 1
                output_item["dialogue_realization_failures"].append(
                    {
                        "task_id": blueprint.get("task_id", ""),
                        "reason": error,
                    }
                )
                continue
            dialogue_count += 1
            output_item["stable_dialogues"].append(realized_dialogue)
            output_item["collaborative_dialogues"].append(realized_dialogue)
        except Exception as exc:
            failure_count += 1
            output_item["dialogue_realization_failures"].append(
                {
                    "task_id": blueprint.get("task_id", ""),
                    "reason": f"unexpected exception: {exc}",
                    "traceback": traceback.format_exc(limit=6),
                }
            )

    output_item["dialogue_coverage"] = _build_item_coverage(output_item, expected_task_ids)
    return {
        "output_item": output_item,
        "dialogue_count": dialogue_count,
        "failure_count": failure_count,
    }


def main() -> None:
    args = parse_args()
    payload = read_json_file(args.input_file)
    items = _blueprint_items(payload)
    items = _limit_items_to_envs(items, args.max_envs)
    env_count = len({_normalize_string(item.get("env_id")) for item in items if _normalize_string(item.get("env_id"))})

    output_items: list[dict[str, Any]] = []
    dialogue_count = 0
    failure_count = 0
    results_by_index: dict[int, dict[str, Any]] = {}
    total_items = len(items)
    start_time = time.monotonic()

    if total_items > 0:
        _print_stage3_progress(completed_count=0, total_count=total_items, start_time=start_time)

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {executor.submit(_process_state_item, state_item, args): index for index, state_item in enumerate(items)}
        completed_count = 0
        for future in as_completed(futures):
            results_by_index[futures[future]] = future.result()
            completed_count += 1
            _print_stage3_progress(completed_count=completed_count, total_count=total_items, start_time=start_time)

    for index in range(len(items)):
        result = results_by_index.get(index)
        if not isinstance(result, dict):
            continue
        output_item = result.get("output_item")
        if isinstance(output_item, dict):
            output_items.append(output_item)
        dialogue_count += int(result.get("dialogue_count", 0))
        failure_count += int(result.get("failure_count", 0))

    env_coverage_summary = _build_env_coverage_summary(output_items, args)
    env_coverage_failure_count = sum(1 for entry in env_coverage_summary if not entry.get("coverage_passed", False))
    env_coverage_pass_count = sum(1 for entry in env_coverage_summary if entry.get("coverage_passed", False))

    output_payload = {
        "metadata": {
            "generated_at": _now_iso(),
            "source_file": str(Path(args.input_file).resolve()),
            "model": args.model,
            "temperature": args.temperature,
            "max_workers": args.max_workers,
            "env_count": env_count,
            "state_item_count": len(output_items),
            "dialogue_count": dialogue_count,
            "dialogue_failure_count": failure_count,
            "min_dialogues_per_env": args.min_dialogues_per_env,
            "required_dialogue_modes": sorted(_parse_required_dialogue_modes(args.required_dialogue_modes)),
            "min_difficulty_buckets": args.min_difficulty_buckets,
            "env_coverage_pass_count": env_coverage_pass_count,
            "env_coverage_failure_count": env_coverage_failure_count,
        },
        "items": output_items,
        "env_coverage_summary": env_coverage_summary,
    }
    save_json_file(args.output_file, output_payload)

    summary = output_payload["metadata"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved stable dialogues to {args.output_file}")


if __name__ == "__main__":
    main()