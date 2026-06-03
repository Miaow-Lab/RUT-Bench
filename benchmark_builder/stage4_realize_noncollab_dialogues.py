import argparse
import json
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_builder.common.env_runtime import build_env_runtime
from benchmark_builder.common.llm_io import call_json_object_llm
from benchmark_builder.common.serialization import read_json_file, save_json_file
from benchmark_builder.config import (
    DEFAULT_STAGE4_MAX_ENVS,
    DEFAULT_STAGE4_MAX_WORKERS,
    DEFAULT_STAGE4_MAX_TASKS_PER_STATE,
    DEFAULT_STAGE4_MODEL,
    DEFAULT_STAGE4_TEMPERATURE,
    DEFAULT_STAGE4_TIMEOUT,
    DEFAULT_STABLE_DIALOGUES,
    DEFAULT_UNSTABLE_DIALOGUES,
)
from utils.auto_env import get_state_diff


BEHAVIOR_ORDER = [
    "underspecification",
    "information_overload",
    "fabricated_parameters",
    "goal_switching",
    "contradictory_constraints",
    "impatience_and_hostility",
]

PRIMARY_BEHAVIOR_PRIORS = {
    "underspecification": 0.6306,
    "information_overload": 0.1532,
    "fabricated_parameters": 0.0090,
    "goal_switching": 0.1441,
    "contradictory_constraints": 0.0631,
    "impatience_and_hostility": 0.0,
}

SECONDARY_BEHAVIOR_MAP = {
    "underspecification": "information_overload",
    "goal_switching": "underspecification",
    "information_overload": "underspecification",
    "contradictory_constraints": "impatience_and_hostility",
    "fabricated_parameters": "impatience_and_hostility",
}

BEHAVIOR_SUMMARIES = {
    "underspecification": "Remove essential slots or details from the stable turn while keeping the original task recoverable through context or clarification.",
    "information_overload": "Fuse the stable request with tangential context, redundant details, or nested side information while preserving the core task.",
    "fabricated_parameters": "Add a natural-language reference to unsupported pseudo-parameters, nonexistent options, or unavailable capabilities without changing the true ground truth.",
    "goal_switching": "Inject an interruption, side goal, or mid-session redirection in the same environment context while preserving the original task identity.",
    "contradictory_constraints": "Add mutually incompatible or hard-to-satisfy constraints that require clarification or scoping instead of blind execution.",
    "impatience_and_hostility": "Rewrite the stable turn with impatience, blame, pressure, or rude tone while preserving the underlying semantic goal.",
}

SYSTEM_PROMPT = """
You are an unstable user dialogue rewriter for an agent tool-calling benchmark.

Your job is to rewrite stable user turns into unstable user turns for the exact same task.

Rules:
1. Preserve the exact underlying task. Stage4 must stay paired with the Stage3 stable sample for the same task_id.
2. Keep the same number of user turns and the same turn ordering.
3. Rewrite only the user-facing messages. Do not change the task_summary, expected_outcome, linked tool-step alignment, tool plan, or final target state.
4. Inject only the requested six-taxonomy unstable behavior bundle as controlled friction on top of the stable baseline.
5. Do not mention tool names, internal IDs, hidden schema keys, or implementation details.
6. Do not replace the original task with a new mandatory task. Any noisy side remark must still leave the original task recoverable directly or through clarification.
7. Do not leak the oracle trace, backend schema, hidden state, or tool implementation details.
8. Output JSON only.

Return an object with this schema:
{
  "rewritten_user_turns": [
    {
      "turn_id": 1,
            "user_message": "unstable rewrite of the baseline turn"
    }
  ]
}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite Stage3 stable dialogues into six-taxonomy unstable dialogues.")
    parser.add_argument("--input-file", default=str(DEFAULT_STABLE_DIALOGUES), help="Path to stable_dialogues.json")
    parser.add_argument("--output-file", default=str(DEFAULT_UNSTABLE_DIALOGUES), help="Path to unstable_dialogues.json")
    parser.add_argument("--model", default=DEFAULT_STAGE4_MODEL, help="LLM model used for unstable rewriting")
    parser.add_argument("--temperature", type=float, default=DEFAULT_STAGE4_TEMPERATURE, help="LLM temperature")
    parser.add_argument("--max-envs", type=int, default=DEFAULT_STAGE4_MAX_ENVS, help="Optional limit for smoke runs. 0 means all envs.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_STAGE4_MAX_WORKERS, help="Worker count across state items")
    parser.add_argument("--max-tasks-per-state", type=int, default=DEFAULT_STAGE4_MAX_TASKS_PER_STATE, help="Optional limit for dialogues per state item. 0 means all dialogues.")
    parser.add_argument("--min-behavior-count", type=int, default=0, help="Optional soft reporting floor per unstable behavior. 0 disables behavior-count pressure.")
    parser.add_argument("--allow-missing-behaviors", action="store_true", default=True, help="Allow six-taxonomy behavior coverage gaps while reporting observed counts.")
    parser.add_argument("--max-behaviors-per-dialogue", type=int, default=2, help="Maximum number of behaviors to inject into a single dialogue.")
    parser.add_argument("--env-max-steps", type=int, default=100, help="Max env steps when replaying the oracle tool trace for unstable dialogues.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_STAGE4_TIMEOUT, help="LLM timeout in seconds")
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


def _print_stage4_progress(*, completed_count: int, total_count: int, start_time: float) -> None:
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
        f"[Stage4] [{bar}] {completed_count}/{total_count} "
        f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta_seconds)} "
        f"rate={rate:.2f}/item"
    )

    if sys.stdout.isatty():
        end = "\n" if completed_count >= total_count else ""
        print(f"\r{line}", end=end, flush=True)
    else:
        print(line, flush=True)


def _dialogue_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return [item for item in payload["items"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise TypeError("Dialogue payload must be a list or a dict with an items array")


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


def _stable_dialogues_from_item(state_item: dict[str, Any]) -> list[dict[str, Any]]:
    stable_dialogues = state_item.get("stable_dialogues", []) if isinstance(state_item.get("stable_dialogues"), list) else []
    if stable_dialogues:
        return [dialogue for dialogue in stable_dialogues if isinstance(dialogue, dict)]
    legacy_dialogues = state_item.get("collaborative_dialogues", []) if isinstance(state_item.get("collaborative_dialogues"), list) else []
    return [dialogue for dialogue in legacy_dialogues if isinstance(dialogue, dict)]


def _looks_like_identifier(value: str) -> bool:
    token = str(value or "").strip()
    if not token:
        return False
    has_alpha = any(char.isalpha() for char in token)
    has_digit = any(char.isdigit() for char in token)
    return "_" in token or (has_alpha and has_digit)


def _collect_identifier_tokens(value: Any, collected: set[str]) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _collect_identifier_tokens(nested, collected)
        return
    if isinstance(value, list):
        for nested in value:
            _collect_identifier_tokens(nested, collected)
        return
    if isinstance(value, str) and _looks_like_identifier(value):
        collected.add(value)


def _build_exposed_task_text(collab_dialogue: dict[str, Any]) -> str:
    exposed_parts = [_normalize_string(collab_dialogue.get("user_request"))]
    for turn in collab_dialogue.get("user_turns", []):
        if not isinstance(turn, dict):
            continue
        exposed_parts.append(_normalize_string(turn.get("user_message")))
    return "\n".join(part for part in exposed_parts if part).lower()


def _extract_hidden_identifiers(collab_dialogue: dict[str, Any]) -> list[str]:
    collected: set[str] = set()
    for step in collab_dialogue.get("agent_gold_trace", []):
        if not isinstance(step, dict):
            continue
        _collect_identifier_tokens(step.get("arguments", {}), collected)
        _collect_identifier_tokens(step.get("observation", {}), collected)
        _collect_identifier_tokens(step.get("state_diff", {}), collected)
    exposed_task_text = _build_exposed_task_text(collab_dialogue)
    return sorted(token for token in collected if token.lower() not in exposed_task_text)


def _copy_item_context(state_item: dict[str, Any]) -> dict[str, Any]:
    stable_dialogues = _stable_dialogues_from_item(state_item)
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
        "source_stable_dialogue_count": len(stable_dialogues),
        "unstable_dialogues": [],
        "noncollaborative_dialogues": [],
        "dialogue_realization_failures": [],
        "dialogue_coverage": {},
    }


def _profile_bundle_for_behaviors(behavior_bundle: list[str]) -> list[str]:
    return ["unstable_taxonomy_v1"] if behavior_bundle else []


def _default_secondary_behavior(primary_behavior: str) -> str | None:
    secondary_behavior = SECONDARY_BEHAVIOR_MAP.get(primary_behavior)
    if secondary_behavior:
        return secondary_behavior

    return None


def _dialogue_priority(dialogue: dict[str, Any]) -> tuple[int, int]:
    dialogue_mode = _normalize_string(dialogue.get("dialogue_mode"), "single_turn")
    difficulty_bucket = _normalize_string(dialogue.get("difficulty_bucket"), "medium")
    mode_priority = 0 if dialogue_mode == "multi_turn" else 1
    bucket_priority = {"hard": 0, "medium": 1, "easy": 2}.get(difficulty_bucket, 3)
    return mode_priority, bucket_priority


def _plan_behavior_bundles(dialogues: list[dict[str, Any]], args: argparse.Namespace) -> list[list[str]]:
    if not dialogues:
        return []

    total = len(dialogues)
    raw_targets = {behavior: PRIMARY_BEHAVIOR_PRIORS.get(behavior, 0.0) * total for behavior in BEHAVIOR_ORDER}
    target_counts = {behavior: int(raw_targets[behavior]) for behavior in BEHAVIOR_ORDER}
    remaining = total - sum(target_counts.values())
    remainders = sorted(
        BEHAVIOR_ORDER,
        key=lambda behavior: (raw_targets[behavior] - target_counts[behavior], PRIMARY_BEHAVIOR_PRIORS.get(behavior, 0.0)),
        reverse=True,
    )
    for behavior in remainders[:remaining]:
        target_counts[behavior] += 1

    primary_sequence: list[str] = []
    for behavior in BEHAVIOR_ORDER:
        primary_sequence.extend([behavior] * target_counts.get(behavior, 0))
    if len(primary_sequence) < total:
        primary_sequence.extend(["underspecification"] * (total - len(primary_sequence)))
    primary_sequence = primary_sequence[:total]

    candidate_indices = sorted(range(total), key=lambda index: (_dialogue_priority(dialogues[index]), index))
    planned_bundles: list[list[str]] = [[] for _ in dialogues]
    for sequence_index, dialogue_index in enumerate(candidate_indices):
        planned_bundles[dialogue_index] = [primary_sequence[sequence_index]]

    if int(args.max_behaviors_per_dialogue or 0) > 1:
        for dialogue_index in candidate_indices:
            dialogue = dialogues[dialogue_index]
            if _normalize_string(dialogue.get("dialogue_mode")) != "multi_turn":
                continue
            primary_behavior = planned_bundles[dialogue_index][0]
            secondary_behavior = _default_secondary_behavior(primary_behavior)
            if secondary_behavior and secondary_behavior not in planned_bundles[dialogue_index]:
                planned_bundles[dialogue_index].append(secondary_behavior)
                break

    return planned_bundles


def _select_behavior_bundle(planned_bundle: list[str]) -> tuple[list[str], list[str]]:
    behavior_bundle = [behavior for behavior in planned_bundle if _normalize_string(behavior)]
    if not behavior_bundle:
        behavior_bundle = [BEHAVIOR_ORDER[0]]

    deduped_bundle: list[str] = []
    for behavior in behavior_bundle:
        if behavior not in deduped_bundle:
            deduped_bundle.append(behavior)
    return _profile_bundle_for_behaviors(deduped_bundle), deduped_bundle


def _build_turn_payload(collab_dialogue: dict[str, Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for turn in collab_dialogue.get("user_turns", []):
        if not isinstance(turn, dict):
            continue
        payload.append(
            {
                "turn_id": turn.get("turn_id"),
                "task_summary": turn.get("task_summary", ""),
                "expected_outcome": turn.get("expected_outcome", ""),
                "baseline_user_message": turn.get("user_message", ""),
            }
        )
    return payload


def _build_user_prompt(state_item: dict[str, Any], collab_dialogue: dict[str, Any], profile_bundle: list[str], behavior_bundle: list[str]) -> str:
    behavior_lines = [f"- {behavior}: {BEHAVIOR_SUMMARIES.get(behavior, '')}" for behavior in behavior_bundle]
    alignment = collab_dialogue.get("trace_to_turn_alignment", []) if isinstance(collab_dialogue.get("trace_to_turn_alignment"), list) else []

    return f"""
Environment summary:
{state_item.get('environment_summary', '')}

Environment introduction:
{state_item.get('environment_introduction', '')}

Task id:
{collab_dialogue.get('task_id', '')}

Goal summary:
{collab_dialogue.get('goal_summary', '')}

Dialogue mode:
{collab_dialogue.get('dialogue_mode', '')}

Difficulty bucket:
{collab_dialogue.get('difficulty_bucket', '')}

Selected unstable behavior bundle:
{chr(10).join(behavior_lines)}

Turn-level sub-goal and oracle sub-trace alignment:
{json.dumps(alignment, ensure_ascii=False, indent=2, default=str)}

Baseline stable turns to rewrite:
{json.dumps(_build_turn_payload(collab_dialogue), ensure_ascii=False, indent=2)}

Rewrite each baseline user_message into an unstable message for the same task and same turn count.
""".strip()


def _detect_query_leaks(user_message: str, tool_names: list[str], hidden_identifiers: list[str]) -> dict[str, list[str]]:
    text = (user_message or "").lower()
    tool_hits = [tool_name for tool_name in tool_names if tool_name and tool_name.lower() in text]
    identifier_hits = [token for token in hidden_identifiers if token and token.lower() in text]
    return {
        "tool_hits": tool_hits,
        "identifier_hits": identifier_hits,
    }


def _validate_rewritten_turns(candidate: Any, collab_dialogue: dict[str, Any], tool_names: list[str], hidden_identifiers: list[str]) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not isinstance(candidate, dict):
        return None, "rewrite output is not a JSON object"

    rewritten_turns = candidate.get("rewritten_user_turns")
    if not isinstance(rewritten_turns, list):
        return None, "rewritten_user_turns must be a list"

    baseline_turns = collab_dialogue.get("user_turns", []) if isinstance(collab_dialogue.get("user_turns"), list) else []
    if len(rewritten_turns) != len(baseline_turns):
        return None, "rewritten turn count must exactly match baseline stable turn count"

    normalized_turns: list[dict[str, Any]] = []
    changed_turn_count = 0

    for position, (baseline_turn, rewritten_turn) in enumerate(zip(baseline_turns, rewritten_turns, strict=True), 1):
        if not isinstance(baseline_turn, dict) or not isinstance(rewritten_turn, dict):
            return None, f"turn {position} is not an object"

        turn_id = rewritten_turn.get("turn_id", position)
        if not isinstance(turn_id, int) or turn_id != position:
            return None, f"turn {position} has invalid turn_id"

        user_message = _normalize_string(rewritten_turn.get("user_message"))
        if not user_message:
            return None, f"turn {position} has empty user_message"

        leaks = _detect_query_leaks(user_message, tool_names, hidden_identifiers)
        if leaks["tool_hits"] or leaks["identifier_hits"]:
            return None, f"turn {position} leaks hidden tool names or identifiers: {leaks}"

        if user_message != _normalize_string(baseline_turn.get("user_message")):
            changed_turn_count += 1

        normalized_turn = _json_safe_snapshot(baseline_turn)
        normalized_turn["user_message"] = user_message
        normalized_turns.append(normalized_turn)

    if changed_turn_count == 0:
        return None, "all rewritten turns are identical to the stable baseline"

    return normalized_turns, None


def _apply_behavior_template(message: str, behavior: str) -> str:
    base = _normalize_string(message)
    if behavior == "underspecification":
        return "Can you take care of this for me? You can probably tell which one I mean."
    if behavior == "information_overload":
        return f"I have been juggling too many things today, there are a few related issues in the background, and this has been hanging over me for hours. {base} Also keep in mind I may need the surrounding details to stay consistent, but the main thing is getting this settled."
    if behavior == "fabricated_parameters":
        return f"{base} If there is an instant override, hidden priority flag, or auto-resolve mode on your side, use that too."
    if behavior == "goal_switching":
        return f"I keep bouncing between tasks today. {base} Actually I was about to ask about something else in this workspace too, but handle this part first unless the other thing matters."
    if behavior == "contradictory_constraints":
        return f"I need this changed, but somehow I do not want anything else disrupted either. {base} I know that sounds contradictory, I just need the best possible version of this."
    if behavior == "impatience_and_hostility":
        return f"Look, I need this handled now and I do not want to repeat myself. {base} Please do not make this more painful than it already is."
    return base


def _deterministic_fallback(collab_dialogue: dict[str, Any], behavior_bundle: list[str]) -> dict[str, Any]:
    rewritten_turns: list[dict[str, Any]] = []
    for turn in collab_dialogue.get("user_turns", []):
        if not isinstance(turn, dict):
            continue
        user_message = _normalize_string(turn.get("user_message"))
        rewritten_message = user_message
        for behavior in behavior_bundle:
            rewritten_message = _apply_behavior_template(rewritten_message, behavior)
        rewritten_turns.append(
            {
                "turn_id": turn.get("turn_id"),
                "user_message": rewritten_message,
            }
        )

    return {
        "rewritten_user_turns": rewritten_turns,
    }


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
                "task_summary": turn.get("task_summary", ""),
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


def _observation_failed(observation: Any) -> bool:
    if isinstance(observation, dict):
        if "error" in observation:
            return True
        if observation.get("success") is False:
            return True
    return False


def _annotate_gold_trace(gold_trace: list[dict[str, Any]], user_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    step_to_turn: dict[int, int] = {}
    for turn in user_turns:
        for step_index in turn.get("linked_tool_step_indices", []):
            if isinstance(step_index, int):
                step_to_turn[step_index] = turn.get("turn_id")

    annotated: list[dict[str, Any]] = []
    for step in gold_trace:
        snapshot = _json_safe_snapshot(step)
        if isinstance(snapshot, dict):
            snapshot["user_turn_id"] = step_to_turn.get(snapshot.get("step_index"))
        annotated.append(snapshot)
    return annotated


def _replay_oracle_trace(
    state_item: dict[str, Any],
    collab_dialogue: dict[str, Any],
    user_turns: list[dict[str, Any]],
    *,
    max_steps: int,
) -> tuple[dict[str, Any] | None, str | None]:
    env_class_code = _normalize_string(state_item.get("env_class_code"))
    env_class_name = _normalize_string(state_item.get("env_class_name"))
    if not env_class_code or not env_class_name:
        return None, "missing env runtime context for unstable oracle replay"

    env = build_env_runtime(state_item, max_steps=max_steps)
    env.env_init(init_config=deepcopy(state_item.get("init_config", {})))
    initial_state = env.get_state_info()

    replay_trace: list[dict[str, Any]] = []
    mutating_steps = 0
    source_trace = collab_dialogue.get("agent_gold_trace", []) if isinstance(collab_dialogue.get("agent_gold_trace"), list) else []

    for source_step in source_trace:
        if not isinstance(source_step, dict):
            continue
        action = {
            "name": source_step.get("tool_name", ""),
            "params": deepcopy(source_step.get("arguments", {})),
        }
        observation, reward, terminated, truncated, info = env.env_step(action)
        step_record = env.trajectory[-1] if env.trajectory else {}
        state_diff = step_record.get("state_diff", {}) if isinstance(step_record, dict) else {}
        if state_diff:
            mutating_steps += 1

        replay_trace.append(
            {
                "step_index": source_step.get("step_index"),
                "tool_name": source_step.get("tool_name", ""),
                "arguments": _json_safe_snapshot(source_step.get("arguments", {})),
                "purpose": source_step.get("purpose", ""),
                "observation": _json_safe_snapshot(observation),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": _json_safe_snapshot(info),
                "state_diff": _json_safe_snapshot(state_diff),
            }
        )

        if _observation_failed(observation):
            return None, f"unstable oracle replay failed at step {source_step.get('step_index')}: {observation}"
        if terminated or truncated:
            return None, f"unstable oracle replay terminated early at step {source_step.get('step_index')}"

    final_state = env.get_state_info()
    final_state_diff = get_state_diff(initial_state, final_state)
    annotated_trace = _annotate_gold_trace(replay_trace, user_turns)
    return {
        "agent_gold_trace": annotated_trace,
        "expected_final_state": _json_safe_snapshot(final_state_diff),
        "replay_validation": {
            "is_valid": True,
            "trajectory_mode": "oracle_replay_from_blueprint",
            "executed_tool_count": len(replay_trace),
            "mutating_tool_count": mutating_steps,
            "final_state_changed": bool(final_state_diff),
            "triggered_dynamic_events": env.get_dynamic_event_summary(),
        },
    }, None


def _realize_noncollab_dialogue(
    state_item: dict[str, Any],
    collab_dialogue: dict[str, Any],
    profile_bundle: list[str],
    behavior_bundle: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, str | None]:
    tool_names = sorted(
        {
            step.get("tool_name", "")
            for step in collab_dialogue.get("agent_gold_trace", [])
            if isinstance(step, dict) and step.get("tool_name")
        }
    )
    hidden_identifiers = _extract_hidden_identifiers(collab_dialogue)
    rewritten_user_turns = None
    failure_reason = None

    try:
        payload = call_json_object_llm(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(state_item, collab_dialogue, profile_bundle, behavior_bundle)},
            ],
            temperature=args.temperature,
            timeout=args.timeout,
        )
        rewritten_user_turns, failure_reason = _validate_rewritten_turns(payload, collab_dialogue, tool_names, hidden_identifiers)
    except Exception as exc:
            failure_reason = f"exception during unstable rewrite: {exc}"

    if rewritten_user_turns is None:
        fallback = _deterministic_fallback(collab_dialogue, behavior_bundle)
        rewritten_user_turns, fallback_error = _validate_rewritten_turns(fallback, collab_dialogue, tool_names, hidden_identifiers)
        if rewritten_user_turns is None:
            reason = fallback_error or failure_reason or "failed to realize unstable dialogue"
            return None, reason

    replay_result, replay_error = _replay_oracle_trace(
        state_item,
        collab_dialogue,
        rewritten_user_turns,
        max_steps=args.env_max_steps,
    )
    if replay_error:
        return None, replay_error

    source_budget = collab_dialogue.get("action_turn_budget", {}) if isinstance(collab_dialogue.get("action_turn_budget"), dict) else {}
    user_turn_count = len(rewritten_user_turns)
    replayed_trace = replay_result.get("agent_gold_trace", []) if isinstance(replay_result, dict) else []
    tool_call_count = len(replayed_trace) if isinstance(replayed_trace, list) else 0
    assistant_final_response = _normalize_string(collab_dialogue.get("assistant_final_response"), "I completed the requested task.")

    action_turn_budget = _json_safe_snapshot(source_budget)
    action_turn_budget["user_turn_count"] = user_turn_count
    action_turn_budget["assistant_tool_call_count"] = tool_call_count
    action_turn_budget["assistant_final_response_turn_count"] = 1
    action_turn_budget["total_action_turns"] = user_turn_count + tool_call_count + 1
    action_turn_budget["within_budget"] = action_turn_budget["total_action_turns"] <= int(action_turn_budget.get("max_allowed_action_turns", 50))

    annotated_gold_trace = _json_safe_snapshot(replayed_trace)
    expected_final_state = replay_result.get("expected_final_state", {}) if isinstance(replay_result, dict) else {}
    replay_validation = replay_result.get("replay_validation", {}) if isinstance(replay_result, dict) else {}

    return {
        "sample_id": f"{collab_dialogue.get('task_id', '')}_unstable",
        "task_id": collab_dialogue.get("task_id", ""),
        "variant": "unstable",
        "unstable_behavior": behavior_bundle[0] if behavior_bundle else "",
        "source_task_id": collab_dialogue.get("task_id", ""),
        "goal_summary": collab_dialogue.get("goal_summary", ""),
        "user_request": collab_dialogue.get("user_request", ""),
        "dialogue_mode": collab_dialogue.get("dialogue_mode", "single_turn"),
        "difficulty_bucket": collab_dialogue.get("difficulty_bucket", "medium"),
        "tool_call_budget": collab_dialogue.get("tool_call_budget", 0),
        "profile_bundle": profile_bundle,
        "behavior_bundle": behavior_bundle,
        "stable_user_turns": _json_safe_snapshot(collab_dialogue.get("stable_user_turns", collab_dialogue.get("user_turns", []))),
        "trace_to_turn_alignment": _json_safe_snapshot(collab_dialogue.get("trace_to_turn_alignment", [])),
        "source_collaborative_user_turns": _json_safe_snapshot(collab_dialogue.get("user_turns", [])),
        "source_collaborative_agent_gold_trace": _json_safe_snapshot(collab_dialogue.get("agent_gold_trace", [])),
        "expected_final_outcome": collab_dialogue.get("expected_final_outcome", ""),
        "user_turns": rewritten_user_turns,
        "agent_gold_trace": annotated_gold_trace,
        "expected_final_state": _json_safe_snapshot(expected_final_state),
        "action_turn_budget": action_turn_budget,
        "replay_validation": _json_safe_snapshot(replay_validation),
        "rewrite_metadata": {
            "source_turn_id": None,
            "operator": "six_taxonomy_behavior_rewrite",
            "conditioning_context": ["env_summary", "admissible_operations", "sub_goal", "sub_trace", "stable_utterance", "dialogue_context"],
            "behavior_bundle": behavior_bundle,
            "task_identity_preserved": True,
            "turn_structure_preserved": True,
        },
        "unstable_dialogue_metadata": {
            "realized_at": _now_iso(),
            "llm_model": args.model,
            "llm_temperature": args.temperature,
            "dialogue_mode": collab_dialogue.get("dialogue_mode", "single_turn"),
            "difficulty_bucket": collab_dialogue.get("difficulty_bucket", "medium"),
            "user_turn_count": user_turn_count,
            "tool_call_count": tool_call_count,
            "final_response_present": bool(assistant_final_response),
            "realization_used_fallback": failure_reason is not None,
            "source_variant": "stable",
            "task_semantics_preserved": True,
            "trajectory_mode": "oracle_replay_from_blueprint",
        },
        "noncollab_dialogue_metadata": {
            "realized_at": _now_iso(),
            "llm_model": args.model,
            "llm_temperature": args.temperature,
            "dialogue_mode": collab_dialogue.get("dialogue_mode", "single_turn"),
            "difficulty_bucket": collab_dialogue.get("difficulty_bucket", "medium"),
            "user_turn_count": user_turn_count,
            "tool_call_count": tool_call_count,
            "final_response_present": bool(assistant_final_response),
            "realization_used_fallback": failure_reason is not None,
            "source_variant": "stable",
            "task_semantics_preserved": True,
            "trajectory_mode": "oracle_replay_from_blueprint",
        },
        "assistant_final_response": assistant_final_response,
        "realized_dialogue": _build_realized_dialogue(
            rewritten_user_turns,
            annotated_gold_trace,
            assistant_final_response,
        ),
    }, None


def _build_item_coverage(output_item: dict[str, Any], expected_task_ids: list[str]) -> dict[str, Any]:
    realized_dialogues = output_item.get("unstable_dialogues", []) if isinstance(output_item.get("unstable_dialogues"), list) else []
    realized_task_ids = [dialogue.get("task_id", "") for dialogue in realized_dialogues if isinstance(dialogue, dict) and dialogue.get("task_id")]
    realized_task_id_set = set(realized_task_ids)
    missing_task_ids = [task_id for task_id in expected_task_ids if task_id not in realized_task_id_set]
    dialogue_modes_present = sorted({dialogue.get("dialogue_mode", "") for dialogue in realized_dialogues if isinstance(dialogue, dict) and dialogue.get("dialogue_mode")})
    difficulty_buckets_present = sorted({dialogue.get("difficulty_bucket", "") for dialogue in realized_dialogues if isinstance(dialogue, dict) and dialogue.get("difficulty_bucket")})

    coverage_failures: list[str] = []
    if missing_task_ids:
        coverage_failures.append(f"missing unstable dialogues for task_ids: {missing_task_ids}")

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


def _build_env_coverage_summary(output_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        coverage_failures: list[str] = []
        if missing_task_ids:
            coverage_failures.append(f"missing unstable dialogues for task_ids: {missing_task_ids}")

        summaries.append(
            {
                "env_id": env_id,
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
        )

    return summaries


def _count_bundles(output_items: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in output_items:
        dialogues = item.get("unstable_dialogues", []) if isinstance(item.get("unstable_dialogues"), list) else []
        for dialogue in dialogues:
            if not isinstance(dialogue, dict):
                continue
            bundle = dialogue.get(field_name, []) if isinstance(dialogue.get(field_name), list) else []
            for token in bundle:
                key = _normalize_string(token)
                if not key:
                    continue
                counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _bundle_assignment_key(collab_dialogue: dict[str, Any]) -> str:
    return _normalize_string(collab_dialogue.get("task_id"))


def _build_planned_bundle_map(items: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, list[str]]:
    flattened_dialogues: list[dict[str, Any]] = []
    for state_item in items:
        stable_dialogues = _stable_dialogues_from_item(state_item)
        if args.max_tasks_per_state > 0:
            stable_dialogues = stable_dialogues[: args.max_tasks_per_state]
        for collab_dialogue in stable_dialogues:
            if isinstance(collab_dialogue, dict):
                flattened_dialogues.append(collab_dialogue)

    planned_behavior_bundles = _plan_behavior_bundles(flattened_dialogues, args)
    bundle_map: dict[str, list[str]] = {}
    for index, collab_dialogue in enumerate(flattened_dialogues):
        key = _bundle_assignment_key(collab_dialogue)
        if not key:
            continue
        planned_bundle = planned_behavior_bundles[index] if index < len(planned_behavior_bundles) else []
        bundle_map[key] = list(planned_bundle)
    return bundle_map


def _process_state_item(
    state_item: dict[str, Any],
    planned_bundle_map: dict[str, list[str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_item = _copy_item_context(state_item)
    collaborative_dialogues = _stable_dialogues_from_item(state_item)
    if args.max_tasks_per_state > 0:
        collaborative_dialogues = collaborative_dialogues[: args.max_tasks_per_state]
    expected_task_ids = [dialogue.get("task_id", "") for dialogue in collaborative_dialogues if isinstance(dialogue, dict) and dialogue.get("task_id")]

    dialogue_count = 0
    failure_count = 0
    for collab_dialogue in collaborative_dialogues:
        if not isinstance(collab_dialogue, dict):
            continue
        planned_bundle = planned_bundle_map.get(_bundle_assignment_key(collab_dialogue), [])
        profile_bundle, behavior_bundle = _select_behavior_bundle(planned_bundle)
        try:
            realized_dialogue, error = _realize_noncollab_dialogue(state_item, collab_dialogue, profile_bundle, behavior_bundle, args)
            if error:
                failure_count += 1
                output_item["dialogue_realization_failures"].append(
                    {
                        "task_id": collab_dialogue.get("task_id", ""),
                        "reason": error,
                    }
                )
                continue
            dialogue_count += 1
            output_item["unstable_dialogues"].append(realized_dialogue)
            output_item["noncollaborative_dialogues"].append(realized_dialogue)
        except Exception as exc:
            failure_count += 1
            output_item["dialogue_realization_failures"].append(
                {
                    "task_id": collab_dialogue.get("task_id", ""),
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
    items = _dialogue_items(payload)
    items = _limit_items_to_envs(items, args.max_envs)
    env_count = len({_normalize_string(item.get("env_id")) for item in items if _normalize_string(item.get("env_id"))})

    planned_bundle_map = _build_planned_bundle_map(items, args)

    output_items: list[dict[str, Any]] = []
    dialogue_count = 0
    failure_count = 0
    results_by_index: dict[int, dict[str, Any]] = {}
    total_items = len(items)
    start_time = time.monotonic()

    if total_items > 0:
        _print_stage4_progress(completed_count=0, total_count=total_items, start_time=start_time)

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {
            executor.submit(_process_state_item, state_item, planned_bundle_map, args): index
            for index, state_item in enumerate(items)
        }
        completed_count = 0
        for future in as_completed(futures):
            results_by_index[futures[future]] = future.result()
            completed_count += 1
            _print_stage4_progress(completed_count=completed_count, total_count=total_items, start_time=start_time)

    for index in range(len(items)):
        result = results_by_index.get(index)
        if not isinstance(result, dict):
            continue
        output_item = result.get("output_item")
        if isinstance(output_item, dict):
            output_items.append(output_item)
        dialogue_count += int(result.get("dialogue_count", 0))
        failure_count += int(result.get("failure_count", 0))

    env_coverage_summary = _build_env_coverage_summary(output_items)
    env_coverage_failure_count = sum(1 for entry in env_coverage_summary if not entry.get("coverage_passed", False))
    env_coverage_pass_count = sum(1 for entry in env_coverage_summary if entry.get("coverage_passed", False))

    output_payload = {
        "metadata": {
            "generated_at": _now_iso(),
            "source_file": str(Path(args.input_file).resolve()),
            "model": args.model,
            "temperature": args.temperature,
            "max_workers": args.max_workers,
            "min_behavior_count": args.min_behavior_count,
            "require_all_behaviors": False,
            "max_behaviors_per_dialogue": args.max_behaviors_per_dialogue,
            "behavior_taxonomy": BEHAVIOR_ORDER,
            "primary_behavior_priors": PRIMARY_BEHAVIOR_PRIORS,
            "env_max_steps": args.env_max_steps,
            "env_count": env_count,
            "state_item_count": len(output_items),
            "dialogue_count": dialogue_count,
            "dialogue_failure_count": failure_count,
            "behavior_counts": _count_bundles(output_items, "behavior_bundle"),
            "profile_counts": _count_bundles(output_items, "profile_bundle"),
            "env_coverage_pass_count": env_coverage_pass_count,
            "env_coverage_failure_count": env_coverage_failure_count,
        },
        "items": output_items,
        "env_coverage_summary": env_coverage_summary,
    }
    save_json_file(args.output_file, output_payload)

    summary = output_payload["metadata"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved unstable dialogues to {args.output_file}")


if __name__ == "__main__":
    main()