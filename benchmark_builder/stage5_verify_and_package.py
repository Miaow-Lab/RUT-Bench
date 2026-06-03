import argparse
import json
import sys
from math import ceil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_builder.common.serialization import read_json_file, save_json_file, save_jsonl_file
from benchmark_builder.config import (
    DEFAULT_BENCHMARK_FULL,
    DEFAULT_BENCHMARK_PACKAGE_REPORT,
    DEFAULT_BENCHMARK_STABLE,
    DEFAULT_BENCHMARK_UNSTABLE,
    DEFAULT_STAGE5_FAIL_ON_QC_ERRORS,
    DEFAULT_STAGE5_MAX_DIFFICULTY_IMBALANCE_RATIO,
    DEFAULT_STAGE5_MAX_DIALOGUE_MODE_IMBALANCE_RATIO,
    DEFAULT_STAGE5_MAX_ENVS,
    DEFAULT_STAGE5_MIN_BEHAVIOR_COUNT,
    DEFAULT_STAGE5_MIN_DISTINCT_DIFFICULTY_BUCKETS,
    DEFAULT_STAGE5_MIN_SAMPLES_PER_ENV,
    DEFAULT_STAGE5_REQUIRED_DIALOGUE_MODES,
    DEFAULT_STAGE5_REQUIRE_ALL_BEHAVIORS,
    DEFAULT_STAGE5_TARGET_BASE_BLUEPRINTS_PER_ENV,
    DEFAULT_STABLE_DIALOGUES,
    DEFAULT_TASK_BLUEPRINTS,
    DEFAULT_UNSTABLE_DIALOGUES,
)


KNOWN_UNSTABLE_BEHAVIORS = [
    "underspecification",
    "information_overload",
    "fabricated_parameters",
    "goal_switching",
    "contradictory_constraints",
    "impatience_and_hostility",
]

UNSTABLE_PRIMARY_PRIORS = {
    "underspecification": 0.6306,
    "information_overload": 0.1532,
    "fabricated_parameters": 0.0090,
    "goal_switching": 0.1441,
    "contradictory_constraints": 0.0631,
    "impatience_and_hostility": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify paired Stage2-Stage4 artifacts and package stable/unstable benchmark JSONL datasets.")
    parser.add_argument("--blueprint-file", default=str(DEFAULT_TASK_BLUEPRINTS), help="Path to task_blueprints.json")
    parser.add_argument("--stable-file", "--collab-file", dest="stable_file", default=str(DEFAULT_STABLE_DIALOGUES), help="Path to stable_dialogues.json")
    parser.add_argument("--unstable-file", "--noncollab-file", dest="unstable_file", default=str(DEFAULT_UNSTABLE_DIALOGUES), help="Path to unstable_dialogues.json")
    parser.add_argument("--output-stable-file", "--output-collab-file", dest="output_stable_file", default=str(DEFAULT_BENCHMARK_STABLE), help="Path to benchmark_stable.jsonl")
    parser.add_argument("--output-unstable-file", "--output-noncollab-file", dest="output_unstable_file", default=str(DEFAULT_BENCHMARK_UNSTABLE), help="Path to benchmark_unstable.jsonl")
    parser.add_argument("--output-full-file", default=str(DEFAULT_BENCHMARK_FULL), help="Path to benchmark_full.jsonl")
    parser.add_argument("--report-file", default=str(DEFAULT_BENCHMARK_PACKAGE_REPORT), help="Path to Stage5 packaging report JSON")
    parser.add_argument("--max-envs", type=int, default=DEFAULT_STAGE5_MAX_ENVS, help="Optional env limit for smoke packaging. 0 means all envs.")
    parser.add_argument("--target-base-blueprints-per-env", type=int, default=DEFAULT_STAGE5_TARGET_BASE_BLUEPRINTS_PER_ENV, help="Expected base blueprint count per env. 0 disables this gate.")
    parser.add_argument("--min-samples-per-env", type=int, default=DEFAULT_STAGE5_MIN_SAMPLES_PER_ENV, help="Minimum packaged sample count per env. 0 disables this gate.")
    parser.add_argument("--required-dialogue-modes", default=DEFAULT_STAGE5_REQUIRED_DIALOGUE_MODES, help="Comma-separated dialogue modes that must appear in the packaged dataset.")
    parser.add_argument("--min-distinct-difficulty-buckets", type=int, default=DEFAULT_STAGE5_MIN_DISTINCT_DIFFICULTY_BUCKETS, help="Minimum distinct difficulty buckets required globally and per env. 0 disables this gate.")
    parser.add_argument("--max-dialogue-mode-imbalance-ratio", type=float, default=DEFAULT_STAGE5_MAX_DIALOGUE_MODE_IMBALANCE_RATIO, help="Optional max ratio between the largest and smallest non-zero dialogue mode buckets. 0 disables this gate.")
    parser.add_argument("--max-difficulty-imbalance-ratio", type=float, default=DEFAULT_STAGE5_MAX_DIFFICULTY_IMBALANCE_RATIO, help="Optional max ratio between the largest and smallest non-zero difficulty buckets. 0 disables this gate.")
    parser.add_argument("--min-behavior-count", type=int, default=DEFAULT_STAGE5_MIN_BEHAVIOR_COUNT, help="Optional reporting floor for each unstable behavior. 0 disables this gate.")
    parser.add_argument(
        "--require-all-behaviors",
        dest="require_all_behaviors",
        action="store_true",
        default=DEFAULT_STAGE5_REQUIRE_ALL_BEHAVIORS,
        help="Require all known unstable behaviors to appear at least once.",
    )
    parser.add_argument(
        "--allow-missing-behaviors",
        dest="require_all_behaviors",
        action="store_false",
        help="Allow behavior coverage gaps while still reporting them.",
    )
    parser.add_argument(
        "--fail-on-qc-errors",
        action="store_true",
        default=DEFAULT_STAGE5_FAIL_ON_QC_ERRORS,
        help="Exit with a non-zero code when any Stage5 quality gate fails.",
    )
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_safe_snapshot(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _normalize_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _parse_required_dialogue_modes(raw_value: str) -> list[str]:
    allowed = ["single_turn", "multi_turn"]
    requested = {part.strip() for part in str(raw_value or "").split(",") if part.strip()}
    return [mode for mode in allowed if mode in requested]


def _payload_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return [item for item in payload["items"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise TypeError("Expected a list payload or a dict with an items array")


def _select_env_ids(blueprint_items: list[dict[str, Any]], max_envs: int) -> set[str] | None:
    if max_envs <= 0:
        return None
    selected: list[str] = []
    seen: set[str] = set()
    for item in blueprint_items:
        env_id = _normalize_string(item.get("env_id"))
        if not env_id or env_id in seen:
            continue
        seen.add(env_id)
        selected.append(env_id)
        if len(selected) >= max_envs:
            break
    return set(selected)


def _build_blueprint_index(payload: Any, selected_env_ids: set[str] | None) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    blueprint_index: dict[str, dict[str, Any]] = {}
    env_task_ids: dict[str, list[str]] = defaultdict(list)

    for state_item in _payload_items(payload):
        env_id = _normalize_string(state_item.get("env_id"))
        if selected_env_ids is not None and env_id not in selected_env_ids:
            continue
        config_id = _normalize_string(state_item.get("config_id"))
        task_blueprints = state_item.get("task_blueprints", []) if isinstance(state_item.get("task_blueprints"), list) else []
        for blueprint in task_blueprints:
            if not isinstance(blueprint, dict):
                continue
            task_id = _normalize_string(blueprint.get("task_id"))
            if not task_id:
                continue
            blueprint_index[task_id] = {
                "env_id": env_id,
                "config_id": config_id,
                "environment_summary": _normalize_string(state_item.get("environment_summary")),
                "environment_introduction": _normalize_string(state_item.get("environment_introduction")),
                "state_profile": _normalize_string(state_item.get("state_profile")),
                "state_stats": _json_safe_snapshot(state_item.get("state_stats", {})),
                "init_config": _json_safe_snapshot(state_item.get("init_config", {})),
                "blueprint": _json_safe_snapshot(blueprint),
            }
            env_task_ids[env_id].append(task_id)

    for env_id in env_task_ids:
        env_task_ids[env_id] = sorted({task_id for task_id in env_task_ids[env_id] if task_id})
    return blueprint_index, env_task_ids


def _build_dialogue_index(
    payload: Any,
    dialogue_field: str,
    task_key: str,
    selected_env_ids: set[str] | None,
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    dialogue_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    observed_task_ids: set[str] = set()

    for state_item in _payload_items(payload):
        env_id = _normalize_string(state_item.get("env_id"))
        if selected_env_ids is not None and env_id not in selected_env_ids:
            continue
        config_id = _normalize_string(state_item.get("config_id"))
        dialogues = state_item.get(dialogue_field, []) if isinstance(state_item.get(dialogue_field), list) else []
        for dialogue in dialogues:
            if not isinstance(dialogue, dict):
                continue
            task_id = _normalize_string(dialogue.get(task_key))
            if not task_id:
                continue
            observed_task_ids.add(task_id)
            dialogue_index[task_id].append(
                {
                    "env_id": env_id,
                    "config_id": config_id,
                    "dialogue": _json_safe_snapshot(dialogue),
                }
            )

    for task_id in dialogue_index:
        dialogue_index[task_id].sort(key=lambda item: _normalize_string(item.get("dialogue", {}).get("sample_id") or item.get("dialogue", {}).get("task_id")))
    return dialogue_index, observed_task_ids


def _build_gold_action_graph(gold_tool_trace: list[dict[str, Any]]) -> dict[str, Any]:
    required_read_nodes: list[dict[str, Any]] = []
    required_write_nodes: list[dict[str, Any]] = []
    precedence_edges: list[dict[str, int]] = []

    previous_step_index: int | None = None
    for step in gold_tool_trace:
        if not isinstance(step, dict):
            continue
        step_index = int(step.get("step_index", 0) or 0)
        node = {
            "step_index": step_index,
            "tool_name": _normalize_string(step.get("tool_name")),
            "arguments": _json_safe_snapshot(step.get("arguments", {})),
            "purpose": _normalize_string(step.get("purpose")),
            "user_turn_id": step.get("user_turn_id"),
        }
        state_diff = step.get("state_diff", {}) if isinstance(step.get("state_diff"), dict) else {}
        if state_diff:
            node["expected_state_diff"] = _json_safe_snapshot(state_diff)
            required_write_nodes.append(node)
        else:
            required_read_nodes.append(node)
        if previous_step_index is not None and step_index > 0:
            precedence_edges.append({"before_step_index": previous_step_index, "after_step_index": step_index})
        if step_index > 0:
            previous_step_index = step_index

    return {
        "graph_type": "linear_trace_v1",
        "required_read_nodes": required_read_nodes,
        "required_write_nodes": required_write_nodes,
        "clarify_nodes": [],
        "refuse_nodes": [],
        "precedence_edges": precedence_edges,
    }


def _collect_final_state_assertions(value: Any, path: str = "") -> list[dict[str, Any]]:
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
        assertions.extend(_collect_final_state_assertions(nested, nested_path))
    return assertions


def _build_micro_tasks(blueprint: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(blueprint.get("micro_tasks"), list) and blueprint.get("micro_tasks"):
        normalized_existing: list[dict[str, Any]] = []
        for item in blueprint.get("micro_tasks", []):
            if not isinstance(item, dict):
                continue
            task_id = _normalize_string(item.get("task_id"), f"subtask_{len(normalized_existing) + 1}")
            user_goal = _normalize_string(item.get("user_goal"))
            expected_outcome = _normalize_string(item.get("expected_outcome"))
            if not user_goal or not expected_outcome:
                continue
            normalized_existing.append(
                {
                    "task_id": task_id,
                    "user_goal": user_goal,
                    "task_type": _normalize_string(item.get("task_type"), "execute"),
                    "required_entities": _json_safe_snapshot(item.get("required_entities", [])),
                    "expected_outcome": expected_outcome,
                }
            )
        if normalized_existing:
            return normalized_existing

    multi_turn_tasks = blueprint.get("multi_turn_tasks", []) if isinstance(blueprint.get("multi_turn_tasks"), list) else []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(multi_turn_tasks, start=1):
        if not isinstance(item, dict):
            continue
        task_summary = _normalize_string(item.get("task_summary"))
        expected_outcome = _normalize_string(item.get("expected_outcome"))
        if not task_summary or not expected_outcome:
            continue
        normalized.append(
            {
                "task_id": f"subtask_{index}",
                "user_goal": task_summary,
                "task_type": _normalize_string(item.get("task_type"), _normalize_string(blueprint.get("task_type"), "execute")),
                "required_entities": _json_safe_snapshot(item.get("required_entities", blueprint.get("relevant_entities", []))),
                "expected_outcome": expected_outcome,
            }
        )
    if normalized:
        return normalized
    return [
        {
            "task_id": "subtask_1",
            "user_goal": _normalize_string(blueprint.get("goal_summary")),
            "task_type": _normalize_string(blueprint.get("task_type"), "execute"),
            "required_entities": _json_safe_snapshot(blueprint.get("relevant_entities", [])),
            "expected_outcome": _normalize_string(blueprint.get("expected_final_outcome")),
        }
    ]


def _infer_min_required_action_turns(dialogue: dict[str, Any], gold_tool_trace: list[dict[str, Any]]) -> int:
    budget = dialogue.get("action_turn_budget", {}) if isinstance(dialogue.get("action_turn_budget"), dict) else {}
    try:
        total = int(budget.get("total_action_turns", 0))
    except (TypeError, ValueError):
        total = 0
    if total > 0:
        return total
    user_turn_count = len(dialogue.get("user_turns", [])) if isinstance(dialogue.get("user_turns"), list) else 0
    return user_turn_count + len(gold_tool_trace) + 1


def _build_sample_metadata(blueprint_entry: dict[str, Any], dialogue: dict[str, Any], variant: str) -> dict[str, Any]:
    blueprint = blueprint_entry["blueprint"]
    if variant == "stable":
        source_metadata = dialogue.get("stable_dialogue_metadata", dialogue.get("collab_dialogue_metadata", {}))
    else:
        source_metadata = dialogue.get("unstable_dialogue_metadata", dialogue.get("noncollab_dialogue_metadata", {}))
    has_native_ground_truth = bool(blueprint.get("canonical_action_graph")) and bool(blueprint.get("final_state_assertions"))
    sample_metadata = {
        "environment_summary": blueprint_entry.get("environment_summary", ""),
        "environment_introduction": blueprint_entry.get("environment_introduction", ""),
        "state_profile": blueprint_entry.get("state_profile", ""),
        "state_stats": _json_safe_snapshot(blueprint_entry.get("state_stats", {})),
        "goal_summary": _normalize_string(blueprint.get("goal_summary")),
        "expected_phases": _json_safe_snapshot(blueprint.get("expected_phases", [])),
        "quality_checks": _normalize_string(blueprint.get("quality_checks")),
        "unique_answer_rationale": _normalize_string(blueprint.get("unique_answer_rationale")),
        "canonical_answer": _normalize_string(blueprint.get("canonical_answer")),
        "fallback_answer_hint": _normalize_string(blueprint.get("fallback_answer_hint")),
        "fallback_inferior_reason": _normalize_string(blueprint.get("fallback_inferior_reason")),
        "relevant_entities": _json_safe_snapshot(blueprint.get("relevant_entities", [])),
        "task_type": _normalize_string(blueprint.get("task_type"), "execute"),
        "clarification_requirements": _json_safe_snapshot(blueprint.get("clarification_requirements", [])),
        "refusal_requirements": _json_safe_snapshot(blueprint.get("refusal_requirements", [])),
        "forbidden_actions": _json_safe_snapshot(blueprint.get("forbidden_actions", [])),
        "tool_call_range": _json_safe_snapshot(blueprint.get("tool_call_range", {})),
        "tool_plan": _json_safe_snapshot(blueprint.get("tool_plan", [])),
        "replay_validation": _json_safe_snapshot(blueprint.get("replay_validation", {})),
        "source_dialogue_metadata": _json_safe_snapshot(source_metadata if isinstance(source_metadata, dict) else {}),
        "derived_ground_truth": not has_native_ground_truth,
        "supports_full_action_graph": bool(blueprint.get("canonical_action_graph")),
    }
    if variant == "unstable":
        sample_metadata["stable_user_turns"] = _json_safe_snapshot(dialogue.get("stable_user_turns", dialogue.get("source_collaborative_user_turns", [])))
        sample_metadata["rewrite_metadata"] = _json_safe_snapshot(dialogue.get("rewrite_metadata", {}))
        sample_metadata["trajectory_replay_validation"] = _json_safe_snapshot(dialogue.get("trajectory_replay_validation", {}))
    return sample_metadata


def _build_packaged_sample(blueprint_entry: dict[str, Any], dialogue: dict[str, Any], variant: str) -> dict[str, Any]:
    blueprint = blueprint_entry["blueprint"]
    gold_tool_trace = dialogue.get("agent_gold_trace", []) if isinstance(dialogue.get("agent_gold_trace"), list) else []
    expected_state_diff = dialogue.get("expected_final_state", {}) if isinstance(dialogue.get("expected_final_state"), dict) else {}
    if not expected_state_diff and isinstance(blueprint.get("final_state_diff"), dict):
        expected_state_diff = blueprint.get("final_state_diff", {})
    profile_bundle = dialogue.get("profile_bundle", []) if isinstance(dialogue.get("profile_bundle"), list) else []
    behavior_bundle = dialogue.get("behavior_bundle", []) if isinstance(dialogue.get("behavior_bundle"), list) else []
    stable_user_turns = dialogue.get("stable_user_turns", dialogue.get("user_turns", [])) if isinstance(dialogue.get("stable_user_turns", dialogue.get("user_turns", [])), list) else []
    trace_to_turn_alignment = dialogue.get("trace_to_turn_alignment", []) if isinstance(dialogue.get("trace_to_turn_alignment"), list) else []
    action_turn_budget = dialogue.get("action_turn_budget", {}) if isinstance(dialogue.get("action_turn_budget"), dict) else {}
    micro_tasks = _build_micro_tasks(blueprint)
    gold_action_graph = blueprint.get("canonical_action_graph") if isinstance(blueprint.get("canonical_action_graph"), dict) else _build_gold_action_graph(gold_tool_trace)
    final_state_assertions = blueprint.get("final_state_assertions") if isinstance(blueprint.get("final_state_assertions"), list) else _collect_final_state_assertions(expected_state_diff)

    try:
        min_required_action_turns = int(blueprint.get("min_required_action_turns", 0) or 0)
    except (TypeError, ValueError):
        min_required_action_turns = 0
    if min_required_action_turns <= 0:
        min_required_action_turns = _infer_min_required_action_turns(dialogue, gold_tool_trace)

    try:
        max_allowed_action_turns = int(action_turn_budget.get("max_allowed_action_turns", blueprint.get("agent_action_budget", 50)))
    except (TypeError, ValueError):
        max_allowed_action_turns = 50

    return {
        "sample_id": _normalize_string(dialogue.get("sample_id"), f"{_normalize_string(dialogue.get('task_id'))}_{variant}"),
        "env_id": blueprint_entry.get("env_id", ""),
        "config_id": blueprint_entry.get("config_id", ""),
        "task_id": _normalize_string(dialogue.get("task_id")),
        "variant": variant,
        "source_task_id": _normalize_string(dialogue.get("source_task_id"), _normalize_string(dialogue.get("task_id"))),
        "profile_bundle": _json_safe_snapshot(profile_bundle),
        "behavior_bundle": _json_safe_snapshot(behavior_bundle),
        "unstable_behavior": _normalize_string(dialogue.get("unstable_behavior"), behavior_bundle[0] if behavior_bundle else "") if variant == "unstable" else "",
        "dialogue_mode": _normalize_string(dialogue.get("dialogue_mode"), _normalize_string(blueprint.get("dialogue_mode"))),
        "difficulty_bucket": _normalize_string(dialogue.get("difficulty_bucket"), _normalize_string(blueprint.get("difficulty_bucket"))),
        "goal_summary": _normalize_string(dialogue.get("goal_summary"), _normalize_string(blueprint.get("goal_summary"))),
        "user_request": _normalize_string(dialogue.get("user_request"), _normalize_string(blueprint.get("user_request"))),
        "expected_final_outcome": _normalize_string(dialogue.get("expected_final_outcome"), _normalize_string(blueprint.get("expected_final_outcome"))),
        "task_type": _normalize_string(blueprint.get("task_type"), "execute"),
        "clarification_requirements": _json_safe_snapshot(blueprint.get("clarification_requirements", [])),
        "refusal_requirements": _json_safe_snapshot(blueprint.get("refusal_requirements", [])),
        "forbidden_actions": _json_safe_snapshot(blueprint.get("forbidden_actions", [])),
        "init_config": _json_safe_snapshot(blueprint_entry.get("init_config", {})),
        "user_turns": _json_safe_snapshot(dialogue.get("user_turns", [])),
        "stable_user_turns": _json_safe_snapshot(stable_user_turns),
        "trace_to_turn_alignment": _json_safe_snapshot(trace_to_turn_alignment),
        "rewrite_metadata": _json_safe_snapshot(dialogue.get("rewrite_metadata", {})) if variant == "unstable" else {},
        "micro_tasks": micro_tasks,
        "gold_action_graph": _json_safe_snapshot(gold_action_graph),
        "gold_tool_trace": _json_safe_snapshot(gold_tool_trace),
        "expected_state_diff": _json_safe_snapshot(expected_state_diff),
        "final_state_assertions": _json_safe_snapshot(final_state_assertions),
        "tool_call_budget": blueprint.get("tool_call_budget", dialogue.get("tool_call_budget", 0)),
        "tool_call_range": _json_safe_snapshot(blueprint.get("tool_call_range", {})),
        "min_required_action_turns": min_required_action_turns,
        "max_allowed_action_turns": max_allowed_action_turns,
        "action_turn_budget": _json_safe_snapshot(action_turn_budget),
        "metadata": _build_sample_metadata(blueprint_entry, dialogue, variant),
    }


def _validate_sample(sample: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    gold_tool_trace = sample.get("gold_tool_trace", []) if isinstance(sample.get("gold_tool_trace"), list) else []
    gold_action_graph = sample.get("gold_action_graph", {}) if isinstance(sample.get("gold_action_graph"), dict) else {}
    final_state_assertions = sample.get("final_state_assertions", []) if isinstance(sample.get("final_state_assertions"), list) else []
    expected_state_diff = sample.get("expected_state_diff", {}) if isinstance(sample.get("expected_state_diff"), dict) else {}
    action_turn_budget = sample.get("action_turn_budget", {}) if isinstance(sample.get("action_turn_budget"), dict) else {}
    task_type = _normalize_string(sample.get("task_type"), "execute")

    has_non_execution_ground_truth = bool(gold_action_graph.get("clarify_nodes") or gold_action_graph.get("refuse_nodes"))

    if not gold_tool_trace and not has_non_execution_ground_truth:
        failures.append("missing gold_tool_trace")
    if not gold_action_graph or (
        not gold_action_graph.get("required_read_nodes") and not gold_action_graph.get("required_write_nodes")
        and not gold_action_graph.get("clarify_nodes") and not gold_action_graph.get("refuse_nodes")
    ):
        failures.append("missing gold_action_graph")
    if not expected_state_diff and task_type == "execute":
        failures.append("missing expected_state_diff")
    if not final_state_assertions and task_type == "execute":
        failures.append("missing final_state_assertions")

    try:
        tool_call_budget = int(sample.get("tool_call_budget", 0) or 0)
    except (TypeError, ValueError):
        tool_call_budget = 0
        failures.append("invalid tool_call_budget")
    tool_call_count = len(gold_tool_trace)
    if tool_call_budget > 0 and tool_call_count > tool_call_budget:
        failures.append(f"tool_call_count {tool_call_count} exceeds tool_call_budget {tool_call_budget}")

    try:
        total_action_turns = int(action_turn_budget.get("total_action_turns", sample.get("min_required_action_turns", 0)) or 0)
    except (TypeError, ValueError):
        total_action_turns = 0
        failures.append("invalid total_action_turns")
    try:
        max_allowed_action_turns = int(sample.get("max_allowed_action_turns", 0) or 0)
    except (TypeError, ValueError):
        max_allowed_action_turns = 0
        failures.append("invalid max_allowed_action_turns")
    if max_allowed_action_turns > 0 and total_action_turns > max_allowed_action_turns:
        failures.append(
            f"total_action_turns {total_action_turns} exceeds max_allowed_action_turns {max_allowed_action_turns}"
        )

    if sample.get("variant") == "unstable":
        if not sample.get("behavior_bundle"):
            failures.append("missing behavior_bundle for unstable sample")
        if not sample.get("unstable_behavior"):
            failures.append("missing unstable_behavior for unstable sample")
        if not sample.get("rewrite_metadata"):
            failures.append("missing rewrite_metadata for unstable sample")
    if not sample.get("stable_user_turns"):
        failures.append("missing stable_user_turns")
    if not sample.get("trace_to_turn_alignment"):
        failures.append("missing trace_to_turn_alignment")

    return {
        "sample_id": sample.get("sample_id", ""),
        "task_id": sample.get("task_id", ""),
        "env_id": sample.get("env_id", ""),
        "variant": sample.get("variant", ""),
        "tool_call_count": tool_call_count,
        "tool_call_budget": tool_call_budget,
        "total_action_turns": total_action_turns,
        "max_allowed_action_turns": max_allowed_action_turns,
        "qc_failures": failures,
        "qc_passed": not failures,
    }


def _sample_sort_key(sample: dict[str, Any]) -> tuple[str, str, int, str]:
    variant_order = 0 if sample.get("variant") == "stable" else 1
    return (
        _normalize_string(sample.get("env_id")),
        _normalize_string(sample.get("task_id")),
        variant_order,
        _normalize_string(sample.get("sample_id")),
    )


def _build_task_pair_summary(
    task_id: str,
    env_id: str,
    stable_sample: dict[str, Any] | None,
    unstable_samples: list[dict[str, Any]],
    sample_qc_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    stable_sample_id = _normalize_string(stable_sample.get("sample_id")) if isinstance(stable_sample, dict) else ""
    unstable_sample_ids = [
        _normalize_string(sample.get("sample_id"))
        for sample in unstable_samples
        if isinstance(sample, dict) and _normalize_string(sample.get("sample_id"))
    ]

    if not stable_sample_id:
        failures.append("missing stable variant")
    if not unstable_sample_ids:
        failures.append("missing unstable variant")

    if stable_sample_id and not sample_qc_by_id.get(stable_sample_id, {}).get("qc_passed", False):
        failures.append("stable variant failed sample-level QC")
    for sample_id in unstable_sample_ids:
        if not sample_qc_by_id.get(sample_id, {}).get("qc_passed", False):
            failures.append(f"unstable variant failed sample-level QC: {sample_id}")

    return {
        "task_id": task_id,
        "env_id": env_id,
        "stable_sample_id": stable_sample_id,
        "unstable_sample_ids": unstable_sample_ids,
        "variant_count": (1 if stable_sample_id else 0) + len(unstable_sample_ids),
        "qc_failures": failures,
        "qc_passed": not failures,
    }


def _build_env_summary(
    env_id: str,
    task_ids: list[str],
    packaged_stable: list[dict[str, Any]],
    packaged_unstable: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    stable_task_ids = sorted({_normalize_string(sample.get("task_id")) for sample in packaged_stable if _normalize_string(sample.get("task_id"))})
    unstable_task_ids = sorted({_normalize_string(sample.get("source_task_id") or sample.get("task_id")) for sample in packaged_unstable if _normalize_string(sample.get("source_task_id") or sample.get("task_id"))})
    all_samples = list(packaged_stable) + list(packaged_unstable)
    dialogue_modes_present = sorted({_normalize_string(sample.get("dialogue_mode")) for sample in all_samples if _normalize_string(sample.get("dialogue_mode"))})
    difficulty_buckets_present = sorted({_normalize_string(sample.get("difficulty_bucket")) for sample in all_samples if _normalize_string(sample.get("difficulty_bucket"))})

    missing_stable_task_ids = [task_id for task_id in task_ids if task_id not in set(stable_task_ids)]
    missing_unstable_task_ids = [task_id for task_id in task_ids if task_id not in set(unstable_task_ids)]

    failures: list[str] = []
    base_blueprint_count = len(task_ids)
    sample_count = len(all_samples)
    if args.target_base_blueprints_per_env > 0 and base_blueprint_count < args.target_base_blueprints_per_env:
        failures.append(
            f"base_blueprint_count {base_blueprint_count} is below target_base_blueprints_per_env {args.target_base_blueprints_per_env}"
        )
    if missing_stable_task_ids:
        failures.append(f"missing stable variants for task_ids: {missing_stable_task_ids}")
    if missing_unstable_task_ids:
        failures.append(f"missing unstable variants for task_ids: {missing_unstable_task_ids}")
    if args.min_samples_per_env > 0 and sample_count < args.min_samples_per_env:
        failures.append(f"sample_count {sample_count} is below min_samples_per_env {args.min_samples_per_env}")

    required_dialogue_modes = _parse_required_dialogue_modes(args.required_dialogue_modes)
    missing_dialogue_modes = [mode for mode in required_dialogue_modes if mode not in set(dialogue_modes_present)]
    if missing_dialogue_modes:
        failures.append(f"missing required dialogue modes: {missing_dialogue_modes}")

    if args.min_distinct_difficulty_buckets > 0 and len(difficulty_buckets_present) < args.min_distinct_difficulty_buckets:
        failures.append(
            "distinct difficulty bucket count "
            f"{len(difficulty_buckets_present)} is below min_distinct_difficulty_buckets {args.min_distinct_difficulty_buckets}"
        )

    return {
        "env_id": env_id,
        "base_blueprint_count": base_blueprint_count,
        "stable_sample_count": len(packaged_stable),
        "unstable_sample_count": len(packaged_unstable),
        "sample_count": sample_count,
        "expected_paired_sample_count": base_blueprint_count * 2,
        "task_ids": task_ids,
        "stable_task_ids": stable_task_ids,
        "unstable_task_ids": unstable_task_ids,
        "missing_stable_task_ids": missing_stable_task_ids,
        "missing_unstable_task_ids": missing_unstable_task_ids,
        "dialogue_modes_present": dialogue_modes_present,
        "difficulty_buckets_present": difficulty_buckets_present,
        "qc_failures": failures,
        "qc_passed": not failures,
    }


def _imbalance_ratio(counter: Counter[str]) -> float | None:
    non_zero_counts = [count for count in counter.values() if count > 0]
    if len(non_zero_counts) < 2:
        return None
    smallest = min(non_zero_counts)
    largest = max(non_zero_counts)
    if smallest <= 0:
        return None
    return round(largest / smallest, 4)


def _build_global_summary(
    stable_samples: list[dict[str, Any]],
    unstable_samples: list[dict[str, Any]],
    full_samples: list[dict[str, Any]],
    orphan_stable_task_ids: list[str],
    orphan_unstable_task_ids: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    behavior_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    dialogue_mode_counts: Counter[str] = Counter()
    difficulty_bucket_counts: Counter[str] = Counter()
    behavior_bundle_size_counts: Counter[str] = Counter()

    for sample in unstable_samples:
        behavior_bundle = sample.get("behavior_bundle", []) if isinstance(sample.get("behavior_bundle"), list) else []
        behavior_bundle_size_counts[str(len(behavior_bundle))] += 1
        for behavior in behavior_bundle:
            key = _normalize_string(behavior)
            if key:
                behavior_counts[key] += 1
        for profile in sample.get("profile_bundle", []):
            key = _normalize_string(profile)
            if key:
                profile_counts[key] += 1

    for sample in full_samples:
        mode = _normalize_string(sample.get("dialogue_mode"))
        difficulty_bucket = _normalize_string(sample.get("difficulty_bucket"))
        if mode:
            dialogue_mode_counts[mode] += 1
        if difficulty_bucket:
            difficulty_bucket_counts[difficulty_bucket] += 1

    missing_behaviors = [behavior for behavior in KNOWN_UNSTABLE_BEHAVIORS if behavior_counts.get(behavior, 0) == 0]
    under_min_behaviors = {
        behavior: behavior_counts.get(behavior, 0)
        for behavior in KNOWN_UNSTABLE_BEHAVIORS
        if args.min_behavior_count > 0 and behavior_counts.get(behavior, 0) < args.min_behavior_count
    }
    required_dialogue_modes = _parse_required_dialogue_modes(args.required_dialogue_modes)
    missing_dialogue_modes = [mode for mode in required_dialogue_modes if dialogue_mode_counts.get(mode, 0) == 0]
    distinct_difficulty_bucket_count = sum(1 for count in difficulty_bucket_counts.values() if count > 0)
    dialogue_mode_imbalance_ratio = _imbalance_ratio(dialogue_mode_counts)
    difficulty_imbalance_ratio = _imbalance_ratio(difficulty_bucket_counts)
    behavior_token_count = sum(behavior_counts.values())
    min_required_behavior_count = max(1 if args.require_all_behaviors else 0, max(0, int(args.min_behavior_count or 0)))
    required_behavior_token_floor = len(KNOWN_UNSTABLE_BEHAVIORS) * min_required_behavior_count
    behavior_token_slack = max(0, behavior_token_count - required_behavior_token_floor)
    max_behavior_bundle_size = max((int(size) for size in behavior_bundle_size_counts if size.isdigit()), default=0)
    single_behavior_dialogue_count = behavior_bundle_size_counts.get("1", 0)
    multi_behavior_dialogue_count = len(unstable_samples) - single_behavior_dialogue_count
    observed_single_behavior_majority = single_behavior_dialogue_count > multi_behavior_dialogue_count

    extra_behavior_tokens_needed = max(0, required_behavior_token_floor - len(unstable_samples))
    if extra_behavior_tokens_needed == 0:
        min_multi_behavior_dialogues_needed = 0
        single_behavior_majority_feasible = True
    elif max_behavior_bundle_size <= 1:
        min_multi_behavior_dialogues_needed = None
        single_behavior_majority_feasible = False
    else:
        min_multi_behavior_dialogues_needed = ceil(extra_behavior_tokens_needed / (max_behavior_bundle_size - 1))
        single_behavior_majority_feasible = min_multi_behavior_dialogues_needed < (len(unstable_samples) / 2)

    primary_behavior_counts = Counter(_normalize_string(sample.get("unstable_behavior")) for sample in unstable_samples)
    primary_behavior_counts.pop("", None)
    primary_behavior_distribution = {
        behavior: round(primary_behavior_counts.get(behavior, 0) / len(unstable_samples), 4) if unstable_samples else 0.0
        for behavior in KNOWN_UNSTABLE_BEHAVIORS
    }

    planning_warnings: list[str] = []
    if required_behavior_token_floor > 0 and behavior_token_slack == 0:
        planning_warnings.append(
            "all observed unstable behavior tokens are consumed by the minimum behavior floor; this split has no slack to approximate empirical behavior frequencies"
        )
    if not observed_single_behavior_majority:
        if min_multi_behavior_dialogues_needed is None:
            planning_warnings.append(
                "majority single-behavior unstable samples is impossible because the current split needs more behavior tokens than one behavior per dialogue can provide"
            )
        elif not single_behavior_majority_feasible:
            planning_warnings.append(
                "majority single-behavior unstable samples is mathematically impossible for this split under the observed behavior-bundle size cap"
            )

    failures: list[str] = []
    if orphan_stable_task_ids:
        failures.append(f"orphan stable task_ids not found in Stage2 blueprints: {orphan_stable_task_ids}")
    if orphan_unstable_task_ids:
        failures.append(f"orphan unstable task_ids not found in Stage2 blueprints: {orphan_unstable_task_ids}")
    if args.require_all_behaviors and missing_behaviors:
        failures.append(f"missing required unstable behaviors: {missing_behaviors}")
    if under_min_behaviors:
        failures.append(f"behaviors below min_behavior_count {args.min_behavior_count}: {under_min_behaviors}")
    if missing_dialogue_modes:
        failures.append(f"missing required dialogue modes in packaged samples: {missing_dialogue_modes}")
    if args.min_distinct_difficulty_buckets > 0 and distinct_difficulty_bucket_count < args.min_distinct_difficulty_buckets:
        failures.append(
            "distinct difficulty bucket count "
            f"{distinct_difficulty_bucket_count} is below min_distinct_difficulty_buckets {args.min_distinct_difficulty_buckets}"
        )
    if (
        args.max_dialogue_mode_imbalance_ratio > 0
        and dialogue_mode_imbalance_ratio is not None
        and dialogue_mode_imbalance_ratio > args.max_dialogue_mode_imbalance_ratio
    ):
        failures.append(
            "dialogue mode imbalance ratio "
            f"{dialogue_mode_imbalance_ratio} exceeds max_dialogue_mode_imbalance_ratio {args.max_dialogue_mode_imbalance_ratio}"
        )
    if (
        args.max_difficulty_imbalance_ratio > 0
        and difficulty_imbalance_ratio is not None
        and difficulty_imbalance_ratio > args.max_difficulty_imbalance_ratio
    ):
        failures.append(
            "difficulty bucket imbalance ratio "
            f"{difficulty_imbalance_ratio} exceeds max_difficulty_imbalance_ratio {args.max_difficulty_imbalance_ratio}"
        )

    return {
        "stable_sample_count": len(stable_samples),
        "unstable_sample_count": len(unstable_samples),
        "sample_count": len(full_samples),
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "primary_behavior_counts": dict(sorted(primary_behavior_counts.items())),
        "primary_behavior_distribution": primary_behavior_distribution,
        "reference_primary_behavior_distribution": UNSTABLE_PRIMARY_PRIORS,
        "behavior_bundle_size_counts": dict(sorted(behavior_bundle_size_counts.items())),
        "behavior_token_count": behavior_token_count,
        "required_behavior_token_floor": required_behavior_token_floor,
        "behavior_token_slack": behavior_token_slack,
        "max_behavior_bundle_size": max_behavior_bundle_size,
        "single_behavior_dialogue_count": single_behavior_dialogue_count,
        "multi_behavior_dialogue_count": multi_behavior_dialogue_count,
        "observed_single_behavior_majority": observed_single_behavior_majority,
        "single_behavior_majority_feasible_under_observed_bundle_cap": single_behavior_majority_feasible,
        "min_multi_behavior_dialogues_needed_for_floor": min_multi_behavior_dialogues_needed,
        "profile_counts": dict(sorted(profile_counts.items())),
        "dialogue_mode_counts": dict(sorted(dialogue_mode_counts.items())),
        "difficulty_bucket_counts": dict(sorted(difficulty_bucket_counts.items())),
        "missing_behaviors": missing_behaviors,
        "behaviors_below_min_count": under_min_behaviors,
        "missing_dialogue_modes": missing_dialogue_modes,
        "distinct_difficulty_bucket_count": distinct_difficulty_bucket_count,
        "dialogue_mode_imbalance_ratio": dialogue_mode_imbalance_ratio,
        "difficulty_imbalance_ratio": difficulty_imbalance_ratio,
        "orphan_stable_task_ids": orphan_stable_task_ids,
        "orphan_unstable_task_ids": orphan_unstable_task_ids,
        "planning_warnings": planning_warnings,
        "qc_failures": failures,
        "qc_passed": not failures,
    }


def main() -> None:
    args = parse_args()

    blueprint_payload = read_json_file(args.blueprint_file)
    blueprint_items = _payload_items(blueprint_payload)
    selected_env_ids = _select_env_ids(blueprint_items, args.max_envs)

    blueprint_index, env_task_ids = _build_blueprint_index(blueprint_payload, selected_env_ids)
    stable_payload = read_json_file(args.stable_file)
    unstable_payload = read_json_file(args.unstable_file)
    stable_index, stable_task_ids = _build_dialogue_index(stable_payload, "stable_dialogues", "task_id", selected_env_ids)
    if not stable_index:
        stable_index, stable_task_ids = _build_dialogue_index(stable_payload, "collaborative_dialogues", "task_id", selected_env_ids)
    unstable_index, unstable_task_ids = _build_dialogue_index(
        unstable_payload,
        "unstable_dialogues",
        "source_task_id",
        selected_env_ids,
    )
    if not unstable_index:
        unstable_index, unstable_task_ids = _build_dialogue_index(
            unstable_payload,
            "noncollaborative_dialogues",
            "source_task_id",
            selected_env_ids,
        )

    packaged_stable: list[dict[str, Any]] = []
    packaged_unstable: list[dict[str, Any]] = []
    sample_qc_entries: list[dict[str, Any]] = []
    sample_qc_by_id: dict[str, dict[str, Any]] = {}
    task_pair_summaries: list[dict[str, Any]] = []

    for task_id in sorted(blueprint_index):
        blueprint_entry = blueprint_index[task_id]
        stable_dialogue_entries = stable_index.get(task_id, [])
        stable_sample: dict[str, Any] | None = None
        if stable_dialogue_entries:
            stable_sample = _build_packaged_sample(blueprint_entry, stable_dialogue_entries[0]["dialogue"], "stable")
            packaged_stable.append(stable_sample)
            qc_entry = _validate_sample(stable_sample)
            sample_qc_entries.append(qc_entry)
            sample_qc_by_id[_normalize_string(stable_sample.get("sample_id"))] = qc_entry

        task_unstable_samples: list[dict[str, Any]] = []
        for unstable_entry in unstable_index.get(task_id, []):
            packaged_sample = _build_packaged_sample(blueprint_entry, unstable_entry["dialogue"], "unstable")
            packaged_unstable.append(packaged_sample)
            task_unstable_samples.append(packaged_sample)
            qc_entry = _validate_sample(packaged_sample)
            sample_qc_entries.append(qc_entry)
            sample_qc_by_id[_normalize_string(packaged_sample.get("sample_id"))] = qc_entry

        task_pair_summaries.append(
            _build_task_pair_summary(task_id, blueprint_entry.get("env_id", ""), stable_sample, task_unstable_samples, sample_qc_by_id)
        )

    packaged_stable.sort(key=_sample_sort_key)
    packaged_unstable.sort(key=_sample_sort_key)
    full_samples = sorted(packaged_stable + packaged_unstable, key=_sample_sort_key)

    env_summaries: list[dict[str, Any]] = []
    for env_id in sorted(env_task_ids):
        env_stable_samples = [sample for sample in packaged_stable if _normalize_string(sample.get("env_id")) == env_id]
        env_unstable_samples = [sample for sample in packaged_unstable if _normalize_string(sample.get("env_id")) == env_id]
        env_summaries.append(_build_env_summary(env_id, env_task_ids[env_id], env_stable_samples, env_unstable_samples, args))

    orphan_stable_task_ids = sorted(task_id for task_id in stable_task_ids if task_id not in blueprint_index)
    orphan_unstable_task_ids = sorted(task_id for task_id in unstable_task_ids if task_id not in blueprint_index)
    global_summary = _build_global_summary(
        packaged_stable,
        packaged_unstable,
        full_samples,
        orphan_stable_task_ids,
        orphan_unstable_task_ids,
        args,
    )

    sample_qc_failure_count = sum(1 for entry in sample_qc_entries if not entry.get("qc_passed", False))
    task_pair_failure_count = sum(1 for entry in task_pair_summaries if not entry.get("qc_passed", False))
    env_qc_failure_count = sum(1 for entry in env_summaries if not entry.get("qc_passed", False))
    global_qc_failure_count = 0 if global_summary.get("qc_passed", False) else 1
    qc_passed = sample_qc_failure_count == 0 and task_pair_failure_count == 0 and env_qc_failure_count == 0 and global_qc_failure_count == 0

    save_jsonl_file(args.output_stable_file, packaged_stable)
    save_jsonl_file(args.output_unstable_file, packaged_unstable)
    save_jsonl_file(args.output_full_file, full_samples)

    report_payload = {
        "metadata": {
            "generated_at": _now_iso(),
            "blueprint_file": str(Path(args.blueprint_file).resolve()),
            "stable_file": str(Path(args.stable_file).resolve()),
            "unstable_file": str(Path(args.unstable_file).resolve()),
            "output_stable_file": str(Path(args.output_stable_file).resolve()),
            "output_unstable_file": str(Path(args.output_unstable_file).resolve()),
            "output_full_file": str(Path(args.output_full_file).resolve()),
            "target_base_blueprints_per_env": args.target_base_blueprints_per_env,
            "min_samples_per_env": args.min_samples_per_env,
            "required_dialogue_modes": _parse_required_dialogue_modes(args.required_dialogue_modes),
            "min_distinct_difficulty_buckets": args.min_distinct_difficulty_buckets,
            "max_dialogue_mode_imbalance_ratio": args.max_dialogue_mode_imbalance_ratio,
            "max_difficulty_imbalance_ratio": args.max_difficulty_imbalance_ratio,
            "min_behavior_count": args.min_behavior_count,
            "require_all_behaviors": args.require_all_behaviors,
            "selected_env_count": len(env_task_ids),
            "base_task_count": len(blueprint_index),
            "stable_sample_count": len(packaged_stable),
            "unstable_sample_count": len(packaged_unstable),
            "sample_count": len(full_samples),
            "sample_qc_failure_count": sample_qc_failure_count,
            "task_pair_failure_count": task_pair_failure_count,
            "env_qc_failure_count": env_qc_failure_count,
            "global_qc_failure_count": global_qc_failure_count,
            "qc_passed": qc_passed,
        },
        "sample_qc": sample_qc_entries,
        "task_pair_summary": task_pair_summaries,
        "env_summary": env_summaries,
        "global_summary": global_summary,
    }
    save_json_file(args.report_file, report_payload)

    print(json.dumps(report_payload["metadata"], ensure_ascii=False, indent=2))
    print(f"Saved Stage5 stable benchmark to {args.output_stable_file}")
    print(f"Saved Stage5 unstable benchmark to {args.output_unstable_file}")
    print(f"Saved Stage5 full benchmark to {args.output_full_file}")
    print(f"Saved Stage5 packaging report to {args.report_file}")

    if args.fail_on_qc_errors and not qc_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()