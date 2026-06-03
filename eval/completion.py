import json
from typing import Any


WRITE_VERB_HINTS = {
    "add",
    "assign",
    "block",
    "cancel",
    "complete",
    "create",
    "delete",
    "force",
    "release",
    "remove",
    "reschedule",
    "schedule",
    "set",
    "unassign",
    "update",
}

IGNORABLE_OPTIONAL_ARGUMENT_NAMES = {
    "changed_by",
    "comment",
    "message",
    "note",
    "reason",
}

IGNORABLE_DYNAMIC_ARGUMENT_NAMES = {
    "current_time",
    "request_time",
    "timestamp",
}

SOFT_STATE_FIELD_NAMES = {
    "dismissal_time",
    "log_id",
    "timestamp",
}

SOFT_STATE_PATH_PREFIXES = {
    "adherence_log",
}

VISIBLE_STATE_AGENT_CONTEXT_MODES = {
    "visible_state",
}


def _json_safe_snapshot(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _normalize_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe_snapshot(value), ensure_ascii=False, sort_keys=True, default=str)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _values_equivalent(expected: Any, actual: Any) -> bool:
    if _is_numeric(expected) and _is_numeric(actual):
        return float(expected) == float(actual)
    return _json_safe_snapshot(expected) == _json_safe_snapshot(actual)


def _path_parts(path: str) -> tuple[str, ...]:
    return tuple(segment for segment in str(path or "").split(".") if segment)


def _is_soft_state_path(path: str | tuple[str, ...]) -> bool:
    parts = path if isinstance(path, tuple) else _path_parts(path)
    if not parts:
        return False
    return parts[0] in SOFT_STATE_PATH_PREFIXES or parts[-1] in SOFT_STATE_FIELD_NAMES


def _semantic_contains(
    expected: Any,
    actual: Any,
    path: tuple[str, ...] = (),
    ignored_keys: set[str] | None = None,
) -> bool:
    if _is_soft_state_path(path):
        return True
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        for key, value in expected.items():
            if ignored_keys and key in ignored_keys:
                continue
            next_path = path + (key,)
            if _is_soft_state_path(next_path):
                continue
            if key not in actual:
                return False
            if not _semantic_contains(value, actual[key], next_path):
                return False
        return True
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return False
        return all(_semantic_contains(expected_item, actual_item, path) for expected_item, actual_item in zip(expected, actual))
    return _values_equivalent(expected, actual)


def _generated_identifier_keys(expected_value: Any, path: str) -> set[str]:
    if not isinstance(expected_value, dict):
        return set()
    leaf = _path_parts(path)
    if not leaf:
        return set()
    path_leaf = leaf[-1]
    return {
        key
        for key, value in expected_value.items()
        if key.endswith("_id") and isinstance(value, str) and value == path_leaf
    }


def _find_semantic_added_match(
    path: str,
    expected_value: Any,
    initial_state: dict[str, Any],
    final_state: dict[str, Any],
) -> tuple[bool, Any]:
    parts = _path_parts(path)
    if len(parts) < 2 or not isinstance(expected_value, dict):
        return False, None
    parent_path = ".".join(parts[:-1])
    initial_parent_exists, initial_parent = _resolve_path(initial_state, parent_path)
    final_parent_exists, final_parent = _resolve_path(final_state, parent_path)
    if not final_parent_exists or not isinstance(final_parent, dict):
        return False, None
    initial_items = initial_parent if initial_parent_exists and isinstance(initial_parent, dict) else {}
    ignored_keys = _generated_identifier_keys(expected_value, path)
    for key, candidate in final_parent.items():
        if key in initial_items or not isinstance(candidate, dict):
            continue
        candidate_path = tuple(parts[:-1]) + (key,)
        if _semantic_contains(expected_value, candidate, candidate_path, ignored_keys=ignored_keys):
            return True, candidate
    return False, None


def _build_tool_schema_index(sample: dict[str, Any]) -> dict[str, dict[str, Any]]:
    runtime = sample.get("runtime", {}) if isinstance(sample.get("runtime"), dict) else {}
    raw_tools = runtime.get("tools", []) if isinstance(runtime.get("tools"), list) else []
    index: dict[str, dict[str, Any]] = {}
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            continue
        tool_name = _normalize_string(raw_tool.get("name"))
        parameters = raw_tool.get("parameters") if isinstance(raw_tool.get("parameters"), dict) else {}
        if tool_name:
            index[tool_name] = parameters
    return index


def _get_optional_parameter_schema(
    tool_name: str,
    key: str,
    tool_schema_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    schema = tool_schema_index.get(tool_name, {})
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    property_schema = properties.get(key)
    return property_schema if isinstance(property_schema, dict) else {}


def _is_omissible_optional_argument(
    tool_name: str,
    key: str,
    expected_value: Any,
    tool_schema_index: dict[str, dict[str, Any]],
) -> bool:
    if expected_value is None:
        return True
    if isinstance(expected_value, list) and not expected_value:
        return True
    property_schema = _get_optional_parameter_schema(tool_name, key, tool_schema_index)
    if property_schema and "default" in property_schema:
        return _values_equivalent(expected_value, property_schema.get("default"))
    return False


def _resolve_path(container: Any, path: str) -> tuple[bool, Any]:
    current = container
    for part in [segment for segment in str(path or "").split(".") if segment]:
        if isinstance(current, dict):
            if part not in current:
                return False, None
            current = current[part]
            continue
        return False, None
    return True, current


def _diff_contains(expected: Any, actual: Any) -> bool:
    return _semantic_contains(expected, actual)


def _arguments_match(
    tool_name: str,
    expected_arguments: dict[str, Any],
    actual_arguments: dict[str, Any],
    tool_schema_index: dict[str, dict[str, Any]],
) -> bool:
    schema = tool_schema_index.get(tool_name, {})
    required_keys = schema.get("required", []) if isinstance(schema.get("required"), list) else []

    for key in required_keys:
        if key not in expected_arguments or key not in actual_arguments:
            return False
        if key in IGNORABLE_DYNAMIC_ARGUMENT_NAMES:
            continue
        if not _values_equivalent(expected_arguments.get(key), actual_arguments.get(key)):
            return False

    for key, expected_value in expected_arguments.items():
        if key in required_keys:
            continue
        if key in IGNORABLE_OPTIONAL_ARGUMENT_NAMES:
            continue
        if key in IGNORABLE_DYNAMIC_ARGUMENT_NAMES:
            if key not in actual_arguments:
                return False
            continue
        if key not in actual_arguments:
            if _is_omissible_optional_argument(tool_name, key, expected_value, tool_schema_index):
                continue
            return False
        if not _values_equivalent(expected_value, actual_arguments.get(key)):
            return False

    return True


def _tool_node_matches(
    node: dict[str, Any],
    actual_step: dict[str, Any],
    tool_schema_index: dict[str, dict[str, Any]],
) -> bool:
    if _normalize_string(node.get("tool_name")) != _normalize_string(actual_step.get("tool_name")):
        return False
    tool_name = _normalize_string(node.get("tool_name"))
    expected_arguments = node.get("arguments", {}) if isinstance(node.get("arguments"), dict) else {}
    actual_arguments = actual_step.get("arguments", {}) if isinstance(actual_step.get("arguments"), dict) else {}
    if not _arguments_match(tool_name, expected_arguments, actual_arguments, tool_schema_index):
        return False
    expected_state_diff = node.get("expected_state_diff") if isinstance(node.get("expected_state_diff"), dict) else {}
    if expected_state_diff and not _diff_contains(expected_state_diff, actual_step.get("state_diff", {})):
        return False
    return True


def _match_required_nodes(
    required_nodes: list[dict[str, Any]],
    actual_tool_trace: list[dict[str, Any]],
    used_actual_positions: set[int],
    tool_schema_index: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    results: list[dict[str, Any]] = []
    step_to_actual_position: dict[int, int] = {}

    for node in required_nodes:
        expected_step_index = int(node.get("step_index", 0) or 0)
        matched_actual_position = None
        matched_actual_step = None
        for position, actual_step in enumerate(actual_tool_trace, start=1):
            if position in used_actual_positions:
                continue
            if _tool_node_matches(node, actual_step, tool_schema_index):
                matched_actual_position = position
                matched_actual_step = actual_step
                used_actual_positions.add(position)
                break
        matched = matched_actual_position is not None
        if matched and expected_step_index > 0:
            step_to_actual_position[expected_step_index] = matched_actual_position
        results.append(
            {
                "step_index": expected_step_index,
                "tool_name": _normalize_string(node.get("tool_name")),
                "matched": matched,
                "matched_actual_position": matched_actual_position,
                "matched_actual_tool_name": _normalize_string((matched_actual_step or {}).get("tool_name")),
            }
        )

    return results, step_to_actual_position


def _match_read_equivalence_sets(
    equivalence_sets: list[dict[str, Any]],
    actual_tool_trace: list[dict[str, Any]],
    used_actual_positions: set[int],
    tool_schema_index: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    results: list[dict[str, Any]] = []
    step_to_actual_position: dict[int, int] = {}

    for item in equivalence_sets:
        expected_step_index = int(item.get("step_index", 0) or 0)
        equivalent_nodes = item.get("equivalent_nodes", []) if isinstance(item.get("equivalent_nodes"), list) else []
        matched_actual_position = None
        matched_node = None
        matched_actual_step = None
        for position, actual_step in enumerate(actual_tool_trace, start=1):
            if position in used_actual_positions:
                continue
            for node in equivalent_nodes:
                if not isinstance(node, dict):
                    continue
                if _tool_node_matches(node, actual_step, tool_schema_index):
                    matched_actual_position = position
                    matched_node = node
                    matched_actual_step = actual_step
                    used_actual_positions.add(position)
                    break
            if matched_actual_position is not None:
                break
        matched = matched_actual_position is not None
        if matched and expected_step_index > 0:
            step_to_actual_position[expected_step_index] = matched_actual_position
        results.append(
            {
                "step_index": expected_step_index,
                "matched": matched,
                "matched_actual_position": matched_actual_position,
                "matched_tool_name": _normalize_string((matched_node or {}).get("tool_name")),
                "actual_tool_name": _normalize_string((matched_actual_step or {}).get("tool_name")),
            }
        )

    return results, step_to_actual_position


def _match_required_tool_names(
    required_nodes: list[dict[str, Any]],
    actual_tool_trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    next_position = 0
    for node in required_nodes:
        tool_name = _normalize_string(node.get("tool_name"))
        matched_actual_position = None
        for position, actual_step in enumerate(actual_tool_trace, start=1):
            if position <= next_position:
                continue
            if _normalize_string(actual_step.get("tool_name")) == tool_name:
                matched_actual_position = position
                next_position = position
                break
        results.append(
            {
                "step_index": int(node.get("step_index", 0) or 0),
                "tool_name": tool_name,
                "matched": matched_actual_position is not None,
                "matched_actual_position": matched_actual_position,
            }
        )
    return results


def _evaluate_precedence(
    precedence_edges: list[dict[str, Any]],
    step_to_actual_position: dict[int, int],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for edge in precedence_edges:
        before_step_index = int(edge.get("before_step_index", 0) or 0)
        after_step_index = int(edge.get("after_step_index", 0) or 0)
        before_position = step_to_actual_position.get(before_step_index)
        after_position = step_to_actual_position.get(after_step_index)
        satisfied = bool(
            before_position is not None
            and after_position is not None
            and before_position < after_position
        )
        results.append(
            {
                "before_step_index": before_step_index,
                "after_step_index": after_step_index,
                "before_actual_position": before_position,
                "after_actual_position": after_position,
                "satisfied": satisfied,
            }
        )
    return results


def _extract_write_tool_verbs(actual_tool_trace: list[dict[str, Any]]) -> list[dict[str, str]]:
    extracted: list[dict[str, str]] = []
    for actual_step in actual_tool_trace:
        if not actual_step.get("is_write"):
            continue
        tool_name = _normalize_string(actual_step.get("tool_name"))
        parts = [part for part in tool_name.lower().split("_") if part]
        verb = ""
        for part in parts:
            if part in WRITE_VERB_HINTS:
                verb = part
                break
        extracted.append({"tool_name": tool_name, "verb": verb})
    return extracted


def _step_matches_exact_forbidden_target(actual_step: dict[str, Any], suffix: str) -> bool:
    target = suffix.strip()
    if not target:
        return True
    arguments = actual_step.get("arguments", {}) if isinstance(actual_step.get("arguments"), dict) else {}
    return any(_normalize_string(value).lower() == target for value in arguments.values())


def _step_uses_inactive_medication(actual_step: dict[str, Any], initial_state: dict[str, Any]) -> bool:
    arguments = actual_step.get("arguments", {}) if isinstance(actual_step.get("arguments"), dict) else {}
    medication_id = _normalize_string(arguments.get("medication_id"))
    medications = initial_state.get("medications", {}) if isinstance(initial_state.get("medications"), dict) else {}
    medication = medications.get(medication_id, {}) if medication_id else {}
    return bool(medication) and medication.get("active_status") is False


def _step_has_conflicting_schedule(actual_step: dict[str, Any], initial_state: dict[str, Any]) -> bool:
    arguments = actual_step.get("arguments", {}) if isinstance(actual_step.get("arguments"), dict) else {}
    user_id = _normalize_string(arguments.get("user_id"))
    medication_id = _normalize_string(arguments.get("medication_id"))
    scheduled_time = arguments.get("scheduled_time")
    if not user_id or not medication_id or scheduled_time is None:
        return False
    reminders = initial_state.get("reminders", {}) if isinstance(initial_state.get("reminders"), dict) else {}
    for reminder in reminders.values():
        if not isinstance(reminder, dict):
            continue
        if _normalize_string(reminder.get("user_id")) != user_id:
            continue
        if _normalize_string(reminder.get("medication_id")) != medication_id:
            continue
        if _normalize_string(reminder.get("status")) != "scheduled":
            continue
        if _values_equivalent(reminder.get("scheduled_time"), scheduled_time):
            return True
    return False


def _forbidden_matches_step(forbidden: str, actual_step: dict[str, Any], initial_state: dict[str, Any]) -> bool:
    tool_name = _normalize_string(actual_step.get("tool_name")).lower()
    readable_tool_name = tool_name.replace("_", " ")
    text = _normalize_string(forbidden).lower()
    if not text:
        return False

    prefix = ""
    if text.startswith(tool_name):
        prefix = tool_name
    elif text.startswith(readable_tool_name):
        prefix = readable_tool_name
    else:
        return False

    suffix = text[len(prefix):].strip()
    if not suffix:
        return True
    if suffix.startswith("on "):
        return _step_matches_exact_forbidden_target(actual_step, suffix[3:])
    if suffix == "with an inactive medication":
        return _step_uses_inactive_medication(actual_step, initial_state)
    if suffix == "at a conflicting scheduled_time":
        return _step_has_conflicting_schedule(actual_step, initial_state)
    return False


def _evaluate_forbidden_actions(
    sample: dict[str, Any],
    actual_tool_trace: list[dict[str, Any]],
    initial_state: dict[str, Any],
) -> list[str]:
    violations: list[str] = []
    forbidden_actions = sample.get("forbidden_actions", []) if isinstance(sample.get("forbidden_actions"), list) else []

    for forbidden in forbidden_actions:
        for actual_step in actual_tool_trace:
            if not actual_step.get("is_write"):
                continue
            if _forbidden_matches_step(forbidden, actual_step, initial_state):
                tool_name = _normalize_string(actual_step.get("tool_name"))
                violations.append(f"forbidden action matched executed write tool '{tool_name}': {forbidden}")
                break
    return violations


def _evaluate_final_state_assertions(
    assertions: list[dict[str, Any]],
    initial_state: dict[str, Any],
    final_state: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for assertion in assertions:
        if not isinstance(assertion, dict):
            continue
        path = _normalize_string(assertion.get("path"))
        assertion_type = _normalize_string(assertion.get("assertion_type"))
        is_hard_requirement = not _is_soft_state_path(path)
        initial_exists, initial_value = _resolve_path(initial_state, path)
        final_exists, final_value = _resolve_path(final_state, path)
        passed = False
        if assertion_type == "changed":
            if is_hard_requirement:
                passed = (
                    initial_exists
                    and final_exists
                    and _values_equivalent(initial_value, assertion.get("old_value"))
                    and _values_equivalent(final_value, assertion.get("new_value"))
                )
            else:
                passed = initial_exists and final_exists and _json_safe_snapshot(initial_value) != _json_safe_snapshot(final_value)
        elif assertion_type == "added":
            expected_value = assertion.get("value")
            passed = (not initial_exists) and final_exists and _semantic_contains(expected_value, final_value, _path_parts(path), ignored_keys=_generated_identifier_keys(expected_value, path))
            if not passed:
                passed, final_value = _find_semantic_added_match(path, expected_value, initial_state, final_state)
                final_exists = final_exists or passed
        elif assertion_type == "removed":
            passed = initial_exists and (not final_exists) and _semantic_contains(assertion.get("value"), initial_value, _path_parts(path))
        results.append(
            {
                "path": path,
                "assertion_type": assertion_type,
                "is_hard_requirement": is_hard_requirement,
                "passed": passed,
                "initial_exists": initial_exists,
                "final_exists": final_exists,
                "initial_value": _json_safe_snapshot(initial_value) if initial_exists else None,
                "final_value": _json_safe_snapshot(final_value) if final_exists else None,
            }
        )
    return results


def evaluate_completion(sample: dict[str, Any], run_result: dict[str, Any]) -> dict[str, Any]:
    gold_action_graph = sample.get("gold_action_graph", {}) if isinstance(sample.get("gold_action_graph"), dict) else {}
    actual_tool_trace = run_result.get("actual_tool_trace", []) if isinstance(run_result.get("actual_tool_trace"), list) else []
    actual_actions = run_result.get("actual_actions", []) if isinstance(run_result.get("actual_actions"), list) else []
    initial_state = run_result.get("initial_state", {}) if isinstance(run_result.get("initial_state"), dict) else {}
    final_state = run_result.get("final_state", {}) if isinstance(run_result.get("final_state"), dict) else {}
    final_state_diff = run_result.get("final_state_diff", {}) if isinstance(run_result.get("final_state_diff"), dict) else {}
    agent_context_mode = _normalize_string(run_result.get("agent_context_mode"))
    tool_schema_index = _build_tool_schema_index(sample)

    used_actual_positions: set[int] = set()
    required_read_equivalence_sets = gold_action_graph.get("required_read_equivalence_sets", []) if isinstance(gold_action_graph.get("required_read_equivalence_sets"), list) else []
    required_write_nodes = gold_action_graph.get("required_write_nodes", []) if isinstance(gold_action_graph.get("required_write_nodes"), list) else []
    precedence_edges = gold_action_graph.get("precedence_edges", []) if isinstance(gold_action_graph.get("precedence_edges"), list) else []
    clarify_nodes = gold_action_graph.get("clarify_nodes", []) if isinstance(gold_action_graph.get("clarify_nodes"), list) else []
    refuse_nodes = gold_action_graph.get("refuse_nodes", []) if isinstance(gold_action_graph.get("refuse_nodes"), list) else []

    read_results, read_step_positions = _match_read_equivalence_sets(
        required_read_equivalence_sets,
        actual_tool_trace,
        used_actual_positions,
        tool_schema_index,
    )
    write_results, write_step_positions = _match_required_nodes(
        required_write_nodes,
        actual_tool_trace,
        used_actual_positions,
        tool_schema_index,
    )
    write_tool_name_results = _match_required_tool_names(required_write_nodes, actual_tool_trace)

    step_to_actual_position = {}
    step_to_actual_position.update(read_step_positions)
    step_to_actual_position.update(write_step_positions)

    precedence_results = _evaluate_precedence(precedence_edges, step_to_actual_position)

    final_state_assertions = sample.get("final_state_assertions", []) if isinstance(sample.get("final_state_assertions"), list) else []
    final_state_results = _evaluate_final_state_assertions(final_state_assertions, initial_state, final_state)
    expected_state_diff = sample.get("expected_state_diff", {}) if isinstance(sample.get("expected_state_diff"), dict) else {}
    expected_state_diff_passed = True
    if expected_state_diff:
        expected_state_diff_passed = _diff_contains(expected_state_diff, final_state_diff)

    respond_actions = [action for action in actual_actions if action.get("kind") == "respond"]
    write_action_indices = [
        int(action.get("action_index", 0) or 0)
        for action in actual_actions
        if action.get("kind") == "tool" and action.get("is_write")
    ]
    first_write_action_index = min(write_action_indices) if write_action_indices else None
    clarification_satisfied = True
    if clarify_nodes:
        clarification_satisfied = any(
            int(action.get("action_index", 0) or 0) < (first_write_action_index or 10**9)
            for action in respond_actions
        )

    refusal_satisfied = True
    if refuse_nodes:
        if required_write_nodes:
            refusal_satisfied = bool(respond_actions)
        else:
            refusal_satisfied = bool(respond_actions) and not write_action_indices

    forbidden_action_violations = _evaluate_forbidden_actions(sample, actual_tool_trace, initial_state)

    read_passed = all(result.get("matched") for result in read_results)
    write_passed = all(result.get("matched") for result in write_results)
    write_tool_name_passed = all(result.get("matched") for result in write_tool_name_results)
    precedence_passed = all(result.get("satisfied") for result in precedence_results)
    final_state_passed = all(result.get("passed") for result in final_state_results)
    hard_final_state_results = [result for result in final_state_results if result.get("is_hard_requirement", True)]
    hard_final_state_passed = all(result.get("passed") for result in hard_final_state_results)

    hard_failures: list[str] = []
    soft_failures: list[str] = []

    if not read_passed:
        hard_failures.append("missing required read-equivalence matches")
    if not write_passed:
        hard_failures.append("missing required write matches")
    if required_write_nodes and not write_tool_name_passed:
        hard_failures.append("missing required write tool usage")
    if not precedence_passed:
        hard_failures.append("precedence constraints violated")
    if final_state_assertions and not final_state_passed:
        hard_failures.append("final state assertions failed")
    if hard_final_state_results and not hard_final_state_passed:
        hard_failures.append("critical final state assertions failed")
    if not expected_state_diff_passed:
        hard_failures.append("expected final state diff not contained in actual final state diff")
    if not clarification_satisfied:
        hard_failures.append("clarification requirement not satisfied before write")
    if not refusal_satisfied:
        hard_failures.append("refusal/scope requirement not satisfied")
    if forbidden_action_violations:
        hard_failures.extend(forbidden_action_violations)

    failures = hard_failures + soft_failures
    semantic_success = not hard_failures
    strict_success = not failures
    return {
        "passed": semantic_success,
        "success": semantic_success,
        "sr": 1.0 if semantic_success else 0.0,
        "semantic_success": semantic_success,
        "semantic_sr": 1.0 if semantic_success else 0.0,
        "strict_success": strict_success,
        "strict_sr": 1.0 if strict_success else 0.0,
        "agent_context_mode": agent_context_mode,
        "task_type": _normalize_string(sample.get("task_type"), gold_action_graph.get("task_type", "execute")),
        "required_read_match_count": sum(1 for result in read_results if result.get("matched")),
        "required_read_total": len(read_results),
        "required_write_match_count": sum(1 for result in write_results if result.get("matched")),
        "required_write_total": len(write_results),
        "required_write_tool_name_match_count": sum(1 for result in write_tool_name_results if result.get("matched")),
        "required_write_tool_name_total": len(write_tool_name_results),
        "precedence_satisfied_count": sum(1 for result in precedence_results if result.get("satisfied")),
        "precedence_total": len(precedence_results),
        "final_state_assertion_pass_count": sum(1 for result in final_state_results if result.get("passed")),
        "final_state_assertion_total": len(final_state_results),
        "critical_final_state_assertion_pass_count": sum(1 for result in hard_final_state_results if result.get("passed")),
        "critical_final_state_assertion_total": len(hard_final_state_results),
        "expected_state_diff_passed": expected_state_diff_passed,
        "clarification_satisfied": clarification_satisfied,
        "refusal_satisfied": refusal_satisfied,
        "forbidden_action_violations": forbidden_action_violations,
        "read_results": read_results,
        "write_results": write_results,
        "write_tool_name_results": write_tool_name_results,
        "precedence_results": precedence_results,
        "final_state_results": final_state_results,
        "hard_failures": hard_failures,
        "soft_failures": soft_failures,
        "failures": failures,
    }
