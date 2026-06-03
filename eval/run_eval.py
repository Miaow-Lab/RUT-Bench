import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_builder.common.discoverability import merge_discoverability_tools
from benchmark_builder.common.env_runtime import build_env_runtime
from benchmark_builder.common.serialization import read_json_file, read_jsonl_file, save_json_file
from benchmark_builder.config import (
    DEFAULT_BENCHMARK_FULL,
    DEFAULT_EVAL_AGENT_TEMPERATURE,
    DEFAULT_EVAL_JUDGE_MODEL,
    DEFAULT_EVAL_JUDGE_TEMPERATURE,
    DEFAULT_EVAL_MAX_SAMPLES,
    DEFAULT_EVAL_MAX_STEPS,
    DEFAULT_EVAL_MAX_WORKERS,
    DEFAULT_EVAL_MODEL,
    DEFAULT_EVAL_RESULTS_DIR,
    DEFAULT_EVAL_TIMEOUT,
    DEFAULT_TASK_BLUEPRINTS,
)
from eval.completion import evaluate_completion
from eval.efficiency import evaluate_efficiency
from eval.reliability import evaluate_reliability
from utils.auto_env import get_state_diff
from utils.call_llm import llm_inference_with_tools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and score EnvBuilder benchmark samples.")
    parser.add_argument("--benchmark-file", default=str(DEFAULT_BENCHMARK_FULL), help="Path to benchmark_full.jsonl")
    parser.add_argument("--blueprint-file", default=str(DEFAULT_TASK_BLUEPRINTS), help="Path to task_blueprints.json")
    parser.add_argument(
        "--output-file",
        default=str(DEFAULT_EVAL_RESULTS_DIR / "eval_results.json"),
        help="Path to the aggregated evaluation report JSON",
    )
    parser.add_argument("--model", default=DEFAULT_EVAL_MODEL, help="Agent model used for benchmark execution")
    parser.add_argument("--temperature", type=float, default=DEFAULT_EVAL_AGENT_TEMPERATURE, help="Agent sampling temperature")
    parser.add_argument("--judge-model", default=DEFAULT_EVAL_JUDGE_MODEL, help="Reliability judge model")
    parser.add_argument("--judge-temperature", type=float, default=DEFAULT_EVAL_JUDGE_TEMPERATURE, help="Judge sampling temperature")
    parser.add_argument("--timeout", type=int, default=DEFAULT_EVAL_TIMEOUT, help="Timeout in seconds for agent and judge calls")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_EVAL_MAX_SAMPLES, help="Optional sample limit for smoke evaluation. 0 means all samples.")
    parser.add_argument("--variant", choices=["all", "stable", "unstable", "collab", "noncollab"], default="all", help="Optional sample variant filter")
    parser.add_argument("--sample-id", default="", help="Optional single sample_id selector")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_EVAL_MAX_STEPS, help="Max assistant action steps per sample. 0 means use the sample budget.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_EVAL_MAX_WORKERS, help="Concurrent worker count across benchmark samples")
    parser.add_argument("--skip-reliability", action="store_true", help="Skip the LLM reliability judge")
    parser.add_argument("--agent-api-key", default="", help="API key for the agent model (overrides OPENAI_API_KEY / GEMINI_API_KEY)")
    parser.add_argument("--agent-base-url", default="", help="Base URL for the agent model (overrides OPENAI_BASE_URL; ignored for gemini provider)")
    parser.add_argument("--judge-api-key", default="", help="API key for the reliability judge (overrides OPENAI_API_KEY)")
    parser.add_argument("--judge-base-url", default="", help="Base URL for the reliability judge (overrides OPENAI_BASE_URL)")
    parser.add_argument(
        "--agent-provider",
        default="openai",
        choices=["openai", "gemini"],
        help=(
            "API provider for the agent model. "
            "'openai' uses any OpenAI-compatible endpoint (set --agent-base-url for proxies). "
            "'gemini' calls the Google Generative Language REST API directly, "
            "which properly preserves thought_signature for Gemini 2.5+ multi-turn tool calling "
            "(set GEMINI_API_KEY or GOOGLE_API_KEY in .env)."
        ),
    )
    parser.add_argument(
        "--agent-extra-body",
        default="",
        help="JSON string or path to a JSON file with extra body params for the agent model API call (openai provider only)",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=-1,
        help=(
            "Thinking budget for the agent model. "
            "For 'gemini' provider: passed as generationConfig.thinkingConfig.thinkingBudget "
            "(0 disables thinking, removing the thought_signature requirement). "
            "For 'openai' provider: added to extra_body as thinking_config.thinking_budget. "
            "-1 means not set."
        ),
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


def _normalize_variant(value: Any) -> str:
    variant = _normalize_string(value)
    if variant == "collab":
        return "stable"
    if variant == "noncollab":
        return "unstable"
    return variant


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_json_loads(raw_value: Any) -> dict[str, Any]:
    if isinstance(raw_value, dict):
        return raw_value
    if not isinstance(raw_value, str):
        return {}
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _observation_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_for_prompt(value: Any) -> str:
    return json.dumps(_json_safe_snapshot(value), ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _normalized_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_normalize_string(item) for item in value if _normalize_string(item)]


def _assistant_message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return _json_safe_snapshot(message)
    if hasattr(message, "model_dump"):
        return _json_safe_snapshot(message.model_dump())

    payload: dict[str, Any] = {
        "role": getattr(message, "role", "assistant"),
        "content": getattr(message, "content", None),
    }
    tool_calls = getattr(message, "tool_calls", None)
    normalized_tool_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls or []:
        if hasattr(tool_call, "model_dump"):
            normalized_tool_calls.append(_json_safe_snapshot(tool_call.model_dump()))
            continue
        function = getattr(tool_call, "function", None)
        normalized_tool_calls.append(
            {
                "id": getattr(tool_call, "id", None),
                "type": getattr(tool_call, "type", "function"),
                "function": {
                    "name": getattr(function, "name", None),
                    "arguments": getattr(function, "arguments", "{}"),
                },
            }
        )
    if normalized_tool_calls:
        payload["tool_calls"] = normalized_tool_calls
    return payload


def _convert_tools_for_openai(raw_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            normalized.append(_json_safe_snapshot(tool))
            continue
        name = _normalize_string(tool.get("name"))
        if not name:
            continue
        parameters = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {"type": "object", "properties": {}}
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": _normalize_string(tool.get("description")),
                    "parameters": _json_safe_snapshot(parameters),
                },
            }
        )
    return normalized


def _build_runtime_index(payload: Any) -> dict[str, dict[str, Any]]:
    items = payload.get("items", []) if isinstance(payload, dict) else []
    runtime_index: dict[str, dict[str, Any]] = {}
    for state_item in items:
        if not isinstance(state_item, dict):
            continue
        base_entry = {
            "env_id": _normalize_string(state_item.get("env_id")),
            "config_id": _normalize_string(state_item.get("config_id")),
            "environment_summary": _normalize_string(state_item.get("environment_summary")),
            "environment_introduction": _normalize_string(state_item.get("environment_introduction")),
            "env_class_name": _normalize_string(state_item.get("env_class_name")),
            "env_class_code": _normalize_string(state_item.get("env_class_code")),
            "constraints_rules_text": _normalize_string(state_item.get("constraints_rules_text")),
            "tools": _json_safe_snapshot(state_item.get("tools", [])),
        }
        blueprints = state_item.get("task_blueprints", []) if isinstance(state_item.get("task_blueprints"), list) else []
        for blueprint in blueprints:
            if not isinstance(blueprint, dict):
                continue
            task_id = _normalize_string(blueprint.get("task_id"))
            if not task_id:
                continue
            runtime_index[task_id] = _json_safe_snapshot(base_entry)
    return runtime_index


def _select_samples(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    filtered = []
    requested_variant = _normalize_variant(args.variant)
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        if requested_variant != "all" and _normalize_variant(sample.get("variant")) != requested_variant:
            continue
        if args.sample_id and _normalize_string(sample.get("sample_id")) != args.sample_id:
            continue
        filtered.append(sample)
    if args.max_samples > 0:
        filtered = filtered[: args.max_samples]
    return filtered


def _build_agent_system_prompt(sample: dict[str, Any], runtime_entry: dict[str, Any]) -> str:
    lines = [
        "You are a careful task-oriented assistant operating a simulated software environment through tools.",
        "Use tools when needed, stay grounded in tool observations, and avoid inventing hidden state.",
        "You do not have direct access to the environment state; inspect it through tool calls before making unsupported assumptions.",
        "If the request lacks required information, ask a concise clarification question.",
        "If part of the request is impossible, unsupported, or unsafe in this environment, scope or refuse that part instead of guessing.",
        "After completing the task, give a brief user-facing response.",
        "",
        f"Environment summary: {runtime_entry.get('environment_summary', '')}",
        f"Environment introduction: {runtime_entry.get('environment_introduction', '')}",
    ]
    constraints = _normalize_string(runtime_entry.get("constraints_rules_text"))
    if constraints:
        lines.extend(["", "Environment constraints:", constraints])
    return "\n".join(lines).strip()


def _build_enriched_sample(sample: dict[str, Any], runtime_entry: dict[str, Any]) -> dict[str, Any]:
    enriched = _json_safe_snapshot(sample)
    enriched["runtime"] = _json_safe_snapshot(runtime_entry)
    return enriched


def _compute_relaxed_completion(completion: dict[str, Any]) -> dict[str, Any]:
    """Derive relaxed success from a completion dict returned by evaluate_completion."""
    final_state_total = int(completion.get("final_state_assertion_total", 0) or 0)
    final_state_pass_count = int(completion.get("final_state_assertion_pass_count", 0) or 0)
    final_state_passed = final_state_total == 0 or final_state_pass_count == final_state_total
    expected_state_diff_passed = bool(completion.get("expected_state_diff_passed", True))

    forbidden_action_violations = completion.get("forbidden_action_violations", [])
    if not isinstance(forbidden_action_violations, list):
        forbidden_action_violations = []
    forbidden_actions_passed = not forbidden_action_violations

    relaxed_failures: list[str] = []
    if not final_state_passed:
        relaxed_failures.append("final_state_assertions_failed")
    if not expected_state_diff_passed:
        relaxed_failures.append("expected_state_diff_failed")
    if not forbidden_actions_passed:
        relaxed_failures.append("forbidden_actions_violated")

    relaxed_success = not relaxed_failures
    original_success = bool(completion.get("semantic_success", completion.get("success", completion.get("passed", False))))

    return {
        "original_success": original_success,
        "relaxed_success": relaxed_success,
        "relaxed_sr": 1.0 if relaxed_success else 0.0,
        "recovered_by_relaxed": relaxed_success and not original_success,
        "final_state_assertions_passed": final_state_passed,
        "final_state_assertion_total": final_state_total,
        "final_state_assertion_pass_count": final_state_pass_count,
        "expected_state_diff_passed": expected_state_diff_passed,
        "forbidden_actions_passed": forbidden_actions_passed,
        "forbidden_action_violations": forbidden_action_violations,
        "original_hard_failures": completion.get("hard_failures", []),
        "relaxed_failures": relaxed_failures,
    }


def _run_single_sample(
    sample: dict[str, Any],
    args: argparse.Namespace,
    agent_extra_body: dict | None = None,
    thinking_budget: int | None = None,
) -> dict[str, Any]:
    runtime_entry = sample["runtime"]
    env_item = {
        "env_class_code": runtime_entry["env_class_code"],
        "env_class_name": runtime_entry["env_class_name"],
    }
    max_steps = args.max_steps if args.max_steps > 0 else _safe_int(sample.get("max_allowed_action_turns"), DEFAULT_EVAL_MAX_STEPS)
    env = build_env_runtime(env_item, max_steps=max_steps)
    env.env_init(init_config=_json_safe_snapshot(sample.get("init_config", {})))
    initial_state = _json_safe_snapshot(env.get_state_info())
    tools = _convert_tools_for_openai(merge_discoverability_tools(runtime_entry.get("tools", [])))

    user_turns = sample.get("user_turns", []) if isinstance(sample.get("user_turns"), list) else []
    transcript: list[dict[str, Any]] = []
    messages = [{"role": "system", "content": _build_agent_system_prompt(sample, runtime_entry)}]
    if user_turns:
        first_turn = _normalize_string(user_turns[0].get("user_message"))
    else:
        first_turn = _normalize_string(sample.get("user_request"))
    messages.append({"role": "user", "content": first_turn})
    transcript.append({"role": "user", "content": first_turn})
    next_user_turn_index = 1

    actual_actions: list[dict[str, Any]] = []
    actual_tool_trace: list[dict[str, Any]] = []
    error_message = None

    for action_index in range(1, max_steps + 1):
        assistant_message = llm_inference_with_tools(
            provider=args.agent_provider,
            model=args.model,
            messages=messages,
            tools=tools,
            temperature=args.temperature,
            timeout=args.timeout,
            api_key=args.agent_api_key or None,
            base_url=args.agent_base_url or None,
            extra_body=agent_extra_body,
            thinking_budget=thinking_budget,
        )
        if assistant_message is None:
            error_message = "agent returned no assistant message"
            break

        assistant_payload = _assistant_message_to_dict(assistant_message)
        tool_calls = assistant_payload.get("tool_calls", []) if isinstance(assistant_payload.get("tool_calls"), list) else []

        if tool_calls:
            tool_response_messages: list[dict[str, Any]] = []
            terminated = truncated = False
            for tc_index, tool_call in enumerate(tool_calls):
                function_block = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                tool_name = _normalize_string(function_block.get("name"))
                raw_arguments = function_block.get("arguments", "{}")
                arguments = _safe_json_loads(raw_arguments)

                observation, reward, terminated, truncated, info = env.env_step({"name": tool_name, "params": arguments})
                step_record = env.trajectory[-1] if env.trajectory else {}
                state_diff = step_record.get("state_diff", {}) if isinstance(step_record, dict) and isinstance(step_record.get("state_diff"), dict) else {}
                observation_text = _observation_to_text(observation)

                tool_response_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", f"tool_call_{action_index}_{tc_index}"),
                        "name": tool_name,
                        "content": observation_text,
                    }
                )

                action_record = {
                    "action_index": action_index,
                    "kind": "tool",
                    "tool_name": tool_name,
                    "arguments": _json_safe_snapshot(arguments),
                    "observation": _json_safe_snapshot(observation),
                    "state_diff": _json_safe_snapshot(state_diff),
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": _json_safe_snapshot(info),
                    "is_write": bool(state_diff),
                }
                actual_actions.append(action_record)
                actual_tool_trace.append(action_record)

                if terminated or truncated:
                    break

            messages.append(assistant_payload)
            messages.extend(tool_response_messages)

            if terminated or truncated:
                break
            continue

        assistant_text = _normalize_string(assistant_payload.get("content"))
        if not assistant_text:
            error_message = "assistant produced neither tool call nor text response"
            break

        messages.append({"role": "assistant", "content": assistant_text})
        transcript.append({"role": "assistant", "content": assistant_text})
        actual_actions.append({"action_index": action_index, "kind": "respond", "content": assistant_text})

        if next_user_turn_index < len(user_turns):
            next_user_turn = _normalize_string(user_turns[next_user_turn_index].get("user_message"))
            next_user_turn_index += 1
            messages.append({"role": "user", "content": next_user_turn})
            transcript.append({"role": "user", "content": next_user_turn})
            continue
        break

    final_state = _json_safe_snapshot(env.get_state_info())
    final_state_diff = _json_safe_snapshot(get_state_diff(initial_state, final_state))

    return {
        "sample_id": sample.get("sample_id"),
        "agent_context_mode": "tool_only",
        "transcript": transcript,
        "actual_actions": actual_actions,
        "actual_tool_trace": actual_tool_trace,
        "initial_state": initial_state,
        "final_state": final_state,
        "final_state_diff": final_state_diff,
        "constraints_rules_text": runtime_entry.get("constraints_rules_text", ""),
        "run_error": error_message,
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _format_duration(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds_part:02d}"
    return f"{minutes:02d}:{seconds_part:02d}"


def _as_completed_with_progress(futures: dict[Any, int]) -> Any:
    total = len(futures)
    if total <= 0:
        return

    if tqdm is not None:
        yield from tqdm(
            as_completed(futures),
            total=total,
            desc="Evaluating samples",
            unit="sample",
            dynamic_ncols=True,
        )
        return

    start_time = time.monotonic()
    print("Evaluating samples: 0/%d [elapsed 00:00, ETA --:--]" % total, file=sys.stderr, flush=True)
    completed = 0
    for future in as_completed(futures):
        completed += 1
        elapsed = max(time.monotonic() - start_time, 1e-9)
        rate = completed / elapsed if completed else 0.0
        eta = (total - completed) / rate if rate > 0 else 0.0
        print(
            "Evaluating samples: "
            f"{completed}/{total} "
            f"[elapsed {_format_duration(elapsed)}, ETA {_format_duration(eta)}]",
            file=sys.stderr,
            flush=True,
        )
        yield future


def _reliability_scores(rows: list[dict[str, Any]], key_name: str) -> list[float]:
    return [float(row.get("reliability", {}).get(key_name, 0.0) or 0.0) for row in rows]


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    # success_rate uses relaxed_success computed inline from each row's completion dict
    relaxed_sr = _mean([
        float(_compute_relaxed_completion(row.get("completion", {})).get("relaxed_success", False))
        for row in rows
    ])
    strict_success_rate = _mean(
        [
            float(
                row.get("completion", {}).get(
                    "strict_sr",
                    float(bool(row.get("completion", {}).get("strict_success", False))),
                )
            )
            for row in rows
        ]
    )
    avg_faithfulness_score = _mean(_reliability_scores(rows, "faithfulness_score"))
    avg_clarification_score = _mean(_reliability_scores(rows, "clarification_score"))
    avg_tool_discipline_score = _mean(_reliability_scores(rows, "tool_discipline_score"))
    return {
        "success_rate": relaxed_sr,
        "sr": relaxed_sr,
        "completion_rate": relaxed_sr,
        "semantic_success_rate": relaxed_sr,
        "strict_success_rate": strict_success_rate,
        "avg_efficiency": _mean([float(row.get("efficiency", {}).get("efficiency", 0.0) or 0.0) for row in rows]),
        "avg_task_score": _mean([float(row.get("efficiency", {}).get("task_score", 0.0) or 0.0) for row in rows]),
        "avg_reliability": _mean(_reliability_scores(rows, "reliability")),
        "avg_faithfulness_score": avg_faithfulness_score,
        "avg_clarification_score": avg_clarification_score,
        "avg_tool_discipline_score": avg_tool_discipline_score,
        "avg_faithfulness_norm": avg_faithfulness_score / 4.0 if rows else 0.0,
        "avg_clarification_norm": avg_clarification_score / 3.0 if rows else 0.0,
        "avg_tool_discipline_norm": avg_tool_discipline_score / 3.0 if rows else 0.0,
    }


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(rows),
        **_metric_summary(rows),
        "run_error_count": sum(1 for row in rows if row.get("run", {}).get("run_error")),
    }


def _group_rows(rows: list[dict[str, Any]], key_name: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _normalize_string(row.get(key_name), "unknown")
        grouped[key].append(row)
    return {key: _summarize_group(value) for key, value in sorted(grouped.items())}


def _group_flat_labels(rows: list[dict[str, Any]], key_name: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        labels = row.get(key_name, []) if isinstance(row.get(key_name), list) else []
        for label in labels:
            normalized = _normalize_string(label)
            if normalized:
                grouped[normalized].append(row)
    return {key: _summarize_group(value) for key, value in sorted(grouped.items())}


def _evaluate_single_sample(
    index: int,
    sample: dict[str, Any],
    runtime_index: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    agent_extra_body: dict | None = None,
    thinking_budget: int | None = None,
) -> tuple[int, dict[str, Any] | None, str | None]:
    task_id = _normalize_string(sample.get("task_id"))
    runtime_entry = runtime_index.get(task_id)
    if runtime_entry is None:
        return index, None, task_id

    enriched_sample = _build_enriched_sample(sample, runtime_entry)
    try:
        run_result = _run_single_sample(
            enriched_sample, args,
            agent_extra_body=agent_extra_body,
            thinking_budget=thinking_budget,
        )
        completion = evaluate_completion(enriched_sample, run_result)
        efficiency = evaluate_efficiency(enriched_sample, run_result, completion.get("passed", False))
        reliability = (
            {
                "judge_model": None,
                "faithfulness_score": 0,
                "clarification_score": 0,
                "tool_discipline_score": 0,
                "faithfulness_reason": "",
                "clarification_reason": "",
                "tool_discipline_reason": "",
                "reliability": 0.0,
                "judge_error": "skipped",
            }
            if args.skip_reliability
            else evaluate_reliability(
                enriched_sample,
                run_result,
                model=args.judge_model,
                temperature=args.judge_temperature,
                timeout=args.timeout,
                api_key=args.judge_api_key or None,
                base_url=args.judge_base_url or None,
            )
        )
    except Exception as exc:
        run_result = {
            "sample_id": enriched_sample.get("sample_id"),
            "transcript": [],
            "actual_actions": [],
            "actual_tool_trace": [],
            "initial_state": {},
            "final_state": {},
            "final_state_diff": {},
            "constraints_rules_text": runtime_entry.get("constraints_rules_text", ""),
            "run_error": f"{exc}\n{traceback.format_exc()}",
        }
        completion = evaluate_completion(enriched_sample, run_result)
        efficiency = evaluate_efficiency(enriched_sample, run_result, completion.get("passed", False))
        reliability = {
            "judge_model": args.judge_model,
            "faithfulness_score": 0,
            "clarification_score": 0,
            "tool_discipline_score": 0,
            "faithfulness_reason": "",
            "clarification_reason": "",
            "tool_discipline_reason": "",
            "reliability": 0.0,
            "judge_error": "run failed before reliability scoring",
        }

    user_turns = enriched_sample.get("user_turns") or []
    num_user_turns = len(user_turns) if isinstance(user_turns, list) else 0
    min_required_action_turn = _safe_int(enriched_sample.get("min_required_action_turn"), 0)

    return index, {
        "sample_id": enriched_sample.get("sample_id"),
        "task_id": enriched_sample.get("task_id"),
        "env_id": enriched_sample.get("env_id"),
        "variant": _normalize_variant(enriched_sample.get("variant")),
        "difficulty_bucket": enriched_sample.get("difficulty_bucket"),
        "task_type": enriched_sample.get("task_type"),
        "num_user_turns": num_user_turns,
        "min_required_action_turn": min_required_action_turn,
        "profile_bundle": enriched_sample.get("profile_bundle", []),
        "behavior_bundle": enriched_sample.get("behavior_bundle", []),
        "unstable_behavior": enriched_sample.get("unstable_behavior", ""),
        "run": _json_safe_snapshot(run_result),
        "completion": completion,
        "efficiency": efficiency,
        "reliability": reliability,
    }, None


def main() -> int:
    args = parse_args()
    agent_extra_body: dict | None = None
    if args.agent_extra_body:
        raw = args.agent_extra_body.strip()
        if Path(raw).is_file():
            raw = Path(raw).read_text(encoding="utf-8")
        try:
            agent_extra_body = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"[ERROR] --agent-extra-body is not valid JSON: {exc}\nReceived: {raw!r}", file=sys.stderr)
            return 1
    thinking_budget: int | None = args.thinking_budget if args.thinking_budget >= 0 else None
    if thinking_budget is not None and args.agent_provider == "openai":
        agent_extra_body = agent_extra_body or {}
        agent_extra_body["thinking_config"] = {"thinking_budget": thinking_budget}

    benchmark_rows = read_jsonl_file(args.benchmark_file)
    selected_samples = _select_samples(benchmark_rows, args)
    blueprint_payload = read_json_file(args.blueprint_file)
    runtime_index = _build_runtime_index(blueprint_payload)

    per_sample_results: list[dict[str, Any]] = []
    missing_runtime_task_ids: list[str] = []

    ordered_results: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {
            executor.submit(
                _evaluate_single_sample, index, sample, runtime_index, args,
                agent_extra_body, thinking_budget,
            ): index
            for index, sample in enumerate(selected_samples)
        }
        for future in _as_completed_with_progress(futures):
            _, result_row, missing_task_id = future.result()
            if missing_task_id:
                missing_runtime_task_ids.append(missing_task_id)
                continue
            if isinstance(result_row, dict):
                ordered_results[futures[future]] = result_row

    for index in range(len(selected_samples)):
        result_row = ordered_results.get(index)
        if isinstance(result_row, dict):
            per_sample_results.append(result_row)

    summary = {
        "sample_count": len(per_sample_results),
        "missing_runtime_task_ids": missing_runtime_task_ids,
        **_metric_summary(per_sample_results),
        "variant_summary": _group_rows(per_sample_results, "variant"),
        "difficulty_summary": _group_rows(per_sample_results, "difficulty_bucket"),
        "task_type_summary": _group_rows(per_sample_results, "task_type"),
        "num_user_turns_summary": _group_rows(per_sample_results, "num_user_turns"),
        "min_required_action_turn_summary": _group_rows(per_sample_results, "min_required_action_turn"),
        "behavior_summary": _group_flat_labels(per_sample_results, "behavior_bundle"),
        "unstable_behavior_summary": _group_rows(per_sample_results, "unstable_behavior"),
        "profile_summary": _group_flat_labels(per_sample_results, "profile_bundle"),
    }

    output_payload = {
        "metadata": {
            "generated_at": _now_iso(),
            "benchmark_file": str(args.benchmark_file),
            "blueprint_file": str(args.blueprint_file),
            "model": args.model,
            "agent_provider": args.agent_provider,
            "agent_base_url": args.agent_base_url or None,
            "thinking_budget": thinking_budget,
            "temperature": args.temperature,
            "judge_model": None if args.skip_reliability else args.judge_model,
            "judge_base_url": (args.judge_base_url or None) if not args.skip_reliability else None,
            "judge_temperature": args.judge_temperature,
            "variant_filter": _normalize_variant(args.variant),
            "sample_id_filter": args.sample_id,
            "max_samples": args.max_samples,
            "max_steps": args.max_steps,
            "max_workers": args.max_workers,
            "skip_reliability": args.skip_reliability,
        },
        "summary": summary,
        "samples": per_sample_results,
    }
    save_json_file(args.output_file, output_payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
