import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_builder.common.env_runtime import stable_json_dumps, validate_init_config
from benchmark_builder.common.llm_io import call_json_object_llm, unwrap_init_config
from benchmark_builder.common.serialization import read_json_file, save_json_file
from benchmark_builder.config import (
    DEFAULT_ENV_MANIFEST,
    DEFAULT_INIT_STATE_BANK,
    DEFAULT_STAGE1_CONFIGS_PER_ENV,
    DEFAULT_STAGE1_ENV_MAX_STEPS,
    DEFAULT_STAGE1_MAX_ATTEMPTS_PER_CONFIG,
    DEFAULT_STAGE1_MAX_ENVS,
    DEFAULT_STAGE1_MAX_WORKERS,
    DEFAULT_STAGE1_MODEL,
    DEFAULT_STAGE1_TEMPERATURE,
)


TARGET_STATE_PROFILES = ["sparse", "balanced", "crowded"]


SYSTEM_PROMPT = """
You are an expert environment state initializer for an agent tool-calling benchmark.

Return one valid JSON object that can be passed directly to env.env_init(init_config=...).

Requirements:
1. Use the exact top-level state container names provided by the user.
2. Populate realistic, internally consistent, cross-referenced entities.
3. Make the state rich enough for multi-step tool use.
4. Use only fictional data.
5. Output JSON only.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a reusable init state bank from the normalized env manifest.")
    parser.add_argument("--input-file", default=str(DEFAULT_ENV_MANIFEST), help="Path to env_manifest.json")
    parser.add_argument("--output-file", default=str(DEFAULT_INIT_STATE_BANK), help="Path to init_state_bank.json")
    parser.add_argument("--model", default=DEFAULT_STAGE1_MODEL, help="LLM model used for init state generation")
    parser.add_argument("--temperature", type=float, default=DEFAULT_STAGE1_TEMPERATURE, help="LLM temperature")
    parser.add_argument("--configs-per-env", type=int, default=DEFAULT_STAGE1_CONFIGS_PER_ENV, help="Accepted init configs per env")
    parser.add_argument("--max-attempts-per-config", type=int, default=DEFAULT_STAGE1_MAX_ATTEMPTS_PER_CONFIG, help="LLM attempts per target config")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_STAGE1_MAX_WORKERS, help="Worker count across environments")
    parser.add_argument("--max-envs", type=int, default=DEFAULT_STAGE1_MAX_ENVS, help="Optional limit for smoke tests. 0 means all envs.")
    parser.add_argument("--env-max-steps", type=int, default=DEFAULT_STAGE1_ENV_MAX_STEPS, help="Max env steps used during init validation")
    parser.add_argument("--timeout", type=int, default=120, help="LLM timeout in seconds")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key override for Stage1")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"), help="OpenAI base URL override for Stage1")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing output file")
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


def _print_stage1_progress(
    *,
    completed_count: int,
    total_count: int,
    start_time: float,
    accepted_state_count: int,
    failure_count: int,
) -> None:
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
        f"[Stage1] [{bar}] {completed_count}/{total_count} "
        f"states={accepted_state_count} failures={failure_count} "
        f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta_seconds)} "
        f"rate={rate:.2f}/env"
    )

    if sys.stdout.isatty():
        end = "\n" if completed_count >= total_count else ""
        print(f"\r{line}", end=end, flush=True)
    else:
        print(line, flush=True)


def _manifest_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return [item for item in payload["items"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise TypeError("Manifest must be a list or a dict with an items array")


def _state_container_prompt_text(env_item: dict[str, Any]) -> str:
    states = (env_item.get("env_structure") or {}).get("states", {})
    lines = []
    for name in env_item.get("state_container_names", []):
        details = states.get(name, {}) if isinstance(states, dict) else {}
        if isinstance(details, dict):
            type_name = details.get("type", "unknown")
            used_by = len(details.get("used_by", [])) if isinstance(details.get("used_by"), list) else 0
            modified_by = len(details.get("modified_by", [])) if isinstance(details.get("modified_by"), list) else 0
            lines.append(f"- {name}: {type_name} | used_by={used_by} | modified_by={modified_by}")
        else:
            lines.append(f"- {name}: {details}")
    return "\n".join(lines)


def _profile_instructions(target_profile: str) -> str:
    if target_profile == "sparse":
        return "Generate a small but still useful state: at least two populated collections, but keep most collections lean."
    if target_profile == "crowded":
        return "Generate a dense state: multiple populated collections with several entries and realistic cross-references."
    return "Generate a balanced state: multiple populated collections with moderate variety and enough data for several tasks."


def _build_user_prompt(env_item: dict[str, Any], target_profile: str, failure_reasons: list[str]) -> str:
    failure_block = "\n".join(f"- {reason}" for reason in failure_reasons[-3:]) or "- none"
    return f"""
Environment ID: {env_item.get('env_id', '')}
Environment summary:
{env_item.get('environment_summary', '')}

Environment introduction:
{env_item.get('environment_introduction', '')}

State containers to populate:
{_state_container_prompt_text(env_item)}

Constraint rules:
{env_item.get('constraints_rules_text', '')}

Available tools summary:
{env_item.get('tool_summary_text', '')}

Relevant code excerpt:
```python
{env_item.get('promptable_env_code', '')}
```

Target state profile:
{target_profile}

Profile guidance:
{_profile_instructions(target_profile)}

Recent rejected attempts:
{failure_block}

Return one JSON object that can be used directly as init_config.
""".strip()


def _dedupe_existing_state_bank(state_bank: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for item in state_bank:
        init_config = item.get("init_config")
        if isinstance(init_config, dict):
            deduped[stable_json_dumps(init_config)] = item
    return list(deduped.values())


def _load_existing_output(output_file: str, use_resume: bool) -> dict[str, dict[str, Any]]:
    if not use_resume:
        return {}
    output_path = Path(output_file)
    if not output_path.exists():
        return {}
    payload = read_json_file(output_path)
    items = payload.get("items", []) if isinstance(payload, dict) else []
    existing_map: dict[str, dict[str, Any]] = {}
    for item in items:
        if isinstance(item, dict) and item.get("env_id"):
            item["state_bank"] = _dedupe_existing_state_bank(item.get("state_bank", []))
            existing_map[str(item["env_id"])] = item
    return existing_map


def _copy_manifest_fields(env_item: dict[str, Any]) -> dict[str, Any]:
    return {
        "env_id": env_item.get("env_id", ""),
        "environment_summary": env_item.get("environment_summary", ""),
        "environment_introduction": env_item.get("environment_introduction", ""),
        "env_class_name": env_item.get("env_class_name", ""),
        "env_class_def": env_item.get("env_class_def", ""),
        "env_class_code": env_item.get("env_class_code", ""),
        "promptable_env_code": env_item.get("promptable_env_code", ""),
        "constraints_rules_list": env_item.get("constraints_rules_list", []),
        "constraints_rules_text": env_item.get("constraints_rules_text", ""),
        "tools": env_item.get("tools", []),
        "tool_summary": env_item.get("tool_summary", []),
        "tool_summary_text": env_item.get("tool_summary_text", ""),
        "env_structure": env_item.get("env_structure", {}),
        "state_container_names": env_item.get("state_container_names", []),
        "method_names": env_item.get("method_names", []),
        "complexity": env_item.get("complexity", {}),
        "state_bank": [],
        "state_generation_failures": [],
    }


def _process_env_item(env_item: dict[str, Any], args: argparse.Namespace, existing_item: dict[str, Any] | None) -> dict[str, Any]:
    result = _copy_manifest_fields(env_item)
    if existing_item:
        result["state_bank"] = list(existing_item.get("state_bank", []))
        result["state_generation_failures"] = list(existing_item.get("state_generation_failures", []))

    seen_configs = {stable_json_dumps(item["init_config"]) for item in result["state_bank"] if isinstance(item.get("init_config"), dict)}
    failure_reasons = [str(item.get("reason", "")) for item in result["state_generation_failures"] if isinstance(item, dict)]

    while len(result["state_bank"]) < args.configs_per_env:
        target_index = len(result["state_bank"])
        target_profile = TARGET_STATE_PROFILES[target_index % len(TARGET_STATE_PROFILES)]
        accepted = False

        for attempt in range(1, args.max_attempts_per_config + 1):
            try:
                payload = call_json_object_llm(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _build_user_prompt(env_item, target_profile, failure_reasons)},
                    ],
                    temperature=args.temperature,
                    timeout=args.timeout,
                    api_key=getattr(args, "api_key", None),
                    base_url=getattr(args, "base_url", None),
                )
                init_config = unwrap_init_config(payload)
                if not isinstance(init_config, dict):
                    reason = "LLM response did not contain a JSON object init_config"
                    failure_reasons.append(reason)
                    result["state_generation_failures"].append(
                        {"target_profile": target_profile, "attempt": attempt, "reason": reason}
                    )
                    continue

                fingerprint = stable_json_dumps(init_config)
                if fingerprint in seen_configs:
                    reason = "duplicate init_config candidate"
                    failure_reasons.append(reason)
                    result["state_generation_failures"].append(
                        {"target_profile": target_profile, "attempt": attempt, "reason": reason}
                    )
                    continue

                validation = validate_init_config(env_item, init_config, max_steps=args.env_max_steps)
                if not validation.get("is_valid"):
                    reason = str(validation.get("reason", "init_config validation failed"))
                    failure_reasons.append(reason)
                    result["state_generation_failures"].append(
                        {
                            "target_profile": target_profile,
                            "attempt": attempt,
                            "reason": reason,
                            "validation": validation,
                        }
                    )
                    continue

                config_id = f"{result['env_id']}_cfg_{len(result['state_bank']) + 1}"
                result["state_bank"].append(
                    {
                        "config_id": config_id,
                        "target_profile": target_profile,
                        "state_profile": validation["state_stats"].get("state_profile", "unknown"),
                        "init_config": init_config,
                        "validation": validation,
                        "llm_model": args.model,
                        "llm_temperature": args.temperature,
                        "generated_at": _now_iso(),
                    }
                )
                seen_configs.add(fingerprint)
                accepted = True
                break
            except Exception as exc:
                reason = f"exception during generation: {exc}"
                failure_reasons.append(reason)
                result["state_generation_failures"].append(
                    {
                        "target_profile": target_profile,
                        "attempt": attempt,
                        "reason": reason,
                        "traceback": traceback.format_exc(limit=3),
                    }
                )

        if not accepted:
            break

    return result


def _build_output_payload(items: list[dict[str, Any]], args: argparse.Namespace, input_file: str) -> dict[str, Any]:
    total_states = sum(len(item.get("state_bank", [])) for item in items)
    failed_envs = sum(1 for item in items if len(item.get("state_bank", [])) < args.configs_per_env)
    total_failures = sum(len(item.get("state_generation_failures", [])) for item in items)
    return {
        "metadata": {
            "generated_at": _now_iso(),
            "source_file": str(Path(input_file).resolve()),
            "model": args.model,
            "temperature": args.temperature,
            "configs_per_env": args.configs_per_env,
            "max_attempts_per_config": args.max_attempts_per_config,
            "env_max_steps": args.env_max_steps,
            "env_count": len(items),
            "accepted_state_count": total_states,
            "failed_env_count": failed_envs,
            "generation_failure_count": total_failures,
        },
        "items": items,
    }


def main() -> None:
    args = parse_args()
    manifest_payload = read_json_file(args.input_file)
    manifest_items = _manifest_items(manifest_payload)
    if args.max_envs > 0:
        manifest_items = manifest_items[: args.max_envs]

    existing_map = _load_existing_output(args.output_file, use_resume=not args.no_resume)
    results_by_env: dict[str, dict[str, Any]] = {}
    total_count = len(manifest_items)
    start_time = time.monotonic()

    if total_count > 0:
        _print_stage1_progress(
            completed_count=0,
            total_count=total_count,
            start_time=start_time,
            accepted_state_count=0,
            failure_count=0,
        )

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {
            executor.submit(_process_env_item, item, args, existing_map.get(str(item.get("env_id", "")))): item
            for item in manifest_items
        }
        for index, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            env_id = str(item.get("env_id", ""))
            result = future.result()
            results_by_env[env_id] = result

            accepted_state_count = sum(
                len(entry.get("state_bank", []))
                for entry in results_by_env.values()
                if isinstance(entry, dict)
            )
            failure_count = sum(
                len(entry.get("state_generation_failures", []))
                for entry in results_by_env.values()
                if isinstance(entry, dict)
            )
            _print_stage1_progress(
                completed_count=index,
                total_count=total_count,
                start_time=start_time,
                accepted_state_count=accepted_state_count,
                failure_count=failure_count,
            )

            if index % 2 == 0:
                ordered_items = [results_by_env.get(str(manifest_item.get("env_id", "")), existing_map.get(str(manifest_item.get("env_id", "")))) for manifest_item in manifest_items]
                ordered_items = [entry for entry in ordered_items if isinstance(entry, dict)]
                save_json_file(args.output_file, _build_output_payload(ordered_items, args, args.input_file))

    ordered_items = []
    for manifest_item in manifest_items:
        env_id = str(manifest_item.get("env_id", ""))
        if env_id in results_by_env:
            ordered_items.append(results_by_env[env_id])
        elif env_id in existing_map:
            ordered_items.append(existing_map[env_id])

    payload = _build_output_payload(ordered_items, args, args.input_file)
    save_json_file(args.output_file, payload)
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))
    print(f"Saved init state bank for {len(ordered_items)} environments to {args.output_file}")


if __name__ == "__main__":
    main()