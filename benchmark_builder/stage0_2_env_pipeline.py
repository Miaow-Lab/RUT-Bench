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

from benchmark_builder.common.serialization import read_json_file, save_json_file
from benchmark_builder.config import (
    DEFAULT_ENV_MANIFEST,
    DEFAULT_INIT_STATE_BANK,
    DEFAULT_MERGED_ENV_DATA,
    DEFAULT_TASK_BLUEPRINTS,
)
from benchmark_builder.stage0_prepare_env_manifest import _normalize_record, _summarize_manifest
from benchmark_builder.stage1_generate_init_state_bank import _build_output_payload as build_stage1_payload
from benchmark_builder.stage1_generate_init_state_bank import _process_env_item as run_stage1_for_env
from benchmark_builder.stage2_generate_task_blueprints import _parse_required_dialogue_modes
from benchmark_builder.stage2_generate_task_blueprints import _process_env_item as run_stage2_for_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage0-2 as one per-environment pipeline with failure-based filtering and parallel workers."
    )
    parser.add_argument("--input-file", default=str(DEFAULT_MERGED_ENV_DATA), help="Path to source merged_env_data.json")
    parser.add_argument("--output-merged-file", default=str(PROJECT_ROOT / "benchmark_builder" / "output" / "merged_env_data.filtered.json"), help="Filtered merged_env_data output")
    parser.add_argument("--output-manifest-file", default=str(DEFAULT_ENV_MANIFEST), help="Filtered Stage0 env manifest output")
    parser.add_argument("--output-init-state-file", default=str(DEFAULT_INIT_STATE_BANK), help="Filtered Stage1 init_state_bank output")
    parser.add_argument("--output-task-blueprints-file", default=str(DEFAULT_TASK_BLUEPRINTS), help="Filtered Stage2 task_blueprints output")
    parser.add_argument("--output-report-file", default=str(PROJECT_ROOT / "benchmark_builder" / "output" / "stage0_2_pipeline_report.json"), help="Pipeline report output")

    parser.add_argument("--max-envs", type=int, default=0, help="Optional limit for smoke tests. 0 means all envs.")
    parser.add_argument(
        "--max-worker",
        "--max-workers",
        dest="max_worker",
        type=int,
        default=4,
        help="Number of environments to process in parallel",
    )
    parser.add_argument(
        "--failure-policy",
        choices=["strict", "final"],
        default="strict",
        help="strict: any failure log in stage1/stage2 drops the env; final: only final quota/coverage failure drops the env",
    )

    parser.add_argument("--stage1-model", default="gpt-5.4", help="Stage1 LLM model")
    parser.add_argument("--stage1-temperature", type=float, default=0.5, help="Stage1 temperature")
    parser.add_argument("--configs-per-env", type=int, default=2, help="Stage1 accepted init configs per env")
    parser.add_argument("--max-attempts-per-config", type=int, default=4, help="Stage1 attempts per target config")
    parser.add_argument("--stage1-env-max-steps", type=int, default=100, help="Stage1 env max steps")
    parser.add_argument("--stage1-timeout", type=int, default=120, help="Stage1 timeout seconds")
    parser.add_argument(
        "--stage1-api-key",
        default=os.getenv("OPENAI_STAGE01_API_KEY"),
        help="Stage0-1 OpenAI API key (default: OPENAI_STAGE01_API_KEY or OPENAI_API_KEY)",
    )
    print(os.getenv("OPENAI_STAGE01_API_KEY"))
    parser.add_argument(
        "--stage1-base-url",
        default=os.getenv("OPENAI_STAGE01_BASE_URL"),
        help="Stage0-1 OpenAI base URL (default: OPENAI_STAGE01_BASE_URL or OPENAI_BASE_URL)",
    )
    print(os.getenv("OPENAI_STAGE01_BASE_URL"))

    parser.add_argument("--stage2-model", default="gpt-5.4", help="Stage2 LLM model")
    parser.add_argument("--stage2-temperature", type=float, default=0.4, help="Stage2 temperature")
    parser.add_argument("--tasks-per-state", type=int, default=2, help="Stage2 accepted blueprints per state")
    parser.add_argument("--max-attempts-per-task", type=int, default=4, help="Stage2 attempts per accepted blueprint")
    parser.add_argument("--max-states-per-env", type=int, default=2, help="Stage2 max states per env")
    parser.add_argument("--min-blueprints-per-env", type=int, default=4, help="Stage2 minimum blueprints per env")
    parser.add_argument(
        "--required-dialogue-modes",
        default="single_turn,multi_turn",
        help="Stage2 required dialogue modes",
    )
    parser.add_argument("--min-difficulty-buckets", type=int, default=0, help="Stage2 min difficulty bucket count")
    parser.add_argument("--stage2-env-max-steps", type=int, default=100, help="Stage2 env max steps")
    parser.add_argument("--stage2-timeout", type=int, default=120, help="Stage2 timeout seconds")
    parser.add_argument(
        "--stage2-api-key",
        default=os.getenv("OPENAI_STAGE2_API_KEY"),
        help="Stage2 OpenAI API key (default: OPENAI_STAGE2_API_KEY or OPENAI_API_KEY)",
    )
    print(os.getenv("OPENAI_STAGE2_API_KEY"))
    parser.add_argument(
        "--stage2-base-url",
        default=os.getenv("OPENAI_STAGE2_BASE_URL"),
        help="Stage2 OpenAI base URL (default: OPENAI_STAGE2_BASE_URL or OPENAI_BASE_URL)",
    )
    print(os.getenv("OPENAI_STAGE2_BASE_URL"))
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _iter_source_records(payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    passthrough: dict[str, Any] = {}

    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        for index, item in enumerate(payload["items"]):
            if isinstance(item, dict):
                records.append(
                    {
                        "source_type": "dict_items",
                        "source_index": index,
                        "source_key": str(index),
                        "record": item,
                    }
                )
        passthrough = {"source_root_type": "dict_items", "root_payload": payload}
        return records, passthrough

    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                records.append(
                    {
                        "source_type": "dict_mapping",
                        "source_index": len(records),
                        "source_key": str(key),
                        "record": value,
                    }
                )
            else:
                passthrough[str(key)] = value
        passthrough["source_root_type"] = "dict_mapping"
        return records, passthrough

    if isinstance(payload, list):
        for index, item in enumerate(payload):
            if isinstance(item, dict):
                records.append(
                    {
                        "source_type": "list",
                        "source_index": index,
                        "source_key": str(index),
                        "record": item,
                    }
                )
        passthrough = {"source_root_type": "list"}
        return records, passthrough

    raise TypeError(f"Unsupported merged env data type: {type(payload).__name__}")


def _dedupe_by_env_id(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    seen_env_ids: set[str] = set()

    for item in records:
        record = item.get("record", {})
        env_id = str(record.get("env_id", "")).strip() if isinstance(record, dict) else ""
        if not env_id:
            env_id = f"missing_env_id__{item.get('source_key', '')}"
        if env_id in seen_env_ids:
            duplicates.append(
                {
                    "env_id": env_id,
                    "source_key": item.get("source_key", ""),
                    "reason": "duplicate env_id",
                }
            )
            continue
        seen_env_ids.add(env_id)
        item["resolved_env_id"] = env_id
        selected.append(item)

    return selected, duplicates


def _build_stage1_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model=args.stage1_model,
        temperature=args.stage1_temperature,
        configs_per_env=args.configs_per_env,
        max_attempts_per_config=args.max_attempts_per_config,
        env_max_steps=args.stage1_env_max_steps,
        timeout=args.stage1_timeout,
        api_key=args.stage1_api_key,
        base_url=args.stage1_base_url,
    )


def _build_stage2_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model=args.stage2_model,
        temperature=args.stage2_temperature,
        tasks_per_state=args.tasks_per_state,
        max_attempts_per_task=args.max_attempts_per_task,
        max_states_per_env=args.max_states_per_env,
        min_blueprints_per_env=args.min_blueprints_per_env,
        required_dialogue_modes=args.required_dialogue_modes,
        min_difficulty_buckets=args.min_difficulty_buckets,
        env_max_steps=args.stage2_env_max_steps,
        timeout=args.stage2_timeout,
        api_key=args.stage2_api_key,
        base_url=args.stage2_base_url,
    )


def _has_stage1_failure(stage1_item: dict[str, Any], stage1_args: argparse.Namespace, failure_policy: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    state_bank = stage1_item.get("state_bank", []) if isinstance(stage1_item.get("state_bank"), list) else []
    state_generation_failures = (
        stage1_item.get("state_generation_failures", [])
        if isinstance(stage1_item.get("state_generation_failures"), list)
        else []
    )

    if len(state_bank) < int(stage1_args.configs_per_env):
        reasons.append(
            f"state_bank size {len(state_bank)} < configs_per_env {int(stage1_args.configs_per_env)}"
        )

    if failure_policy == "strict" and state_generation_failures:
        reasons.append(f"state_generation_failures={len(state_generation_failures)}")

    return bool(reasons), reasons


def _has_stage2_failure(stage2_env_result: dict[str, Any], stage2_args: argparse.Namespace, failure_policy: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    coverage_summary = (
        stage2_env_result.get("env_coverage_summary", {})
        if isinstance(stage2_env_result.get("env_coverage_summary"), dict)
        else {}
    )
    if not coverage_summary.get("coverage_passed", False):
        reasons.append("env_coverage_summary.coverage_passed is false")

    if int(stage2_args.min_blueprints_per_env) > 0:
        blueprint_count = int(stage2_env_result.get("blueprint_count", 0) or 0)
        if blueprint_count < int(stage2_args.min_blueprints_per_env):
            reasons.append(
                f"blueprint_count {blueprint_count} < min_blueprints_per_env {int(stage2_args.min_blueprints_per_env)}"
            )

    if failure_policy == "strict":
        failure_count = int(stage2_env_result.get("failure_count", 0) or 0)
        if failure_count > 0:
            reasons.append(f"task_generation_failures={failure_count}")

    return bool(reasons), reasons


def _run_env_pipeline(
    source_item: dict[str, Any],
    stage1_args: argparse.Namespace,
    stage2_args: argparse.Namespace,
    failure_policy: str,
) -> dict[str, Any]:
    env_id = str(source_item.get("resolved_env_id", ""))
    raw_record = source_item.get("record", {})
    started_at = time.monotonic()

    try:
        manifest_item = _normalize_record(raw_record)
    except Exception as exc:
        return {
            "env_id": env_id,
            "accepted": False,
            "failed_stage": "stage0",
            "failure_reasons": [f"stage0 normalize exception: {exc}"],
            "traceback": traceback.format_exc(limit=6),
            "elapsed_seconds": round(time.monotonic() - started_at, 2),
        }

    normalized_env_id = str(manifest_item.get("env_id", "")).strip()
    if not normalized_env_id:
        return {
            "env_id": env_id,
            "accepted": False,
            "failed_stage": "stage0",
            "failure_reasons": ["stage0 output missing env_id"],
            "elapsed_seconds": round(time.monotonic() - started_at, 2),
        }

    stage1_item = run_stage1_for_env(manifest_item, stage1_args, None)
    stage1_failed, stage1_reasons = _has_stage1_failure(stage1_item, stage1_args, failure_policy)
    if stage1_failed:
        return {
            "env_id": normalized_env_id,
            "accepted": False,
            "failed_stage": "stage1",
            "failure_reasons": stage1_reasons,
            "stage1_failure_count": len(stage1_item.get("state_generation_failures", [])),
            "elapsed_seconds": round(time.monotonic() - started_at, 2),
        }

    stage2_env_result = run_stage2_for_env(stage1_item, stage2_args)
    stage2_failed, stage2_reasons = _has_stage2_failure(stage2_env_result, stage2_args, failure_policy)
    if stage2_failed:
        return {
            "env_id": normalized_env_id,
            "accepted": False,
            "failed_stage": "stage2",
            "failure_reasons": stage2_reasons,
            "stage2_failure_count": int(stage2_env_result.get("failure_count", 0) or 0),
            "elapsed_seconds": round(time.monotonic() - started_at, 2),
        }

    return {
        "env_id": normalized_env_id,
        "accepted": True,
        "source_item": source_item,
        "manifest_item": manifest_item,
        "stage1_item": stage1_item,
        "stage2_env_result": stage2_env_result,
        "elapsed_seconds": round(time.monotonic() - started_at, 2),
    }


def _rebuild_filtered_merged_payload(
    source_payload: Any,
    accepted_results: list[dict[str, Any]],
    passthrough: dict[str, Any],
) -> Any:
    root_type = passthrough.get("source_root_type")

    if root_type == "dict_items" and isinstance(source_payload, dict):
        output_payload = dict(source_payload)
        accepted_records = [
            item["source_item"]["record"]
            for item in accepted_results
            if isinstance(item.get("source_item"), dict)
        ]
        output_payload["items"] = accepted_records
        return output_payload

    if root_type == "list" and isinstance(source_payload, list):
        ordered = sorted(
            [item for item in accepted_results if isinstance(item.get("source_item"), dict)],
            key=lambda item: int(item["source_item"].get("source_index", 0)),
        )
        return [item["source_item"]["record"] for item in ordered]

    if root_type == "dict_mapping" and isinstance(source_payload, dict):
        accepted_keys = {
            str(item["source_item"].get("source_key", ""))
            for item in accepted_results
            if isinstance(item.get("source_item"), dict)
        }
        output_payload: dict[str, Any] = {}
        for key, value in source_payload.items():
            key_text = str(key)
            if not isinstance(value, dict):
                output_payload[key_text] = value
                continue
            if key_text in accepted_keys:
                output_payload[key_text] = value
        return output_payload

    raise TypeError("Unsupported source payload shape when rebuilding filtered merged payload")


def _build_manifest_payload(items: list[dict[str, Any]], input_file: str) -> dict[str, Any]:
    return {
        "metadata": {
            "generated_at": _now_iso(),
            "source_file": str(Path(input_file).resolve()),
            "statistics": _summarize_manifest(items),
            "pipeline": "stage0_2_env_pipeline",
        },
        "items": items,
    }


def _build_stage2_payload(
    stage2_env_results: list[dict[str, Any]],
    *,
    input_file: str,
    stage2_args: argparse.Namespace,
) -> dict[str, Any]:
    output_items: list[dict[str, Any]] = []
    env_coverage_summary: list[dict[str, Any]] = []
    blueprint_count = 0
    failure_count = 0

    for env_result in stage2_env_results:
        output_items.extend(env_result.get("output_items", []))
        if isinstance(env_result.get("env_coverage_summary"), dict):
            env_coverage_summary.append(env_result["env_coverage_summary"])
        blueprint_count += int(env_result.get("blueprint_count", 0) or 0)
        failure_count += int(env_result.get("failure_count", 0) or 0)

    env_coverage_pass_count = sum(1 for item in env_coverage_summary if item.get("coverage_passed", False))
    env_coverage_failure_count = len(env_coverage_summary) - env_coverage_pass_count

    return {
        "metadata": {
            "generated_at": _now_iso(),
            "source_file": str(Path(input_file).resolve()),
            "model": stage2_args.model,
            "temperature": stage2_args.temperature,
            "tasks_per_state": stage2_args.tasks_per_state,
            "max_attempts_per_task": stage2_args.max_attempts_per_task,
            "max_workers": 1,
            "env_max_steps": stage2_args.env_max_steps,
            "min_blueprints_per_env": stage2_args.min_blueprints_per_env,
            "required_dialogue_modes": _parse_required_dialogue_modes(stage2_args.required_dialogue_modes),
            "min_difficulty_buckets": stage2_args.min_difficulty_buckets,
            "env_count": len(stage2_env_results),
            "state_item_count": len(output_items),
            "blueprint_count": blueprint_count,
            "generation_failure_count": failure_count,
            "env_coverage_pass_count": env_coverage_pass_count,
            "env_coverage_failure_count": env_coverage_failure_count,
        },
        "items": output_items,
        "env_coverage_summary": env_coverage_summary,
    }


def main() -> None:
    args = parse_args()
    source_payload = read_json_file(args.input_file)
    source_records, passthrough = _iter_source_records(source_payload)
    unique_records, duplicate_records = _dedupe_by_env_id(source_records)

    if args.max_envs > 0:
        unique_records = unique_records[: args.max_envs]

    stage1_args = _build_stage1_args(args)
    stage2_args = _build_stage2_args(args)

    total_envs = len(unique_records)
    print(
        "[pipeline stage0-2] "
        f"env_count={total_envs} max_worker={max(1, int(args.max_worker))} "
        f"failure_policy={args.failure_policy} input={args.input_file}",
        flush=True,
    )

    accepted_by_env_id: dict[str, dict[str, Any]] = {}
    dropped: list[dict[str, Any]] = list(duplicate_records)
    accepted_env_count = 0
    accepted_task_blueprint_count = 0
    started_at = time.monotonic()

    with ThreadPoolExecutor(max_workers=max(1, int(args.max_worker))) as executor:
        futures = {
            executor.submit(
                _run_env_pipeline,
                source_item,
                stage1_args,
                stage2_args,
                args.failure_policy,
            ): source_item
            for source_item in unique_records
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            source_item = futures[future]
            source_env_id = str(source_item.get("resolved_env_id", ""))
            try:
                result = future.result()
            except Exception as exc:
                dropped.append(
                    {
                        "env_id": source_env_id,
                        "source_key": source_item.get("source_key", ""),
                        "reason": f"pipeline exception: {exc}",
                        "traceback": traceback.format_exc(limit=6),
                    }
                )
                print(
                    "[pipeline stage0-2][progress] "
                    f"env={source_env_id} passed=no failed_stage=exception "
                    f"accepted_envs={accepted_env_count} accepted_task_blueprints={accepted_task_blueprint_count} "
                    f"completed={completed}/{total_envs}",
                    flush=True,
                )
                continue

            env_id = str(result.get("env_id", source_env_id))
            if result.get("accepted"):
                stage2_env_result = result.get("stage2_env_result", {})
                env_blueprint_count = (
                    int(stage2_env_result.get("blueprint_count", 0) or 0)
                    if isinstance(stage2_env_result, dict)
                    else 0
                )
                accepted_by_env_id[env_id] = result
                accepted_env_count += 1
                accepted_task_blueprint_count += env_blueprint_count
                print(
                    "[pipeline stage0-2][progress] "
                    f"env={env_id} passed=yes failed_stage=none "
                    f"accepted_envs={accepted_env_count} accepted_task_blueprints={accepted_task_blueprint_count} "
                    f"env_task_blueprints={env_blueprint_count} "
                    f"completed={completed}/{total_envs} elapsed={result.get('elapsed_seconds', 0)}s",
                    flush=True,
                )
            else:
                failed_stage = str(result.get("failed_stage", "unknown"))
                dropped.append(
                    {
                        "env_id": env_id,
                        "source_key": source_item.get("source_key", ""),
                        "failed_stage": failed_stage,
                        "failure_reasons": result.get("failure_reasons", []),
                        "elapsed_seconds": result.get("elapsed_seconds", 0),
                    }
                )
                print(
                    "[pipeline stage0-2][progress] "
                    f"env={env_id} passed=no failed_stage={failed_stage} "
                    f"accepted_envs={accepted_env_count} accepted_task_blueprints={accepted_task_blueprint_count} "
                    f"completed={completed}/{total_envs}",
                    flush=True,
                )

    accepted_results: list[dict[str, Any]] = []
    for source_item in unique_records:
        env_id = str(source_item.get("resolved_env_id", ""))
        if env_id in accepted_by_env_id:
            accepted_results.append(accepted_by_env_id[env_id])

    accepted_manifest_items = [item["manifest_item"] for item in accepted_results]
    accepted_stage1_items = [item["stage1_item"] for item in accepted_results]
    accepted_stage2_env_results = [item["stage2_env_result"] for item in accepted_results]

    merged_output_payload = _rebuild_filtered_merged_payload(source_payload, accepted_results, passthrough)
    manifest_output_payload = _build_manifest_payload(accepted_manifest_items, args.input_file)
    stage1_output_payload = build_stage1_payload(accepted_stage1_items, stage1_args, args.input_file)
    stage2_output_payload = _build_stage2_payload(
        accepted_stage2_env_results,
        input_file=args.input_file,
        stage2_args=stage2_args,
    )

    save_json_file(args.output_merged_file, merged_output_payload)
    save_json_file(args.output_manifest_file, manifest_output_payload)
    save_json_file(args.output_init_state_file, stage1_output_payload)
    save_json_file(args.output_task_blueprints_file, stage2_output_payload)

    total_elapsed = round(time.monotonic() - started_at, 2)
    report_payload = {
        "metadata": {
            "generated_at": _now_iso(),
            "source_file": str(Path(args.input_file).resolve()),
            "max_worker": max(1, int(args.max_worker)),
            "failure_policy": args.failure_policy,
            "total_env_count": total_envs,
            "accepted_env_count": len(accepted_results),
            "dropped_env_count": len(dropped),
            "elapsed_seconds": total_elapsed,
            "output_files": {
                "merged_env_data": str(Path(args.output_merged_file).resolve()),
                "env_manifest": str(Path(args.output_manifest_file).resolve()),
                "init_state_bank": str(Path(args.output_init_state_file).resolve()),
                "task_blueprints": str(Path(args.output_task_blueprints_file).resolve()),
            },
        },
        "accepted_env_ids": [item.get("env_id", "") for item in accepted_results],
        "dropped_envs": dropped,
    }

    save_json_file(args.output_report_file, report_payload)

    print(json.dumps(report_payload["metadata"], ensure_ascii=False, indent=2))
    print(
        "[pipeline stage0-2] "
        f"saved merged={args.output_merged_file} manifest={args.output_manifest_file} "
        f"stage1={args.output_init_state_file} stage2={args.output_task_blueprints_file} report={args.output_report_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
