import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_builder.common.serialization import read_json_file, save_json_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-score eval results with a relaxed completion rule.")
    parser.add_argument("--eval-result-file", required=True, help="Path to an existing eval_results.json file")
    parser.add_argument(
        "--output-file",
        default="",
        help="Optional output path. Defaults to <eval-result-file stem>_relaxed.json in the same directory.",
    )
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _safe_bool(value: Any) -> bool:
    return bool(value)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _resolve_output_path(eval_result_path: Path, output_file: str) -> Path:
    if output_file:
        return Path(output_file)
    return eval_result_path.with_name(f"{eval_result_path.stem}_relaxed.json")


def _compute_relaxed_completion(sample: dict[str, Any]) -> dict[str, Any]:
    completion = sample.get("completion", {}) if isinstance(sample.get("completion"), dict) else {}
    final_state_total = int(completion.get("final_state_assertion_total", 0) or 0)
    final_state_pass_count = int(completion.get("final_state_assertion_pass_count", 0) or 0)
    final_state_passed = final_state_total == 0 or final_state_pass_count == final_state_total
    expected_state_diff_passed = _safe_bool(completion.get("expected_state_diff_passed", True))

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
    original_success = _safe_bool(completion.get("semantic_success", completion.get("success", completion.get("passed", False))))

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


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    original_scores = [float(row["relaxed_completion"]["original_success"]) for row in rows]
    relaxed_scores = [float(row["relaxed_completion"]["relaxed_success"]) for row in rows]
    recovered_count = sum(1 for row in rows if row["relaxed_completion"]["recovered_by_relaxed"])
    final_state_passed_count = sum(1 for row in rows if row["relaxed_completion"]["final_state_assertions_passed"])
    expected_state_diff_passed_count = sum(1 for row in rows if row["relaxed_completion"]["expected_state_diff_passed"])
    forbidden_actions_passed_count = sum(1 for row in rows if row["relaxed_completion"]["forbidden_actions_passed"])
    return {
        "sample_count": len(rows),
        "original_success_rate": _mean(original_scores),
        "relaxed_success_rate": _mean(relaxed_scores),
        "delta_success_rate": _mean(relaxed_scores) - _mean(original_scores),
        "recovered_count": recovered_count,
        "recovered_rate": (recovered_count / len(rows)) if rows else 0.0,
        "final_state_assertions_passed_count": final_state_passed_count,
        "expected_state_diff_passed_count": expected_state_diff_passed_count,
        "forbidden_actions_passed_count": forbidden_actions_passed_count,
    }


def _group_rows(rows: list[dict[str, Any]], key_name: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_normalize_string(row.get(key_name), "unknown")].append(row)
    return {key: _summarize_rows(group_rows) for key, group_rows in sorted(grouped.items())}


def _group_flat_labels(rows: list[dict[str, Any]], key_name: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        labels = row.get(key_name, []) if isinstance(row.get(key_name), list) else []
        for label in labels:
            normalized = _normalize_string(label)
            if normalized:
                grouped[normalized].append(row)
    return {key: _summarize_rows(group_rows) for key, group_rows in sorted(grouped.items())}


def _build_failure_counters(rows: list[dict[str, Any]]) -> dict[str, Any]:
    original_hard_failures = Counter()
    recovered_from_failures = Counter()
    relaxed_failures = Counter()

    for row in rows:
        relaxed_completion = row["relaxed_completion"]
        hard_failures = relaxed_completion.get("original_hard_failures", [])
        if isinstance(hard_failures, list):
            for failure in hard_failures:
                original_hard_failures[_normalize_string(failure)] += 1
                if relaxed_completion.get("recovered_by_relaxed"):
                    recovered_from_failures[_normalize_string(failure)] += 1

        for failure in relaxed_completion.get("relaxed_failures", []):
            relaxed_failures[_normalize_string(failure)] += 1

    return {
        "original_hard_failures": dict(original_hard_failures.most_common()),
        "recovered_from_original_hard_failures": dict(recovered_from_failures.most_common()),
        "relaxed_failures": dict(relaxed_failures.most_common()),
    }


def main() -> int:
    args = parse_args()
    eval_result_path = Path(args.eval_result_file)
    payload = read_json_file(eval_result_path)
    samples = payload.get("samples", []) if isinstance(payload, dict) else []
    if not isinstance(samples, list):
        raise ValueError("eval result payload must contain a top-level 'samples' list")

    rescored_samples: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        rescored_samples.append(
            {
                **sample,
                "relaxed_completion": _compute_relaxed_completion(sample),
            }
        )

    output_payload = {
        "metadata": {
            "generated_at": _now_iso(),
            "source_eval_result_file": str(eval_result_path),
            "source_model": ((payload.get("metadata") or {}).get("model") if isinstance(payload, dict) else None),
            "criteria": [
                "all final_state_assertions must pass when present",
                "expected_state_diff must be contained when present",
                "no forbidden_action_violations",
            ],
        },
        "summary": {
            **_summarize_rows(rescored_samples),
            "variant_summary": _group_rows(rescored_samples, "variant"),
            "difficulty_summary": _group_rows(rescored_samples, "difficulty_bucket"),
            "task_type_summary": _group_rows(rescored_samples, "task_type"),
            "num_user_turns_summary": _group_rows(rescored_samples, "num_user_turns"),
            "min_required_action_turn_summary": _group_rows(rescored_samples, "min_required_action_turn"),
            "behavior_summary": _group_flat_labels(rescored_samples, "behavior_bundle"),
            "unstable_behavior_summary": _group_rows(rescored_samples, "unstable_behavior"),
            "failure_summary": _build_failure_counters(rescored_samples),
        },
        "samples": rescored_samples,
    }

    output_path = _resolve_output_path(eval_result_path, args.output_file)
    save_json_file(output_path, output_payload)
    print(f"Saved relaxed rescore report to {output_path}")
    print(f"Original SR: {output_payload['summary']['original_success_rate']:.4f}")
    print(f"Relaxed SR: {output_payload['summary']['relaxed_success_rate']:.4f}")
    print(f"Recovered samples: {output_payload['summary']['recovered_count']} / {output_payload['summary']['sample_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())