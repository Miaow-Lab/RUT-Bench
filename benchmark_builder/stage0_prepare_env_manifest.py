import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_builder.config import DEFAULT_ENV_MANIFEST, DEFAULT_MERGED_ENV_DATA, PROMPTABLE_CODE_CHAR_LIMIT
from benchmark_builder.common.discoverability import merge_discoverability_tools
from benchmark_builder.common.serialization import read_json_file, save_json_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize merged environment metadata into a stable benchmark manifest.")
    parser.add_argument("--input-file", default=str(DEFAULT_MERGED_ENV_DATA), help="Path to merged_env_data.json")
    parser.add_argument("--output-file", default=str(DEFAULT_ENV_MANIFEST), help="Path to env_manifest.json")
    parser.add_argument("--max-envs", type=int, default=0, help="Optional limit for smoke tests. 0 means all envs.")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        return [item for item in data.values() if isinstance(item, dict)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    raise TypeError(f"Unsupported merged env data type: {type(data).__name__}")


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _normalize_constraints(rules: Any) -> tuple[list[str], str]:
    if isinstance(rules, str):
        items = [line.strip() for line in rules.splitlines() if line.strip()]
    elif isinstance(rules, list):
        items = [_clean_text(item) for item in rules if _clean_text(item)]
    else:
        items = []

    deduped: list[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)

    if not deduped:
        deduped = ["No explicit constraint rules provided."]

    numbered = "\n".join(f"{index}. {item}" for index, item in enumerate(deduped, start=1))
    return deduped, numbered


def _normalize_tool_parameters(parameters: Any) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}, "required": []}

    normalized = dict(parameters)
    if normalized.get("type") != "object":
        normalized["type"] = "object"

    properties = normalized.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    normalized["properties"] = properties

    required = normalized.get("required")
    if not isinstance(required, list):
        required = []
    normalized["required"] = [str(item) for item in required if str(item).strip() in properties]
    return normalized


def _normalize_tools(raw_tools: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tools, list):
        return []

    normalized: list[dict[str, Any]] = []
    seen_names = set()
    for entry in raw_tools:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function", entry)
        if not isinstance(fn, dict):
            continue
        name = _clean_text(fn.get("name"))
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        normalized.append(
            {
                "name": name,
                "description": _clean_text(fn.get("description")),
                "parameters": _normalize_tool_parameters(fn.get("parameters")),
            }
        )
    return normalized


def _build_tool_summary(tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    summaries: list[dict[str, Any]] = []
    lines: list[str] = []
    for tool in tools:
        properties = tool.get("parameters", {}).get("properties", {})
        required = set(tool.get("parameters", {}).get("required", []))
        parameter_names = sorted(str(name) for name in properties.keys())
        entry = {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameter_names": parameter_names,
            "required_parameters": sorted(str(name) for name in required),
            "parameter_count": len(parameter_names),
        }
        summaries.append(entry)
        params_text = ", ".join(f"{name}{'*' if name in required else ''}" for name in parameter_names) or "(none)"
        lines.append(f"- {tool['name']}({params_text}): {tool.get('description', '')}")
    return summaries, "\n".join(lines)


def _promptable_env_code(env_class_code: str, env_class_def: str) -> str:
    code = env_class_code or ""
    if len(code) <= PROMPTABLE_CODE_CHAR_LIMIT:
        return code
    header = env_class_def.strip()
    excerpt_budget = max(PROMPTABLE_CODE_CHAR_LIMIT - len(header) - 32, 0)
    prefix = code[:excerpt_budget]
    return (header + "\n\n" + prefix + "\n...<truncated for prompt budget>...").strip()


def _build_complexity(record: dict[str, Any], tools: list[dict[str, Any]], env_structure: dict[str, Any]) -> dict[str, Any]:
    states = env_structure.get("states", {}) if isinstance(env_structure, dict) else {}
    methods = env_structure.get("methods", {}) if isinstance(env_structure, dict) else {}
    state_names = [str(name) for name in states.keys()]
    method_names = [str(name) for name in methods.keys()]
    return {
        "tool_count": len(tools),
        "state_container_count": len(state_names),
        "method_count": len(method_names),
        "code_length": len(record.get("env_class_code", "") or ""),
        "intro_length": len(record.get("environment_introduction", "") or ""),
        "constraint_count": len(record.get("constraints_rules_list", [])),
    }


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    env_structure = record.get("env_structure") if isinstance(record.get("env_structure"), dict) else {}
    states = env_structure.get("states") if isinstance(env_structure.get("states"), dict) else {}
    methods = env_structure.get("methods") if isinstance(env_structure.get("methods"), dict) else {}
    constraints_list, constraints_text = _normalize_constraints(record.get("constraints_rules"))
    tools = merge_discoverability_tools(_normalize_tools(record.get("tools")))
    tool_summary, tool_summary_text = _build_tool_summary(tools)

    normalized = {
        "env_id": _clean_text(record.get("env_id")),
        "environment_summary": _clean_text(record.get("environment_summary")),
        "environment_introduction": _clean_text(record.get("environment_introduction")),
        "state_space_definition": record.get("state_space_definition", ""),
        "operation_list": record.get("operation_list", []),
        "env_class_name": _clean_text(record.get("env_class_name")),
        "env_class_def": record.get("env_class_def", ""),
        "env_class_code": record.get("env_class_code", ""),
        "promptable_env_code": _promptable_env_code(record.get("env_class_code", ""), record.get("env_class_def", "")),
        "constraints_rules_list": constraints_list,
        "constraints_rules_text": constraints_text,
        "tools": tools,
        "tool_summary": tool_summary,
        "tool_summary_text": tool_summary_text,
        "env_structure": {"states": states, "methods": methods},
        "state_container_names": [str(name) for name in states.keys()],
        "method_names": [str(name) for name in methods.keys()],
    }
    normalized["complexity"] = _build_complexity(normalized, tools, normalized["env_structure"])
    return normalized


def _summarize_manifest(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "env_count": 0,
            "tool_count": {"min": 0, "avg": 0.0, "max": 0},
            "state_container_count": {"min": 0, "avg": 0.0, "max": 0},
            "method_count": {"min": 0, "avg": 0.0, "max": 0},
            "code_length": {"min": 0, "avg": 0.0, "max": 0},
        }

    def build_stats(field: str) -> dict[str, Any]:
        values = [int(item["complexity"][field]) for item in items]
        return {"min": min(values), "avg": round(mean(values), 2), "max": max(values)}

    return {
        "env_count": len(items),
        "tool_count": build_stats("tool_count"),
        "state_container_count": build_stats("state_container_count"),
        "method_count": build_stats("method_count"),
        "code_length": build_stats("code_length"),
    }


def main() -> None:
    args = parse_args()
    raw = read_json_file(args.input_file)
    records = _ensure_records(raw)
    if args.max_envs > 0:
        records = records[: args.max_envs]

    items = [_normalize_record(record) for record in records]
    payload = {
        "metadata": {
            "generated_at": _now_iso(),
            "source_file": str(Path(args.input_file).resolve()),
            "statistics": _summarize_manifest(items),
        },
        "items": items,
    }
    save_json_file(args.output_file, payload)
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))
    print(f"Saved {len(items)} normalized environments to {args.output_file}")


if __name__ == "__main__":
    main()