import json
from copy import deepcopy
from functools import wraps
from typing import Any

from benchmark_builder.common.discoverability import attach_discoverability_tools
from utils.auto_env import build_env_from_str


def build_env_runtime(env_item: dict[str, Any], max_steps: int):
    env = build_env_from_str(
        env_item["env_class_code"],
        env_item["env_class_name"],
        max_steps=max_steps,
    )
    original_env_init = env.env_init

    @wraps(original_env_init)
    def env_init_with_discoverability(*args: Any, **kwargs: Any):
        result = original_env_init(*args, **kwargs)
        if env.env_instance is not None:
            attach_discoverability_tools(env.env_instance)
        return result

    env.env_init = env_init_with_discoverability
    if env.env_instance is not None:
        attach_discoverability_tools(env.env_instance)
    return env


def stable_json_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)


def summarize_state_population(state: dict[str, Any], state_container_names: list[str] | None = None) -> dict[str, Any]:
    container_names = state_container_names or list(state.keys())
    structured_container_count = 0
    populated_container_count = 0
    total_structured_items = 0
    max_container_size = 0
    populated_container_names: list[str] = []
    scalar_field_count = 0

    for name in container_names:
        value = state.get(name)
        if isinstance(value, (dict, list, tuple)):
            structured_container_count += 1
            size = len(value)
            total_structured_items += size
            max_container_size = max(max_container_size, size)
            if size > 0:
                populated_container_count += 1
                populated_container_names.append(name)
        elif value not in (None, ""):
            scalar_field_count += 1

    if total_structured_items <= 4:
        state_profile = "sparse"
    elif total_structured_items <= 15:
        state_profile = "balanced"
    else:
        state_profile = "crowded"

    return {
        "structured_container_count": structured_container_count,
        "populated_container_count": populated_container_count,
        "total_structured_items": total_structured_items,
        "max_container_size": max_container_size,
        "populated_container_names": populated_container_names,
        "scalar_field_count": scalar_field_count,
        "state_profile": state_profile,
    }


def validate_init_config(env_item: dict[str, Any], init_config: dict[str, Any], max_steps: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "is_valid": False,
        "reason": "unknown",
        "warnings": [],
        "state_stats": {},
    }
    try:
        env = build_env_runtime(env_item, max_steps=max_steps)
        env.env_init(init_config=deepcopy(init_config))
        state = env.get_state_info()
    except Exception as exc:
        result["reason"] = f"env init failed: {exc}"
        return result

    if not isinstance(state, dict) or not state:
        result["reason"] = "state snapshot is empty"
        return result

    state_stats = summarize_state_population(state, env_item.get("state_container_names", []))
    result["state_stats"] = state_stats

    total_structured_items = int(state_stats.get("total_structured_items", 0))
    populated_container_count = int(state_stats.get("populated_container_count", 0))
    max_container_size = int(state_stats.get("max_container_size", 0))
    tool_count = int((env_item.get("complexity") or {}).get("tool_count", len(env_item.get("tools", []))))

    if total_structured_items < 3 and max_container_size < 2:
        result["reason"] = "state is too sparse to support non-trivial tasks"
        return result

    if populated_container_count < 1:
        result["reason"] = "no populated structured containers found"
        return result

    if populated_container_count < 2:
        if tool_count >= 5 and max_container_size >= 3:
            result["warnings"].append("only one populated structured container; accepted because the state is still rich enough")
        else:
            result["reason"] = "fewer than two populated structured containers"
            return result

    result["is_valid"] = True
    result["reason"] = "ok"
    return result
