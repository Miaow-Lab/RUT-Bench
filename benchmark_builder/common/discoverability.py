from __future__ import annotations

from copy import deepcopy
from types import MethodType
from typing import Any


_DISCOVERABILITY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_state_containers",
        "description": (
            "List the environment's public state containers and their sizes. Use this first when you "
            "need to discover where entities and ids are stored. Read-only utility."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "list_state_entities",
        "description": (
            "List entities or key/value entries from a state container. Useful for browsing available "
            "records and recovering internal ids when specialized lookup tools are missing. Read-only utility."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "container_name": {"type": "string"},
                "limit": {"type": "integer"},
                "offset": {"type": "integer"},
            },
            "required": ["container_name"],
        },
    },
    {
        "name": "search_state_entities",
        "description": (
            "Search environment state records by free-text query across ids and common scalar fields such as "
            "name, title, username, ticker, symbol, email, phone, and status. Use it to map natural-language "
            "entity mentions to internal records. Read-only utility."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "container_name": {"type": ["string", "null"]},
                "field_names": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
                "exact": {"type": "boolean"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_state_entity",
        "description": (
            "Fetch a specific entity or entry from a state container by its key or id-like handle. Useful after "
            "listing or searching records. Read-only utility."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "container_name": {"type": "string"},
                "entity_key": {"type": "string"},
            },
            "required": ["container_name", "entity_key"],
        },
    },
]


def merge_discoverability_tools(tools: Any) -> list[dict[str, Any]]:
    base_tools = tools if isinstance(tools, list) else []
    merged: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for tool in [*base_tools, *_DISCOVERABILITY_TOOLS]:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name") or "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        merged.append(tool)
    return merged


def attach_discoverability_tools(env_instance: Any) -> None:
    for name, fn in {
        "list_state_containers": _list_state_containers,
        "list_state_entities": _list_state_entities,
        "search_state_entities": _search_state_entities,
        "get_state_entity": _get_state_entity,
    }.items():
        if hasattr(env_instance, name):
            continue
        setattr(env_instance, name, MethodType(fn, env_instance))


def _public_state_items(env_instance: Any) -> dict[str, Any]:
    items: dict[str, Any] = {}
    for name, value in vars(env_instance).items():
        if name.startswith("_"):
            continue
        if callable(value):
            continue
        items[name] = value
    return items


def _copy_value(value: Any) -> Any:
    try:
        return deepcopy(value)
    except Exception:
        return value


def _looks_like_single_entity_record(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    id_like_keys = {"id", "name", "title", "username", "ticker", "symbol", "email", "phone"}
    for key in value.keys():
        key_text = str(key)
        if key_text in id_like_keys or key_text.endswith("_id"):
            return True
    return False


def _discover_entity_key(record: dict[str, Any]) -> str:
    preferred_fields = (
        "id",
        "name",
        "title",
        "username",
        "ticker",
        "symbol",
        "email",
        "phone",
    )
    for field in preferred_fields:
        value = record.get(field)
        if value not in (None, ""):
            return str(value)
    for field, value in record.items():
        field_text = str(field)
        if field_text.endswith("_id") and value not in (None, ""):
            return str(value)
    return "singleton"


def _iter_entities(container: Any) -> list[tuple[str, Any]]:
    if isinstance(container, dict):
        if _looks_like_single_entity_record(container):
            return [(_discover_entity_key(container), container)]
        return [(str(key), value) for key, value in container.items()]
    if isinstance(container, (list, tuple)):
        return [(str(index), value) for index, value in enumerate(container)]
    return []


def _normalize_limit(value: Any, default: int, maximum: int) -> int:
    try:
        limit = int(value)
    except Exception:
        return default
    if limit <= 0:
        return default
    return min(limit, maximum)


def _normalize_offset(value: Any) -> int:
    try:
        offset = int(value)
    except Exception:
        return 0
    return max(offset, 0)


def _scalar_fields(value: Any, prefix: str = "", depth: int = 0) -> list[tuple[str, str]]:
    if depth > 2:
        return []
    if isinstance(value, dict):
        fields: list[tuple[str, str]] = []
        for index, (key, nested_value) in enumerate(value.items()):
            if index >= 20:
                break
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            fields.extend(_scalar_fields(nested_value, nested_prefix, depth + 1))
        return fields
    if isinstance(value, (list, tuple)):
        fields = []
        for index, nested_value in enumerate(value[:10]):
            nested_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            fields.extend(_scalar_fields(nested_value, nested_prefix, depth + 1))
        return fields
    if value in (None, ""):
        return []
    return [(prefix or "value", str(value))]


def _entity_payload(container_name: str, entity_key: str, record: Any) -> dict[str, Any]:
    return {
        "container_name": container_name,
        "entity_key": entity_key,
        "record": _copy_value(record),
    }


def _get_container(self: Any, container_name: str) -> tuple[Any, dict[str, Any] | None]:
    if not isinstance(container_name, str) or not container_name.strip():
        return None, {"success": False, "error": "container_name must be a non-empty string"}
    public_items = _public_state_items(self)
    if container_name not in public_items:
        available = sorted(public_items.keys())
        return None, {
            "success": False,
            "error": f"Unknown container '{container_name}'. Available containers: {available}",
        }
    return public_items[container_name], None


def _list_state_containers(self: Any) -> dict[str, Any]:
    containers = []
    for name, value in _public_state_items(self).items():
        entry = {
            "container_name": name,
            "container_type": type(value).__name__,
            "is_structured": isinstance(value, (dict, list, tuple)),
        }
        if isinstance(value, dict):
            entry["item_count"] = 1 if _looks_like_single_entity_record(value) else len(value)
            entry["sample_keys"] = [str(key) for key in list(value.keys())[:5]]
        elif isinstance(value, (list, tuple)):
            entry["item_count"] = len(value)
            entry["sample_keys"] = [str(index) for index in range(min(len(value), 5))]
        else:
            entry["item_count"] = None
            entry["sample_value"] = _copy_value(value)
        containers.append(entry)
    containers.sort(key=lambda item: item["container_name"])
    return {"success": True, "data": containers}


def _list_state_entities(self: Any, container_name: str, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    container, error = _get_container(self, container_name)
    if error is not None:
        return error
    entities = _iter_entities(container)
    if not entities:
        return {
            "success": False,
            "error": f"Container '{container_name}' is not a browsable entity container",
        }
    normalized_limit = _normalize_limit(limit, default=50, maximum=200)
    normalized_offset = _normalize_offset(offset)
    window = entities[normalized_offset : normalized_offset + normalized_limit]
    return {
        "success": True,
        "data": {
            "container_name": container_name,
            "total_count": len(entities),
            "offset": normalized_offset,
            "limit": normalized_limit,
            "items": [_entity_payload(container_name, entity_key, record) for entity_key, record in window],
        },
    }


def _search_state_entities(
    self: Any,
    query: str,
    container_name: str | None = None,
    field_names: list[str] | None = None,
    exact: bool = False,
    limit: int = 20,
) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        return {"success": False, "error": "query must be a non-empty string"}
    normalized_query = query.strip().lower()
    normalized_limit = _normalize_limit(limit, default=20, maximum=100)
    requested_fields = None
    if field_names is not None:
        if not isinstance(field_names, list):
            return {"success": False, "error": "field_names must be a list of strings when provided"}
        requested_fields = {str(name).strip() for name in field_names if str(name).strip()}

    public_items = _public_state_items(self)
    if container_name is not None:
        container, error = _get_container(self, container_name)
        if error is not None:
            return error
        search_targets = {container_name: container}
    else:
        search_targets = public_items

    matches: list[dict[str, Any]] = []
    for current_container_name, container in search_targets.items():
        for entity_key, record in _iter_entities(container):
            searchable_fields = [("entity_key", entity_key)]
            searchable_fields.extend(_scalar_fields(record))
            if requested_fields is not None:
                searchable_fields = [item for item in searchable_fields if item[0] in requested_fields]
                if not searchable_fields:
                    continue
            matched_fields = []
            score = 2
            for field_name, field_value in searchable_fields:
                candidate = field_value.strip().lower()
                if not candidate:
                    continue
                if exact:
                    is_match = candidate == normalized_query
                else:
                    is_match = normalized_query in candidate
                if not is_match:
                    continue
                matched_fields.append({"field_name": field_name, "field_value": field_value})
                score = min(score, 0 if candidate == normalized_query else 1)
            if not matched_fields:
                continue
            matches.append(
                {
                    "container_name": current_container_name,
                    "entity_key": entity_key,
                    "matched_fields": matched_fields,
                    "record": _copy_value(record),
                    "_score": score,
                }
            )

    matches.sort(key=lambda item: (item["_score"], item["container_name"], item["entity_key"]))
    trimmed_matches = []
    for item in matches[:normalized_limit]:
        payload = dict(item)
        payload.pop("_score", None)
        trimmed_matches.append(payload)
    return {"success": True, "data": trimmed_matches}


def _get_state_entity(self: Any, container_name: str, entity_key: str) -> dict[str, Any]:
    container, error = _get_container(self, container_name)
    if error is not None:
        return error
    if not isinstance(entity_key, str) or not entity_key.strip():
        return {"success": False, "error": "entity_key must be a non-empty string"}
    normalized_key = entity_key.strip()
    for current_entity_key, record in _iter_entities(container):
        if current_entity_key == normalized_key:
            return {"success": True, "data": _entity_payload(container_name, current_entity_key, record)}
    return {
        "success": False,
        "error": f"Entity '{normalized_key}' not found in container '{container_name}'",
    }
