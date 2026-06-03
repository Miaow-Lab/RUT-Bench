from typing import Any

from utils.tool_parser import extract_json_from_text


def call_json_object_llm(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Any:
    from utils.call_llm import llm_inference

    response = llm_inference(
        provider="openai",
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=timeout,
        api_key=api_key,
        base_url=base_url,
        response_format={"type": "json_object"},
    )
    return extract_json_from_text(response or "")


def unwrap_init_config(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        for key in ("init_config", "config", "data"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        return payload
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        return payload[0]
    return None
