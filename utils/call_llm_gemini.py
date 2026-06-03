"""
Native Gemini inference via Google Generative Language REST API.

Properly handles thought_signature for multi-turn tool calling with
Gemini 2.5 and later thinking models.  The key mechanism: the raw Gemini
Content returned from each API call is stored in the "_gemini_content" field
of the returned OpenAI-format message dict.  On subsequent turns,
_messages_to_gemini detects that field and replays the content verbatim,
preserving thought_signature without any proxy stripping.

Environment variables:
  GEMINI_API_KEY  or  GOOGLE_API_KEY  — Google API key.
"""
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

_GEMINI_REST_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


# ---------------------------------------------------------------------------
# OpenAI → Gemini format converters
# ---------------------------------------------------------------------------

def _tools_to_gemini(tools: List[Dict]) -> List[Dict]:
    """Convert OpenAI tool definitions to Gemini functionDeclarations."""
    decls = []
    for tool in tools or []:
        if tool.get("type") != "function":
            continue
        fn = tool.get("function", {})
        decl: Dict[str, Any] = {"name": fn.get("name", "")}
        if fn.get("description"):
            decl["description"] = fn["description"]
        if fn.get("parameters"):
            decl["parameters"] = fn["parameters"]
        decls.append(decl)
    return [{"function_declarations": decls}] if decls else []


def _messages_to_gemini(messages: List[Dict]) -> tuple[Optional[str], List[Dict]]:
    """Convert OpenAI-format messages to (system_instruction, gemini_contents).

    Messages carrying a "_gemini_content" key are replayed verbatim so that
    thought_signature is preserved across conversation turns.

    OpenAI "tool" role messages are batched into user turns with
    functionResponse parts, which is what the Gemini API expects.
    """
    system_instruction: Optional[str] = None
    contents: List[Dict] = []

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")

        if role == "system":
            system_instruction = msg.get("content") or ""
            i += 1
            continue

        # Native Gemini content — replay as-is to preserve thought_signature
        if "_gemini_content" in msg:
            contents.append(msg["_gemini_content"])
            i += 1
            continue

        if role == "user":
            contents.append({
                "role": "user",
                "parts": [{"text": msg.get("content") or ""}],
            })
            i += 1

        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                parts = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    raw_args = fn.get("arguments", "{}")
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            args = {}
                    else:
                        args = raw_args or {}
                    parts.append({"functionCall": {"name": fn.get("name", ""), "args": args}})
                contents.append({"role": "model", "parts": parts})
            else:
                contents.append({
                    "role": "model",
                    "parts": [{"text": msg.get("content") or ""}],
                })
            i += 1

        elif role == "tool":
            # Batch consecutive tool responses into one user turn
            parts = []
            while i < len(messages) and messages[i].get("role") == "tool":
                t = messages[i]
                parts.append({
                    "functionResponse": {
                        "name": t.get("name") or "",
                        "response": {"result": t.get("content") or ""},
                    }
                })
                i += 1
            contents.append({"role": "user", "parts": parts})

        else:
            i += 1

    return system_instruction, contents


# ---------------------------------------------------------------------------
# Gemini → OpenAI format converter
# ---------------------------------------------------------------------------

def _filter_content_for_history(content: Dict) -> Dict:
    """Strip thought-text parts from a Gemini Content dict.

    Thought text can be large.  We only need the functionCall / text parts
    (which carry the thought_signature) for history replay.
    """
    parts = [p for p in (content.get("parts") or []) if not p.get("thought")]
    return {**content, "parts": parts}


def _gemini_response_to_openai(response_json: Dict) -> Dict:
    """Convert a Gemini generateContent response to an OpenAI-format message dict.

    The raw Gemini Content is stored under "_gemini_content" (with thought
    text stripped) so that thought_signature survives the round-trip when
    the message is replayed on the next turn via _messages_to_gemini.
    """
    candidates = response_json.get("candidates") or []
    if not candidates:
        return {"role": "assistant", "content": "", "tool_calls": [], "_gemini_content": {}}

    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []

    text_parts: List[str] = []
    fc_parts: List[Dict] = []

    for part in parts:
        if "functionCall" in part:
            fc_parts.append(part)
        elif "text" in part and not part.get("thought"):
            text_parts.append(part["text"])

    history_content = _filter_content_for_history(content)

    if fc_parts:
        tool_calls = []
        for part in fc_parts:
            fc = part["functionCall"]
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": fc.get("name", ""),
                    "arguments": json.dumps(fc.get("args") or {}, ensure_ascii=False),
                },
            })
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
            "_gemini_content": history_content,
        }

    return {
        "role": "assistant",
        "content": "\n".join(text_parts),
        "tool_calls": [],
        "_gemini_content": history_content,
    }


# ---------------------------------------------------------------------------
# Public inference function
# ---------------------------------------------------------------------------

def gemini_inference_with_tools(
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    api_key: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    temperature: Optional[float] = None,
    timeout: int = 120,
    **kwargs,
) -> Optional[Dict]:
    """Call the Gemini generateContent REST API with tool support.

    Args:
        model:           Gemini model ID, e.g. "gemini-2.5-pro-preview-05-06".
        messages:        OpenAI-format message list.  Messages with a
                         "_gemini_content" field are replayed verbatim to
                         preserve thought_signature for multi-turn tool calling.
        tools:           OpenAI-format tool definitions.
        api_key:         Google API key.  Falls back to GEMINI_API_KEY /
                         GOOGLE_API_KEY env vars.
        thinking_budget: Passed as generationConfig.thinkingConfig.thinkingBudget.
                         0 disables thinking (no thought_signature generated).
                         None leaves it at the model default (thinking enabled).
        temperature:     Sampling temperature.
        timeout:         HTTP timeout in seconds per attempt.

    Returns:
        OpenAI-format message dict with an extra "_gemini_content" field, or
        None after all retries are exhausted.
    """
    resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not resolved_key:
        raise ValueError(
            "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env."
        )

    system_instruction, contents = _messages_to_gemini(messages)
    gemini_tools = _tools_to_gemini(tools or [])

    payload: Dict[str, Any] = {"contents": contents}
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    if gemini_tools:
        payload["tools"] = gemini_tools

    gen_config: Dict[str, Any] = {}
    if thinking_budget is not None:
        gen_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    if temperature is not None:
        gen_config["temperature"] = temperature
    if gen_config:
        payload["generationConfig"] = gen_config

    model_id = model.removeprefix("models/")
    url = f"{_GEMINI_REST_BASE}/{model_id}:generateContent"

    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            resp = httpx.post(
                url,
                params={"key": resolved_key},
                json=payload,
                timeout=float(timeout),
            )
            resp.raise_for_status()
            return _gemini_response_to_openai(resp.json())
        except Exception as e:
            wait = retries * 10 + 10
            err_str = str(e).encode("ascii", errors="replace").decode("ascii")
            print(
                f"[LLM] {model} error: {err_str}. Retrying in {wait}s ({retries + 1}/{max_retries})..."
            )
            time.sleep(wait)
            retries += 1

    print(f"[LLM] Failed to get response from {model} after {max_retries} retries")
    return None
