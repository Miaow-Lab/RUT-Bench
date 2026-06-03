"""
Tool-use helpers for native function calling (OpenAI tools API).

Converts tool schemas to OpenAI format and provides utilities for
serializing tool-call messages and extracting JSON from text.
"""
import re
import json
from typing import Any, Dict, List, Optional


# =====================================================================
# Native function-calling helpers  (OpenAI tools API)
# =====================================================================

# A special "tool" the agent calls when it has completed the task.
TASK_COMPLETE_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "task_complete",
        "description": (
            "Signal that the task is fully completed. "
            "Call this ONLY when all required actions have been performed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of what was accomplished.",
                }
            },
            "required": ["summary"],
        },
    },
}


def build_openai_tools(tools: List[Dict[str, Any]], *, include_task_complete: bool = True) -> List[Dict[str, Any]]:
    """Convert a list of flat tool definitions into OpenAI function-calling format.

    Each input tool should have::

        {"name": "...", "description": "...", "parameters": {...}}

    Output (per tool)::

        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

    If *include_task_complete* is True (default), ``TASK_COMPLETE_TOOL`` is
    appended so that the agent can signal task completion natively.
    """
    openai_tools: List[Dict[str, Any]] = []
    for t in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {
                    "type": "object", "properties": {}, "required": []
                }),
            },
        })
    if include_task_complete:
        openai_tools.append(TASK_COMPLETE_TOOL)
    return openai_tools


def message_to_dict(msg) -> Dict[str, Any]:
    """Convert an OpenAI ChatCompletionMessage object to a plain dict.

    Handles ``tool_calls`` by serializing each call into a JSON-friendly
    structure so that trajectories can be saved to disk.
    """
    d: Dict[str, Any] = {"role": msg.role}
    if msg.content:
        d["content"] = msg.content
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,  # raw JSON string
                },
            }
            for tc in msg.tool_calls
        ]
    return d


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _repair_json(raw: str) -> str:
    """Apply common repairs to a JSON string that failed standard parsing."""
    # Strip markdown fences that some models wrap around JSON inside tags
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw)
    # Remove trailing commas before } or ]
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    # Replace JS single-quoted strings with double-quoted (simple cases)
    raw = re.sub(r"(?<![\\\w])'([^'\\\n]*)'(?![\w])", r'"\1"', raw)
    return raw.strip()


def extract_json_from_text(text: str) -> Optional[Any]:
    """Extract the first valid JSON object or array from arbitrary text.

    Handles:
    - Raw JSON starting with { or [
    - Fenced code blocks: ```json ... ``` or ``` ... ```
    - Trailing commas, single-quoted keys/values (via _repair_json)
    """
    decoder = json.JSONDecoder()

    def _try_parse(candidate: str) -> Optional[Any]:
        """Try direct parse, then repair-and-parse."""
        for ch in ("{", "["):
            idx = candidate.find(ch)
            if idx == -1:
                continue
            snippet = candidate[idx:]
            # Direct
            try:
                obj, _ = decoder.raw_decode(snippet)
                return obj
            except Exception:
                pass
            # Repaired
            try:
                obj, _ = decoder.raw_decode(_repair_json(snippet))
                return obj
            except Exception:
                pass
        return None

    # 1. Try fenced code block first
    fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
    if fence_match:
        result = _try_parse(fence_match.group(1).strip())
        if result is not None:
            return result

    # 2. Fall back to raw text scan
    return _try_parse(text)
