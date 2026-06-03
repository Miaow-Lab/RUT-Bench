"""
Local-model variant of run_eval.py.

Runs the EnvBuilder benchmark against a HuggingFace model loaded locally
(e.g. Qwen/Qwen3-8B), replacing the OpenAI-compatible API backend with a
`transformers` + `torch` inference loop that supports tool calling via
Qwen3's <tool_call> tag protocol.

Hardware notes
--------------
* Qwen3-8B in bfloat16 requires ~16 GB VRAM.  With --load-in-4bit it fits
  in ~5 GB (requires `bitsandbytes`).
* The RTX 5070 (sm_120 / Blackwell) needs PyTorch >= 2.7 which adds sm_120
  support.  Earlier builds (2.6 and below) will fall back to CPU.

Quick-start (inside the `dyco` conda env)
------------------------------------------
# Install newer torch for Blackwell (sm_120) GPU support:
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# Optional: 4-bit quantisation (fits 8B in ~5 GB VRAM):
pip install bitsandbytes>=0.44

# Run (model can be a HF id or a local directory):
python eval/run_eval_local.py \\
    --model-path Qwen/Qwen3-8B \\
    --benchmark-file benchmark_builder/output/run64/benchmark_full.jsonl \\
    --blueprint-file benchmark_builder/output/run64/task_blueprints.json \\
    --output-file benchmark_builder/output/run64/eval_results/qwen3_8b_local_eval_results.json \\
    --max-samples 20 \\
    --max-workers 1 \\
    --load-in-4bit \\
    --skip-reliability
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
import uuid
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load env vars from project root (.env contains OPENAI_* for the judge)
from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from benchmark_builder.common.discoverability import merge_discoverability_tools
from benchmark_builder.common.env_runtime import build_env_runtime
from benchmark_builder.common.serialization import read_json_file, read_jsonl_file, save_json_file
from benchmark_builder.config import (
    DEFAULT_BENCHMARK_FULL,
    DEFAULT_EVAL_JUDGE_MODEL,
    DEFAULT_EVAL_JUDGE_TEMPERATURE,
    DEFAULT_EVAL_MAX_SAMPLES,
    DEFAULT_EVAL_MAX_STEPS,
    DEFAULT_EVAL_MAX_WORKERS,
    DEFAULT_EVAL_RESULTS_DIR,
    DEFAULT_EVAL_TIMEOUT,
    DEFAULT_TASK_BLUEPRINTS,
)
from eval.completion import evaluate_completion
from eval.efficiency import evaluate_efficiency
from eval.reliability import evaluate_reliability
from utils.auto_env import get_state_diff


# ---------------------------------------------------------------------------
# Model family detection
# ---------------------------------------------------------------------------

def _detect_model_family(model_path: str) -> str:
    """Heuristic model family detection from model path or HF model ID.

    Returns one of: "qwen3", "qwen2", "llama3", "llama", "mistral", "generic".
    """
    lower = model_path.lower()
    if "qwen3" in lower or "qwen-3" in lower:
        return "qwen3"
    if "qwen2.5" in lower or "qwen2-5" in lower:
        return "qwen2"
    if "qwen2" in lower or "qwen-2" in lower:
        return "qwen2"
    if "qwen" in lower:
        return "qwen3"  # default newer Qwen to qwen3 behaviour
    if "llama-3" in lower or "llama3" in lower:
        return "llama3"
    if "llama" in lower:
        return "llama"
    if "mistral" in lower or "mixtral" in lower:
        return "mistral"
    return "generic"


# ---------------------------------------------------------------------------
# Tool call parsing helpers
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
# Llama 3.1 wraps tool calls with these special tokens; they appear in the
# raw decoded text when skip_special_tokens=False.
_PYTHON_TAG_RE = re.compile(r"<\|python_tag\|>(.*?)(?:<\|eom_id\|>|$)", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _make_tool_call(name: str, arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        arguments_str = json.dumps(arguments, ensure_ascii=False)
    else:
        arguments_str = str(arguments)
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {"name": name, "arguments": arguments_str},
    }


def _parse_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    """Extract tool calls from model output, supporting multiple formats.

    Format priority:
    1. Qwen3  – <tool_call>{"name": ..., "arguments": {...}}</tool_call>
    2. Llama3 – <|python_tag|>{"name": ..., "parameters": {...}}<|eom_id|>
    3. Generic JSON – entire output is a JSON function-call object
    4. JSON array  – entire output is a list of function-call objects

    Returns (tool_calls, remaining_text).
    """
    tool_calls: list[dict[str, Any]] = []

    # --- Format 1: Qwen3 <tool_call> tags ---
    for match in _TOOL_CALL_RE.finditer(text):
        raw = match.group(1).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        name = str(parsed.get("name") or "").strip()
        if not name:
            continue
        tool_calls.append(_make_tool_call(name, parsed.get("arguments", {})))

    if tool_calls:
        return tool_calls, _TOOL_CALL_RE.sub("", text).strip()

    # --- Format 2: Llama3 <|python_tag|> tokens (raw decode mode) ---
    for match in _PYTHON_TAG_RE.finditer(text):
        raw = match.group(1).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        name = str(parsed.get("name") or "").strip()
        if not name:
            continue
        args = parsed.get("parameters") or parsed.get("arguments") or {}
        tool_calls.append(_make_tool_call(name, args))

    if tool_calls:
        remaining = _PYTHON_TAG_RE.sub("", text).strip()
        return tool_calls, remaining

    # --- Format 3 & 4: Entire output is a JSON function-call object/list ---
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict) and "name" in parsed:
            name = str(parsed.get("name", "")).strip()
            args = parsed.get("parameters") or parsed.get("arguments") or {}
            if name and isinstance(args, dict):
                return [_make_tool_call(name, args)], ""

        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                args = item.get("parameters") or item.get("arguments") or {}
                if name and isinstance(args, dict):
                    tool_calls.append(_make_tool_call(name, args))
            if tool_calls:
                return tool_calls, ""

    return [], text


# ---------------------------------------------------------------------------
# Conversion helpers for messages → Qwen3 template format
# ---------------------------------------------------------------------------

def _convert_messages_for_qwen(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert messages from OpenAI wire format to Qwen3 apply_chat_template format.

    Differences handled:
    - Tool-call messages: `arguments` field must be a dict, not a JSON string.
    - Tool-result messages: only need role/content/name; tool_call_id is optional.
    """
    converted: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role") or "").strip()

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                qwen_calls = []
                for tc in tool_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    qwen_calls.append(
                        {
                            "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                            "type": "function",
                            "function": {
                                "name": str(fn.get("name") or ""),
                                "arguments": args,
                            },
                        }
                    )
                converted.append(
                    {
                        "role": "assistant",
                        "content": msg.get("content") or "",
                        "tool_calls": qwen_calls,
                    }
                )
            else:
                converted.append({"role": "assistant", "content": msg.get("content") or ""})

        elif role == "tool":
            converted.append(
                {
                    "role": "tool",
                    "content": str(msg.get("content") or ""),
                    "name": str(msg.get("name") or ""),
                }
            )

        else:
            converted.append({"role": role, "content": str(msg.get("content") or "")})

    return converted


def _convert_tools_for_qwen(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure tools are in proper OpenAI / Qwen3 schema format."""
    result = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            result.append(tool)
            continue
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        parameters = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {"type": "object", "properties": {}}
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description") or ""),
                    "parameters": parameters,
                },
            }
        )
    return result


# ---------------------------------------------------------------------------
# Local model backend
# ---------------------------------------------------------------------------

class LocalModelBackend:
    """
    Wraps a HuggingFace CausalLM for tool-calling inference.

    Supports multiple model families (Qwen3, Llama3, generic) and auto-detects
    the correct tool-call format from the model path/ID.

    IMPORTANT: always use the *Instruct* (chat/instruction-tuned) variant of the
    model, not the base model.  Base models have no chat template and will
    produce meaningless output for tool-calling tasks.

    Supported tool-call output formats:
      - Qwen3:   <tool_call>{"name":...,"arguments":{...}}</tool_call>
      - Llama3:  <|python_tag|>{"name":...,"parameters":{...}}<|eom_id|>
                 or plain JSON {"name":...,"parameters":{...}}
      - Generic: plain JSON object/array matching a function-call schema
    """

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        device_map: str = "auto",
        max_new_tokens: int = 2048,
        trust_remote_code: bool = True,
        model_family: str = "",
    ) -> None:
        self.model_path = model_path
        self.dtype_str = dtype
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code
        self.model_family = model_family or _detect_model_family(model_path)
        self.tokenizer = None
        self.model = None
        self._device: str = "cpu"

    def load(self) -> None:
        """Load model and tokenizer.  Call once before inference."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(
            f"[LocalModel] Loading tokenizer from {self.model_path!r} "
            f"(detected family: {self.model_family}) ...",
            flush=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )

        # Resolve torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype_str, torch.bfloat16)

        # Check CUDA availability (catches sm_120/Blackwell mismatch at runtime)
        cuda_ok = False
        if torch.cuda.is_available():
            try:
                _probe = torch.tensor([1.0]).cuda()
                _ = _probe + 1
                cuda_ok = True
            except Exception as cuda_err:
                warnings.warn(
                    f"[LocalModel] CUDA detected but unusable ({cuda_err}). "
                    "Falling back to CPU. Install PyTorch >= 2.7 for RTX 5070 (sm_120) support.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        effective_device_map = self.device_map if cuda_ok else "cpu"
        self._device = "cuda" if cuda_ok else "cpu"

        print(
            f"[LocalModel] Loading model from {self.model_path!r} "
            f"(dtype={self.dtype_str}, 4bit={self.load_in_4bit}, device={effective_device_map}) ...",
            flush=True,
        )

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
        }

        if self.load_in_4bit and cuda_ok:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs["quantization_config"] = bnb_config
                model_kwargs["device_map"] = effective_device_map
            except ImportError:
                print(
                    "[LocalModel] bitsandbytes not installed; skipping 4-bit quantisation. "
                    "Install with: pip install bitsandbytes>=0.44",
                    flush=True,
                )
                model_kwargs["dtype"] = torch_dtype
                model_kwargs["device_map"] = effective_device_map
        else:
            model_kwargs["dtype"] = torch_dtype
            if effective_device_map == "cpu":
                model_kwargs["device_map"] = "cpu"
            else:
                model_kwargs["device_map"] = effective_device_map

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        self.model.eval()
        print("[LocalModel] Model loaded.", flush=True)

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Run one inference step and return an OpenAI-compatible message dict.

        Returns {"role": "assistant", "content": str | None, "tool_calls": list | None}
        """
        import torch

        converted_messages = _convert_messages_for_qwen(messages)
        converted_tools = _convert_tools_for_qwen(tools) if tools else None

        # Model-family-specific template kwargs
        template_kwargs: dict[str, Any] = {}
        if self.model_family == "qwen3":
            template_kwargs["enable_thinking"] = False

        # Llama3 outputs tool calls wrapped in <|python_tag|>...<|eom_id|> special
        # tokens.  We must keep those tokens visible so _parse_tool_calls can find
        # them; only strip them after parsing.
        skip_special = self.model_family not in ("llama3", "llama")

        text = self.tokenizer.apply_chat_template(
            converted_messages,
            tools=converted_tools,
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt")
        if self._device == "cuda":
            model_inputs = model_inputs.to("cuda")

        with torch.no_grad():
            if temperature <= 0.0:
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            else:
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                )

        input_len = model_inputs["input_ids"].shape[1]
        new_token_ids = generated_ids[0][input_len:]
        generated_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=skip_special)

        # Strip thinking blocks and parse tool calls
        generated_text = _strip_thinking(generated_text)
        tool_calls, remaining_text = _parse_tool_calls(generated_text)

        # For Llama with skip_special=False, strip any leftover special-token
        # text that isn't part of a parsed tool call
        if not skip_special and not tool_calls:
            generated_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
            generated_text = _strip_thinking(generated_text)
            tool_calls, remaining_text = _parse_tool_calls(generated_text)

        return {
            "role": "assistant",
            "content": remaining_text if remaining_text else None,
            "tool_calls": tool_calls if tool_calls else None,
        }


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

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


def _safe_json_loads(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _observation_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


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


def _run_single_sample_local(
    sample: dict[str, Any],
    backend: LocalModelBackend,
    max_steps: int,
    temperature: float,
) -> dict[str, Any]:
    """Local-model version of _run_single_sample from run_eval.py."""
    runtime_entry = sample["runtime"]
    env_item = {
        "env_class_code": runtime_entry["env_class_code"],
        "env_class_name": runtime_entry["env_class_name"],
    }
    env = build_env_runtime(env_item, max_steps=max_steps)
    env.env_init(init_config=_json_safe_snapshot(sample.get("init_config", {})))
    initial_state = _json_safe_snapshot(env.get_state_info())

    tools = _convert_tools_for_openai(merge_discoverability_tools(runtime_entry.get("tools", [])))
    user_turns = sample.get("user_turns", []) if isinstance(sample.get("user_turns"), list) else []
    transcript: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _build_agent_system_prompt(sample, runtime_entry)}
    ]

    first_turn = (
        _normalize_string(user_turns[0].get("user_message")) if user_turns
        else _normalize_string(sample.get("user_request"))
    )
    messages.append({"role": "user", "content": first_turn})
    transcript.append({"role": "user", "content": first_turn})
    next_user_turn_index = 1

    actual_actions: list[dict[str, Any]] = []
    actual_tool_trace: list[dict[str, Any]] = []
    error_message: str | None = None

    sample_id = sample.get("sample_id", "?")

    for action_index in range(1, max_steps + 1):
        try:
            assistant_payload = backend.chat_with_tools(
                messages=messages,
                tools=tools,
                temperature=temperature,
            )
        except Exception as exc:
            error_message = f"local model inference error at step {action_index}: {exc}"
            print(f"[DIAG][{sample_id}] step {action_index} inference EXCEPTION: {exc}", flush=True)
            traceback.print_exc()
            break

        raw_content = assistant_payload.get("content")
        raw_tool_calls = assistant_payload.get("tool_calls")
        print(
            f"[DIAG][{sample_id}] step {action_index}: "
            f"content={str(raw_content)[:120]!r}  "
            f"tool_calls={'YES (' + str(len(raw_tool_calls)) + ')' if raw_tool_calls else 'none'}",
            flush=True,
        )

        tool_calls = assistant_payload.get("tool_calls") or []

        if tool_calls:
            tool_call = tool_calls[0]
            function_block = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            tool_name = _normalize_string(function_block.get("name"))
            raw_arguments = function_block.get("arguments", "{}")
            arguments = _safe_json_loads(raw_arguments)

            observation, reward, terminated, truncated, info = env.env_step({"name": tool_name, "params": arguments})
            step_record = env.trajectory[-1] if env.trajectory else {}
            state_diff = step_record.get("state_diff", {}) if isinstance(step_record, dict) else {}
            observation_text = _observation_to_text(observation)

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_payload.get("content"),
                    "tool_calls": [tool_call],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", f"call_{action_index}"),
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
            continue

        # No tool call: treat as text response
        assistant_text = _normalize_string(assistant_payload.get("content"))
        if not assistant_text:
            error_message = f"model produced neither tool call nor text at step {action_index}"
            print(
                f"[DIAG][{sample_id}] step {action_index} EMPTY OUTPUT — "
                f"raw content={assistant_payload.get('content')!r}  "
                f"Possible causes: base model (not instruct), wrong model-family, "
                f"model generates EOS immediately (quantisation issue).",
                flush=True,
            )
            break

        messages.append({"role": "assistant", "content": assistant_text})
        transcript.append({"role": "assistant", "content": assistant_text})
        actual_actions.append({"action_index": action_index, "kind": "respond", "content": assistant_text})

        if next_user_turn_index < len(user_turns):
            next_turn_msg = _normalize_string(user_turns[next_user_turn_index].get("user_message"))
            next_user_turn_index += 1
            messages.append({"role": "user", "content": next_turn_msg})
            transcript.append({"role": "user", "content": next_turn_msg})
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


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

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


def _reliability_scores(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row.get("reliability", {}).get(key, 0.0) or 0.0) for row in rows]


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    # success_rate uses relaxed_success computed inline from each row's completion dict
    relaxed_sr = _mean([
        float(_compute_relaxed_completion(row.get("completion", {})).get("relaxed_success", False))
        for row in rows
    ])
    strict_sr = _mean(
        [float(row.get("completion", {}).get("strict_sr", float(bool(row.get("completion", {}).get("strict_success", False))))) for row in rows]
    )
    return {
        "success_rate": relaxed_sr,
        "sr": relaxed_sr,
        "semantic_success_rate": relaxed_sr,
        "strict_success_rate": strict_sr,
        "avg_efficiency": _mean([float(row.get("efficiency", {}).get("efficiency", 0.0) or 0.0) for row in rows]),
        "avg_task_score": _mean([float(row.get("efficiency", {}).get("task_score", 0.0) or 0.0) for row in rows]),
        "avg_reliability": _mean(_reliability_scores(rows, "reliability")),
        "avg_faithfulness_score": _mean(_reliability_scores(rows, "faithfulness_score")),
        "avg_clarification_score": _mean(_reliability_scores(rows, "clarification_score")),
        "avg_tool_discipline_score": _mean(_reliability_scores(rows, "tool_discipline_score")),
    }


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(rows),
        **_metric_summary(rows),
        "run_error_count": sum(1 for row in rows if row.get("run", {}).get("run_error")),
    }


def _group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_normalize_string(row.get(key), "unknown")].append(row)
    return {k: _summarize_group(v) for k, v in sorted(grouped.items())}


def _group_flat_labels(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for label in (row.get(key) or []):
            normalized = _normalize_string(label)
            if normalized:
                grouped[normalized].append(row)
    return {k: _summarize_group(v) for k, v in sorted(grouped.items())}


def _as_completed_with_progress(futures: dict[Any, int]) -> Any:
    total = len(futures)
    if total <= 0:
        return
    if tqdm is not None:
        yield from tqdm(as_completed(futures), total=total, desc="Evaluating samples", unit="sample", dynamic_ncols=True)
        return
    start = time.monotonic()
    done = 0
    for future in as_completed(futures):
        done += 1
        elapsed = max(time.monotonic() - start, 1e-9)
        eta = (total - done) / (done / elapsed) if done else 0.0
        print(f"Evaluating samples: {done}/{total} [elapsed {_format_duration(elapsed)}, ETA {_format_duration(eta)}]", file=sys.stderr, flush=True)
        yield future


# ---------------------------------------------------------------------------
# Single-sample eval orchestration
# ---------------------------------------------------------------------------

def _evaluate_single_sample_local(
    index: int,
    sample: dict[str, Any],
    runtime_index: dict[str, dict[str, Any]],
    backend: LocalModelBackend,
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any] | None, str | None]:
    task_id = _normalize_string(sample.get("task_id"))
    runtime_entry = runtime_index.get(task_id)
    if runtime_entry is None:
        return index, None, task_id

    enriched = _build_enriched_sample(sample, runtime_entry)
    max_steps = args.max_steps if args.max_steps > 0 else _safe_int(sample.get("max_allowed_action_turns"), DEFAULT_EVAL_MAX_STEPS)

    try:
        run_result = _run_single_sample_local(enriched, backend, max_steps, args.temperature)
        completion = evaluate_completion(enriched, run_result)
        efficiency = evaluate_efficiency(enriched, run_result, completion.get("passed", False))
        if args.skip_reliability:
            reliability = {
                "judge_model": None,
                "faithfulness_score": 0,
                "clarification_score": 0,
                "tool_discipline_score": 0,
                "reliability": 0.0,
                "judge_error": "skipped",
            }
        else:
            reliability = evaluate_reliability(
                enriched,
                run_result,
                model=args.judge_model,
                temperature=args.judge_temperature,
                timeout=args.timeout,
                api_key=args.judge_api_key or None,
                base_url=args.judge_base_url or None,
            )
    except Exception as exc:
        run_result = {
            "sample_id": enriched.get("sample_id"),
            "transcript": [],
            "actual_actions": [],
            "actual_tool_trace": [],
            "initial_state": {},
            "final_state": {},
            "final_state_diff": {},
            "constraints_rules_text": runtime_entry.get("constraints_rules_text", ""),
            "run_error": f"{exc}\n{traceback.format_exc()}",
        }
        completion = evaluate_completion(enriched, run_result)
        efficiency = evaluate_efficiency(enriched, run_result, False)
        reliability = {
            "judge_model": args.judge_model,
            "faithfulness_score": 0,
            "clarification_score": 0,
            "tool_discipline_score": 0,
            "reliability": 0.0,
            "judge_error": "run failed before reliability scoring",
        }

    return index, {
        "sample_id": enriched.get("sample_id"),
        "task_id": enriched.get("task_id"),
        "env_id": enriched.get("env_id"),
        "variant": _normalize_variant(enriched.get("variant")),
        "difficulty_bucket": enriched.get("difficulty_bucket"),
        "task_type": enriched.get("task_type"),
        "profile_bundle": enriched.get("profile_bundle", []),
        "behavior_bundle": enriched.get("behavior_bundle", []),
        "unstable_behavior": enriched.get("unstable_behavior", ""),
        "run": _json_safe_snapshot(run_result),
        "completion": completion,
        "efficiency": efficiency,
        "reliability": reliability,
    }, None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate EnvBuilder benchmark samples using a local HuggingFace model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local path or HuggingFace model ID (e.g. Qwen/Qwen3-8B or /path/to/model)",
    )
    parser.add_argument("--benchmark-file", default=str(DEFAULT_BENCHMARK_FULL))
    parser.add_argument("--blueprint-file", default=str(DEFAULT_TASK_BLUEPRINTS))
    parser.add_argument(
        "--output-file",
        default=str(DEFAULT_EVAL_RESULTS_DIR / "local_model_eval_results.json"),
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--judge-model", default=DEFAULT_EVAL_JUDGE_MODEL, help="API-based judge model for reliability scoring")
    parser.add_argument("--judge-temperature", type=float, default=DEFAULT_EVAL_JUDGE_TEMPERATURE)
    parser.add_argument("--timeout", type=int, default=DEFAULT_EVAL_TIMEOUT, help="Timeout for API-based judge calls (seconds)")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_EVAL_MAX_SAMPLES, help="Max samples to evaluate (0 = all)")
    parser.add_argument("--variant", choices=["all", "stable", "unstable", "collab", "noncollab"], default="all")
    parser.add_argument("--sample-id", default="", help="Evaluate a single sample by ID")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_EVAL_MAX_STEPS, help="Max agent action steps per sample")
    parser.add_argument("--max-workers", type=int, default=1, help="Concurrent sample workers (recommend 1 for local GPU)")
    parser.add_argument("--skip-reliability", action="store_true", help="Skip LLM-as-judge reliability scoring")
    parser.add_argument("--judge-api-key", default="", help="API key for the reliability judge (overrides OPENAI_API_KEY)")
    parser.add_argument("--judge-base-url", default="", help="Base URL for the reliability judge (overrides OPENAI_BASE_URL)")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit (requires bitsandbytes)")
    parser.add_argument("--device-map", default="auto", help="HuggingFace device_map (auto / cpu / cuda)")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max tokens to generate per inference step")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument(
        "--model-family",
        default="",
        choices=["", "qwen3", "qwen2", "llama3", "llama", "mistral", "generic"],
        help=(
            "Override model family detection. Leave empty to auto-detect from model path. "
            "Controls chat-template kwargs and tool-call parsing format. "
            "NOTE: always use the *Instruct* variant of the model, not the base model."
        ),
    )
    return parser.parse_args()


def _diag(msg: str) -> None:
    print(f"[DIAG] {msg}", flush=True)


def main() -> int:
    args = parse_args()

    _diag(f"benchmark_file : {args.benchmark_file}")
    _diag(f"blueprint_file : {args.blueprint_file}")
    _diag(f"model_path     : {args.model_path}")
    _diag(f"max_samples    : {args.max_samples}")
    _diag(f"variant filter : {args.variant}")
    _diag(f"sample_id      : {args.sample_id!r}")
    _diag(f"max_steps      : {args.max_steps}")

    # ── 1. Load benchmark ────────────────────────────────────────────────────
    benchmark_rows = read_jsonl_file(args.benchmark_file)
    _diag(f"benchmark rows loaded : {len(benchmark_rows)}")

    selected_samples = _select_samples(benchmark_rows, args)
    _diag(f"samples after filter  : {len(selected_samples)}")
    if not selected_samples:
        print(
            "[LocalEval] ERROR: No samples selected.  Check --benchmark-file path, "
            "--variant, --sample-id, and --max-samples.",
            flush=True,
        )
        return 1
    _diag(f"first sample_id : {selected_samples[0].get('sample_id')}  "
          f"task_id : {selected_samples[0].get('task_id')}  "
          f"variant : {selected_samples[0].get('variant')}")

    # ── 2. Load runtime index ────────────────────────────────────────────────
    blueprint_payload = read_json_file(args.blueprint_file)
    runtime_index = _build_runtime_index(blueprint_payload)
    _diag(f"runtime_index task_ids: {len(runtime_index)}")

    matched = sum(1 for s in selected_samples if runtime_index.get(_normalize_string(s.get("task_id"))))
    _diag(f"selected samples with runtime entry : {matched} / {len(selected_samples)}")
    if matched == 0:
        sample0_tid = _normalize_string(selected_samples[0].get("task_id"))
        _diag(f"first sample task_id '{sample0_tid}' NOT in runtime_index")
        _diag(f"runtime_index sample keys: {list(runtime_index.keys())[:5]}")
        print(
            "[LocalEval] ERROR: No selected samples match any task_id in the blueprint. "
            "Check that --blueprint-file corresponds to --benchmark-file.",
            flush=True,
        )
        return 1

    # ── 3. Load model ────────────────────────────────────────────────────────
    backend = LocalModelBackend(
        model_path=args.model_path,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        trust_remote_code=args.trust_remote_code,
        model_family=args.model_family,
    )
    backend.load()
    _diag(f"model loaded  device={backend._device}  family={backend.model_family}")

    # ── 4. Smoke-test the chat template on sample 0 ─────────────────────────
    _diag("running chat-template smoke test on first sample ...")
    try:
        s0 = selected_samples[0]
        tid0 = _normalize_string(s0.get("task_id"))
        rt0 = runtime_index.get(tid0, {})
        tools0 = _convert_tools_for_openai(merge_discoverability_tools(rt0.get("tools", [])))
        user_turns0 = s0.get("user_turns") or []
        first_msg = _normalize_string(user_turns0[0].get("user_message")) if user_turns0 else _normalize_string(s0.get("user_request"))
        msgs0 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": first_msg},
        ]
        converted0 = _convert_messages_for_qwen(msgs0)
        template_kwargs0: dict[str, Any] = {}
        if backend.model_family == "qwen3":
            template_kwargs0["enable_thinking"] = False
        prompt0 = backend.tokenizer.apply_chat_template(
            converted0,
            tools=_convert_tools_for_qwen(tools0) if tools0 else None,
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs0,
        )
        _diag(f"chat template OK — prompt length {len(prompt0)} chars")
        _diag(f"prompt tail (last 300 chars): ...{prompt0[-300:]!r}")
    except Exception as tmpl_exc:
        _diag(f"chat template FAILED: {tmpl_exc}")
        traceback.print_exc()
        print(
            "[LocalEval] ERROR: apply_chat_template raised an exception. "
            "See traceback above.",
            flush=True,
        )
        return 1

    # ── 5. Smoke-test one inference step ─────────────────────────────────────
    _diag("running one inference step on first sample ...")
    try:
        out0 = backend.chat_with_tools(msgs0, tools0, temperature=0.0)
        _diag(f"inference OK — content={out0.get('content')!r}  tool_calls={out0.get('tool_calls')}")
        if not out0.get("content") and not out0.get("tool_calls"):
            _diag(
                "WARNING: model produced neither text nor tool calls on the first step. "
                "Likely causes: "
                "(1) base model instead of instruct model, "
                "(2) wrong model family — try --model-family llama3, "
                "(3) model generates EOS immediately (quantisation issue)."
            )
    except Exception as inf_exc:
        _diag(f"inference FAILED: {inf_exc}")
        traceback.print_exc()
        return 1

    print(f"[LocalEval] Evaluating {len(selected_samples)} samples with model {args.model_path!r}", flush=True)

    per_sample_results: list[dict[str, Any]] = []
    missing_runtime_task_ids: list[str] = []
    ordered_results: dict[int, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {
            executor.submit(
                _evaluate_single_sample_local, index, sample, runtime_index, backend, args
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
        "env_summary": _group_rows(per_sample_results, "env_id"),
        "difficulty_summary": _group_rows(per_sample_results, "difficulty_bucket"),
        "task_type_summary": _group_rows(per_sample_results, "task_type"),
        "behavior_summary": _group_flat_labels(per_sample_results, "behavior_bundle"),
        "unstable_behavior_summary": _group_rows(per_sample_results, "unstable_behavior"),
        "profile_summary": _group_flat_labels(per_sample_results, "profile_bundle"),
    }

    output_payload = {
        "metadata": {
            "generated_at": _now_iso(),
            "benchmark_file": str(args.benchmark_file),
            "blueprint_file": str(args.blueprint_file),
            "model": args.model_path,
            "model_backend": "local_hf",
            "dtype": args.dtype,
            "load_in_4bit": args.load_in_4bit,
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
