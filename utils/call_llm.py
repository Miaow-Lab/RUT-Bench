"""
Call LLM with different providers.
Supports both plain chat completion and native function-calling (tools API).

All provider-specific or model-specific parameters (temperature, top_p,
top_k, max_tokens, stop, thinking, etc.) are passed via **kwargs so that
callers have full flexibility without this module needing to enumerate
every possible option.
"""
import time
import os
import json
from openai import OpenAI
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ---------------------------------------------------------------------------
# Core LLM inference
# ---------------------------------------------------------------------------

def llm_inference(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> str:
    """Call LLM with different providers (plain chat completion, no function call API).

    Args:
        provider: API provider name. Currently supports "openai" (for any
                  OpenAI-compatible endpoint).
        model:    Model identifier string.
        messages: Chat messages in OpenAI format.
        **kwargs: All additional parameters are forwarded directly to the
                  provider's chat completion API (e.g. temperature, top_p,
                  top_k, max_tokens, stop, extra_body, …).
    """
    if provider == "openai":
        return openai_llm_inference(
            model,
            messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid provider: {provider}")


def openai_llm_inference(
    model: str,
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> str:
    """Call OpenAI-compatible chat completion API with retry mechanism.

    All entries in *kwargs* are passed straight through to
    ``client.chat.completions.create()``.  ``None``-valued keys are
    automatically stripped so callers don't need to guard against them.
    """
    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        base_url=base_url or os.getenv("OPENAI_BASE_URL")
    )

    # Build request payload — strip None values so the API doesn't choke
    api_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    for k, v in kwargs.items():
        if v is not None:
            api_kwargs[k] = v

    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            response = client.chat.completions.create(**api_kwargs)
            output = response.choices[0].message.content
            return output
        except Exception as e:
            wait = retries * 10 + 10
            err_str = str(e).encode("ascii", errors="replace").decode("ascii")
            print(f"[LLM] {model} error: {err_str}. Retrying in {wait}s ({retries+1}/{max_retries})...")
            time.sleep(wait)
            retries += 1
    print(f"[LLM] Failed to get response from {model} after {max_retries} retries, returning empty string")
    return ""


# ---------------------------------------------------------------------------
# Function-calling (tools) inference
# ---------------------------------------------------------------------------

def llm_inference_with_tools(
    provider: str,
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
):
    """Call LLM with native function-calling (tools) support.

    Args:
        provider: API provider name. Supports "openai" (any OpenAI-compatible
                  endpoint) and "gemini" (Google Generative Language REST API,
                  which properly preserves thought_signature for Gemini 2.5+).
        model:    Model identifier string.
        messages: Chat messages in OpenAI format (may include tool-role messages).
                  For "gemini" provider, messages may carry a "_gemini_content"
                  field to preserve thought_signature across turns.
        tools:    OpenAI-format tool definitions (list of {type, function} dicts).
                  If None or empty, behaves like a normal chat completion.
        **kwargs: Extra params forwarded to the API (temperature, thinking_budget,
                  extra_body, etc.).

    Returns:
        Plain dict in OpenAI message format.  For "gemini" provider, an extra
        "_gemini_content" field is included to preserve thought_signature.
    """
    if provider == "openai":
        # thinking_budget is not an OpenAI API parameter — strip it
        kwargs.pop("thinking_budget", None)
        return _openai_inference_with_tools(
            model,
            messages,
            tools,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    elif provider == "gemini":
        from utils.call_llm_gemini import gemini_inference_with_tools
        # extra_body and base_url are OpenAI-specific — strip them
        kwargs.pop("extra_body", None)
        return gemini_inference_with_tools(
            model,
            messages,
            tools,
            api_key=api_key,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid provider: {provider}")


def _openai_inference_with_tools(
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
):
    """OpenAI-compatible chat completion with native tool-calling.

    Uses the OpenAI SDK for URL construction and authentication, but calls
    with_raw_response to bypass Pydantic deserialisation of the response.
    This preserves vendor-specific extra fields (e.g. Gemini's thought_signature
    inside tool_call objects) that Pydantic would silently drop from model_extra.
    The SDK's maybe_transform keeps unknown keys in input message dicts, so
    thought_signature in the history is forwarded correctly on subsequent turns.
    Returns a plain dict (the raw choices[0].message JSON object).
    """
    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        base_url=base_url or os.getenv("OPENAI_BASE_URL"),
    )

    api_kwargs: Dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        api_kwargs["tools"] = tools
    for k, v in kwargs.items():
        if v is not None:
            api_kwargs[k] = v

    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            raw = client.chat.completions.with_raw_response.create(**api_kwargs)
            return json.loads(raw.content)["choices"][0]["message"]
        except Exception as e:
            wait = retries * 10 + 10
            err_str = str(e).encode("ascii", errors="replace").decode("ascii")
            print(f"[LLM] {model} error: {err_str}. Retrying in {wait}s ({retries+1}/{max_retries})...")
            time.sleep(wait)
            retries += 1
    print(f"[LLM] Failed to get response from {model} after {max_retries} retries")
    return None


# ---------------------------------------------------------------------------
# Embedding helpers (unchanged)
# ---------------------------------------------------------------------------

def openai_single_embedding_inference(model: str, text: str) -> List[float]:
    """Get embedding for a single text using OpenAI API with retry mechanism."""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            response = client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        except KeyboardInterrupt:
            print("Operation canceled by user.")
            break
        except Exception as e:
            print(f"Something wrong: {e}. Retrying in {retries*10+10} seconds...")
            time.sleep(retries * 10 + 10)
            retries += 1
    print(f"Failed to get embedding after {max_retries} retries, return empty list")
    return []


def openai_batch_embedding_inference(model: str, texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts using OpenAI API with retry mechanism."""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            response = client.embeddings.create(
                model=model,
                input=texts
            )
            return [d.embedding for d in response.data]
        except Exception as e:
            print(f"Something wrong: {e}. Retrying in {retries*10+10} seconds...")
            time.sleep(retries * 10 + 10)
            retries += 1
    print(f"Failed to get embeddings after {max_retries} retries, return empty list")
    return []


