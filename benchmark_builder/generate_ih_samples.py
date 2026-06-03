"""
Generate Impatience and Hostility (IH) unstable samples from stable benchmark samples.

Reads benchmark_full.jsonl, randomly selects 10 stable samples, rewrites their
user turns with impatience/blame/high-pressure tones using an LLM, and saves
the results as benchmark_ih.jsonl in the same format.
"""
import argparse
import json
import random
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load env vars from project root before any LLM imports
from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from benchmark_builder.common.llm_io import call_json_object_llm
from utils.tool_parser import extract_json_from_text

DEFAULT_INPUT = PROJECT_ROOT / "benchmark_builder" / "output" / "run64" / "benchmark_full.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark_builder" / "output" / "run64" / "benchmark_ih.jsonl"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SAMPLE_COUNT = 10
DEFAULT_SEED = 42

SYSTEM_PROMPT = """
You are a dialogue rewriter for an agent tool-calling benchmark.

Your task is to rewrite stable user utterances into impatient, hostile, or high-pressure versions for the SAME underlying task.

Definition of Impatience and Hostility:
Rewrite the stable user turn(s) with impatience, blame, or high-pressure tone so that the messages become more abrasive and oppressive. The semantic goal and task identity must be preserved exactly — only the tone and style change.

Rules:
1. The underlying task (what the user wants done) must remain identical. Do not add new requirements or change the task's parameters.
2. Keep the same number of user turns and their ordering.
3. Make the rewritten messages sound frustrated, impatient, blaming, demanding, or pressuring. Use expressions of urgency, irritation, blame, or displeasure.
4. Do not mention internal tool names, IDs, schema keys, or backend implementation details.
5. The original task must still be recoverable from the rewritten message (no underspecification).
6. Output JSON only.

Return an object with this exact schema:
{
  "rewritten_user_turns": [
    {
      "turn_id": <integer matching the original>,
      "user_message": "<impatient/hostile rewrite of the baseline turn>"
    }
  ]
}
""".strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip() or default


def _build_user_prompt(sample: dict[str, Any]) -> str:
    env_intro = _normalize(sample.get("metadata", {}).get("environment_introduction") or sample.get("metadata", {}).get("environment_summary", ""))
    goal = _normalize(sample.get("goal_summary", ""))
    dialogue_mode = _normalize(sample.get("dialogue_mode", ""))

    turns_payload = []
    for turn in sample.get("user_turns", []):
        if not isinstance(turn, dict):
            continue
        turns_payload.append({
            "turn_id": turn.get("turn_id"),
            "task_summary": turn.get("task_summary", ""),
            "expected_outcome": turn.get("expected_outcome", ""),
            "baseline_user_message": turn.get("user_message", ""),
        })

    alignment = sample.get("trace_to_turn_alignment", [])

    return f"""Environment:
{env_intro}

Goal summary:
{goal}

Dialogue mode:
{dialogue_mode}

Unstable behavior to inject:
- impatience_and_hostility: Rewrite the stable turn with impatience, blame, pressure, or rude tone while preserving the underlying semantic goal exactly.

Turn-level sub-goal and trace alignment (for context only — do not leak tool names or IDs):
{json.dumps(alignment, ensure_ascii=False, indent=2, default=str)}

Baseline stable turns to rewrite:
{json.dumps(turns_payload, ensure_ascii=False, indent=2)}

Rewrite each baseline_user_message into an impatient/hostile/high-pressure message for the same task.
""".strip()


def _fallback_rewrite(stable_message: str) -> str:
    return f"I can't believe this still isn't done. {stable_message} I need this resolved immediately — stop wasting my time."


def _rewrite_turns_with_llm(
    sample: dict[str, Any],
    model: str,
    temperature: float,
    timeout: int,
) -> list[dict[str, Any]] | None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(sample)},
    ]

    try:
        result = call_json_object_llm(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
        )
    except Exception as exc:
        print(f"  [LLM error] {exc}")
        return None

    if not isinstance(result, dict):
        print(f"  [LLM] unexpected output type: {type(result)}")
        return None

    rewritten = result.get("rewritten_user_turns")
    if not isinstance(rewritten, list):
        print(f"  [LLM] missing rewritten_user_turns in response")
        return None

    baseline_turns = sample.get("user_turns", [])
    if len(rewritten) != len(baseline_turns):
        print(f"  [LLM] turn count mismatch: got {len(rewritten)}, expected {len(baseline_turns)}")
        return None

    # Validate and merge
    normalized: list[dict[str, Any]] = []
    for pos, (baseline, rewritten_turn) in enumerate(zip(baseline_turns, rewritten), 1):
        if not isinstance(rewritten_turn, dict):
            print(f"  [LLM] turn {pos} is not a dict")
            return None
        msg = _normalize(rewritten_turn.get("user_message"))
        if not msg:
            print(f"  [LLM] turn {pos} has empty user_message")
            return None
        # Ensure message actually changed
        if msg == _normalize(baseline.get("user_message")):
            print(f"  [LLM] turn {pos} unchanged, using fallback prefix")
            msg = _fallback_rewrite(msg)

        merged = deepcopy(baseline)
        merged["user_message"] = msg
        normalized.append(merged)

    return normalized


def _build_ih_sample(stable_sample: dict[str, Any], rewritten_turns: list[dict[str, Any]], model: str) -> dict[str, Any]:
    sample = deepcopy(stable_sample)

    # Update identity fields
    original_id = stable_sample["sample_id"]
    sample["sample_id"] = original_id.replace("_stable", "_ih")
    sample["variant"] = "unstable"
    sample["unstable_behavior"] = "impatience_and_hostility"
    sample["behavior_bundle"] = ["impatience_and_hostility"]
    sample["profile_bundle"] = ["unstable_taxonomy_v1"]

    # Preserve stable turns, replace user_turns with rewritten
    sample["stable_user_turns"] = deepcopy(stable_sample.get("user_turns", []))
    sample["user_turns"] = rewritten_turns

    # Set rewrite metadata
    sample["rewrite_metadata"] = {
        "source_turn_id": None,
        "operator": "impatience_and_hostility_rewrite",
        "conditioning_context": ["env_summary", "sub_goal", "sub_trace", "stable_utterance"],
        "behavior_bundle": ["impatience_and_hostility"],
        "task_identity_preserved": True,
        "turn_structure_preserved": True,
        "source_sample_id": original_id,
        "rewritten_at": _now_iso(),
        "llm_model": model,
    }

    # Update metadata for unstable variant tracking
    if isinstance(sample.get("metadata"), dict):
        sample["metadata"]["source_variant"] = "stable"
        sample["metadata"]["task_semantics_preserved"] = True
        sample["metadata"]["trajectory_mode"] = "oracle_replay_from_blueprint"

    return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Impatience and Hostility unstable samples from stable benchmark.")
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT), help="Path to benchmark_full.jsonl")
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT), help="Path to benchmark_ih.jsonl output")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model for rewriting")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="LLM temperature")
    parser.add_argument("--count", type=int, default=DEFAULT_SAMPLE_COUNT, help="Number of stable samples to select")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for sample selection")
    parser.add_argument("--timeout", type=int, default=120, help="LLM timeout in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load all samples
    print(f"[IH] Reading benchmark from: {args.input_file}")
    all_samples: list[dict[str, Any]] = []
    with open(args.input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_samples.append(json.loads(line))

    stable_samples = [s for s in all_samples if s.get("variant") == "stable"]
    print(f"[IH] Total samples: {len(all_samples)}, Stable: {len(stable_samples)}")

    if len(stable_samples) < args.count:
        print(f"[IH] Warning: requested {args.count} samples but only {len(stable_samples)} stable samples available.")
        args.count = len(stable_samples)

    # Random selection
    rng = random.Random(args.seed)
    selected = rng.sample(stable_samples, args.count)
    print(f"[IH] Selected {len(selected)} stable samples for IH rewriting (seed={args.seed}):")
    for s in selected:
        print(f"  - {s['sample_id']}  [{s.get('dialogue_mode', '?')} / {s.get('difficulty_bucket', '?')}]")

    # Rewrite and build IH samples
    ih_samples: list[dict[str, Any]] = []
    for idx, sample in enumerate(selected, 1):
        print(f"\n[IH] Processing {idx}/{len(selected)}: {sample['sample_id']}")
        rewritten_turns = _rewrite_turns_with_llm(
            sample,
            model=args.model,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        if rewritten_turns is None:
            print(f"  [IH] LLM rewrite failed, using deterministic fallback for all turns.")
            rewritten_turns = []
            for turn in sample.get("user_turns", []):
                fallback = deepcopy(turn)
                fallback["user_message"] = _fallback_rewrite(_normalize(turn.get("user_message", "")))
                rewritten_turns.append(fallback)

        ih_sample = _build_ih_sample(sample, rewritten_turns, args.model)
        ih_samples.append(ih_sample)

        # Print first rewritten message for quick review
        if rewritten_turns:
            original_msg = sample.get("user_turns", [{}])[0].get("user_message", "")
            rewritten_msg = rewritten_turns[0].get("user_message", "")
            print(f"  Original : {original_msg[:120]}")
            print(f"  Rewritten: {rewritten_msg[:120]}")

    # Write output
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in ih_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n[IH] Done. Wrote {len(ih_samples)} IH samples to: {output_path}")
    print(f"[IH] All samples use variant=unstable, unstable_behavior=impatience_and_hostility")


if __name__ == "__main__":
    main()
