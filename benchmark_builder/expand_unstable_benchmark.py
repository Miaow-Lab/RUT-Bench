"""
Expand the unstable benchmark to full 6-type coverage.

For every stable sample in benchmark_full.jsonl, create one unstable sample
per behavior type (6 types x N stable = 6N unstable total).  Each rewrite is
done by an LLM using the same prompt schema as stage4.

Output
------
  benchmark_unstable_expanded.jsonl   -- all 6*N unstable samples
  benchmark_full_expanded.jsonl       -- N stable + 6*N unstable

Checkpointing
-------------
Completed samples are appended to a .ckpt.jsonl sidecar as soon as they are
done.  On restart the script skips any (sample_id, behavior) pair already in
the checkpoint, so it can be interrupted and resumed safely.

Usage (from project root)
--------------------------
python benchmark_builder/expand_unstable_benchmark.py \\
    --input-file  benchmark_builder/output/run64/benchmark_full.jsonl \\
    --output-dir  benchmark_builder/output/run64 \\
    --model       gpt-5.4 \\
    --max-workers 8 \\
    --temperature 0.7
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from benchmark_builder.common.llm_io import call_json_object_llm

# ── Behavior definitions (same as stage4) ────────────────────────────────────

BEHAVIORS = [
    "underspecification",
    "information_overload",
    "fabricated_parameters",
    "goal_switching",
    "contradictory_constraints",
    "impatience_and_hostility",
]

BEHAVIOR_SUMMARIES = {
    "underspecification": (
        "Remove essential slots or details from the stable turn while keeping "
        "the original task recoverable through context or clarification."
    ),
    "information_overload": (
        "Fuse the stable request with tangential context, redundant details, or "
        "nested side information while preserving the core task."
    ),
    "fabricated_parameters": (
        "Add a natural-language reference to unsupported pseudo-parameters, "
        "nonexistent options, or unavailable capabilities without changing the "
        "true ground truth."
    ),
    "goal_switching": (
        "Inject an interruption, side goal, or mid-session redirection in the "
        "same environment context while preserving the original task identity."
    ),
    "contradictory_constraints": (
        "Add mutually incompatible or hard-to-satisfy constraints that require "
        "clarification or scoping instead of blind execution."
    ),
    "impatience_and_hostility": (
        "Rewrite the stable turn with impatience, blame, pressure, or rude tone "
        "while preserving the underlying semantic goal."
    ),
}

# Deterministic fallback prefixes when LLM fails
BEHAVIOR_FALLBACKS = {
    "underspecification": lambda msg: f"Can you take care of this? {msg.split('.')[0]}.",
    "information_overload": lambda msg: (
        f"I have a lot going on today and there are several related background "
        f"issues, but the main thing is: {msg} Also keep in mind the broader "
        f"context may matter."
    ),
    "fabricated_parameters": lambda msg: (
        f"{msg} If there is a priority flag, instant override, or auto-resolve "
        f"mode available, please use that too."
    ),
    "goal_switching": lambda msg: (
        f"I keep switching between tasks. {msg} Actually I was about to ask "
        f"about something else too, but handle this first."
    ),
    "contradictory_constraints": lambda msg: (
        f"I need this done, but somehow I do not want anything else disrupted. "
        f"{msg} I know that sounds contradictory — just do the best you can."
    ),
    "impatience_and_hostility": lambda msg: (
        f"I cannot believe this still is not handled. {msg} Please stop wasting "
        f"my time and just get it done."
    ),
}

SYSTEM_PROMPT = """\
You are an unstable user dialogue rewriter for an agent tool-calling benchmark.

Your job is to rewrite stable user turns into unstable user turns for the exact same task.

Rules:
1. Preserve the exact underlying task. The rewrite must stay paired with the stable sample for the same task.
2. Keep the same number of user turns and the same turn ordering.
3. Rewrite only the user-facing messages. Do not change the task_summary, expected_outcome, linked tool-step alignment, tool plan, or final target state.
4. Inject only the requested unstable behavior as controlled friction on top of the stable baseline.
5. Do not mention tool names, internal IDs, hidden schema keys, or implementation details.
6. Do not replace the original task with a new mandatory task. Any noisy side remark must still leave the original task recoverable directly or through clarification.
7. Do not leak the oracle trace, backend schema, hidden state, or tool implementation details.
8. Output JSON only.

Return an object with this schema:
{
  "rewritten_user_turns": [
    {
      "turn_id": 1,
      "user_message": "unstable rewrite of the baseline turn"
    }
  ]
}"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _norm(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip() or default


def _build_user_prompt(sample: dict[str, Any], behavior: str) -> str:
    meta = sample.get("metadata", {}) or {}
    env_intro = _norm(meta.get("environment_introduction") or meta.get("environment_summary"))
    goal = _norm(sample.get("goal_summary"))
    mode = _norm(sample.get("dialogue_mode"))
    alignment = sample.get("trace_to_turn_alignment") or []

    turns_payload = [
        {
            "turn_id": t.get("turn_id"),
            "task_summary": t.get("task_summary", ""),
            "expected_outcome": t.get("expected_outcome", ""),
            "baseline_user_message": t.get("user_message", ""),
        }
        for t in (sample.get("user_turns") or [])
        if isinstance(t, dict)
    ]

    behavior_line = f"- {behavior}: {BEHAVIOR_SUMMARIES[behavior]}"

    return (
        f"Environment:\n{env_intro}\n\n"
        f"Goal summary:\n{goal}\n\n"
        f"Dialogue mode:\n{mode}\n\n"
        f"Unstable behavior to inject:\n{behavior_line}\n\n"
        f"Turn-level sub-goal and trace alignment (context only — do not leak tool names or IDs):\n"
        f"{json.dumps(alignment, ensure_ascii=False, indent=2, default=str)}\n\n"
        f"Baseline stable turns to rewrite:\n"
        f"{json.dumps(turns_payload, ensure_ascii=False, indent=2)}\n\n"
        f"Rewrite each baseline_user_message with the requested behavior."
    )


def _fallback_turns(sample: dict[str, Any], behavior: str) -> list[dict[str, Any]]:
    fn = BEHAVIOR_FALLBACKS[behavior]
    result = []
    for turn in (sample.get("user_turns") or []):
        if not isinstance(turn, dict):
            continue
        new_turn = deepcopy(turn)
        original = _norm(turn.get("user_message"))
        new_turn["user_message"] = fn(original) if original else original
        result.append(new_turn)
    return result


def _rewrite_turns(
    sample: dict[str, Any],
    behavior: str,
    model: str,
    temperature: float,
    timeout: int,
    api_key: str | None,
    base_url: str | None,
) -> list[dict[str, Any]]:
    """Call LLM to rewrite user turns; return fallback on any error."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(sample, behavior)},
    ]
    try:
        result = call_json_object_llm(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            api_key=api_key or None,
            base_url=base_url or None,
        )
    except Exception as exc:
        print(f"  [LLM error] {sample['sample_id']} / {behavior}: {exc}", flush=True)
        return _fallback_turns(sample, behavior)

    if not isinstance(result, dict):
        return _fallback_turns(sample, behavior)

    raw_turns = result.get("rewritten_user_turns")
    if not isinstance(raw_turns, list):
        return _fallback_turns(sample, behavior)

    stable_turns = sample.get("user_turns") or []
    if len(raw_turns) != len(stable_turns):
        return _fallback_turns(sample, behavior)

    rewritten: list[dict[str, Any]] = []
    for baseline, rw in zip(stable_turns, raw_turns):
        if not isinstance(rw, dict):
            return _fallback_turns(sample, behavior)
        msg = _norm(rw.get("user_message"))
        if not msg:
            return _fallback_turns(sample, behavior)
        merged = deepcopy(baseline)
        merged["user_message"] = msg
        rewritten.append(merged)

    # Must have actually changed at least one turn
    changed = any(
        rewritten[i]["user_message"] != _norm((stable_turns[i] or {}).get("user_message"))
        for i in range(len(rewritten))
    )
    if not changed:
        return _fallback_turns(sample, behavior)

    return rewritten


def _build_unstable_sample(
    stable: dict[str, Any],
    behavior: str,
    rewritten_turns: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    sample = deepcopy(stable)
    base_id = stable["sample_id"].replace("_stable", "")
    sample["sample_id"] = f"{base_id}_{behavior}"
    sample["variant"] = "unstable"
    sample["unstable_behavior"] = behavior
    sample["behavior_bundle"] = [behavior]
    sample["profile_bundle"] = ["unstable_taxonomy_v1"]
    # Preserve original stable turns
    sample["stable_user_turns"] = deepcopy(stable.get("user_turns") or [])
    sample["user_turns"] = rewritten_turns
    sample["rewrite_metadata"] = {
        "source_turn_id": None,
        "operator": "six_taxonomy_behavior_rewrite",
        "conditioning_context": [
            "env_summary", "admissible_operations", "sub_goal",
            "sub_trace", "stable_utterance", "dialogue_context",
        ],
        "behavior_bundle": [behavior],
        "task_identity_preserved": True,
        "turn_structure_preserved": True,
        "source_sample_id": stable["sample_id"],
        "rewritten_at": _now_iso(),
        "llm_model": model,
    }
    # Keep metadata consistent with unstable variant
    if isinstance(sample.get("metadata"), dict):
        sample["metadata"]["source_variant"] = "stable"
        sample["metadata"]["task_semantics_preserved"] = True
        sample["metadata"]["trajectory_mode"] = "oracle_replay_from_blueprint"
    return sample


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint(path: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Returns (samples_by_key, done_keys) where key = f'{sample_id}|{behavior}'."""
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done, set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
                key = f"{s.get('rewrite_metadata', {}).get('source_sample_id', '')}|{s.get('unstable_behavior', '')}"
                done[key] = s
            except Exception:
                pass
    return done, set(done.keys())


_ckpt_lock = threading.Lock()


def _append_checkpoint(path: Path, sample: dict[str, Any]) -> None:
    with _ckpt_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


# ── Progress ──────────────────────────────────────────────────────────────────

def _format_dur(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand unstable benchmark to 6 behaviors x N stable samples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-file",
        default="benchmark_builder/output/run64/benchmark_full.jsonl",
        help="Source benchmark_full.jsonl containing stable samples",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_builder/output/run64",
        help="Directory for output and checkpoint files",
    )
    parser.add_argument(
        "--output-unstable",
        default="benchmark_unstable_expanded.jsonl",
        help="Filename for expanded unstable-only output",
    )
    parser.add_argument(
        "--output-full",
        default="benchmark_full_expanded.jsonl",
        help="Filename for full benchmark (stable + expanded unstable)",
    )
    parser.add_argument("--model", default="gpt-5.4", help="LLM model for rewriting")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel LLM call workers")
    parser.add_argument("--api-key", default="", help="Override OPENAI_API_KEY")
    parser.add_argument("--base-url", default="", help="Override OPENAI_BASE_URL")
    parser.add_argument("--behaviors", default=",".join(BEHAVIORS), help="Comma-separated behavior list")
    parser.add_argument("--max-stable", type=int, default=0, help="Limit stable samples for smoke test (0=all)")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore checkpoint and start fresh")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_file)
    unstable_path = output_dir / args.output_unstable
    full_path = output_dir / args.output_full
    ckpt_path = output_dir / (args.output_unstable.replace(".jsonl", ".ckpt.jsonl"))

    behaviors = [b.strip() for b in args.behaviors.split(",") if b.strip()]
    api_key = args.api_key or None
    base_url = args.base_url or None

    # Load stable samples
    all_samples: list[dict[str, Any]] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_samples.append(json.loads(line))

    stable_samples = [s for s in all_samples if s.get("variant") == "stable"]
    if args.max_stable > 0:
        stable_samples = stable_samples[: args.max_stable]

    n_stable = len(stable_samples)
    total_jobs = n_stable * len(behaviors)
    print(f"[Expand] Stable samples: {n_stable}, Behaviors: {len(behaviors)}, Total jobs: {total_jobs}")
    print(f"[Expand] Behaviors: {behaviors}")
    print(f"[Expand] Checkpoint: {ckpt_path}")

    # Load checkpoint
    if args.resume and ckpt_path.exists():
        ckpt_samples, done_keys = _load_checkpoint(ckpt_path)
        print(f"[Expand] Resuming — {len(done_keys)} jobs already done in checkpoint.")
    else:
        ckpt_samples, done_keys = {}, set()
        if ckpt_path.exists() and not args.resume:
            ckpt_path.unlink()
            print("[Expand] Checkpoint cleared (--no-resume).")

    # Build work list
    work: list[tuple[dict[str, Any], str]] = []
    for behavior in behaviors:
        for stable in stable_samples:
            key = f"{stable['sample_id']}|{behavior}"
            if key not in done_keys:
                work.append((stable, behavior))

    print(f"[Expand] Remaining jobs: {len(work)} / {total_jobs}")
    if not work:
        print("[Expand] Nothing to do — all jobs already in checkpoint.")
    else:
        start = time.monotonic()
        completed = 0

        def do_job(stable: dict[str, Any], behavior: str) -> dict[str, Any]:
            turns = _rewrite_turns(
                stable, behavior,
                model=args.model,
                temperature=args.temperature,
                timeout=args.timeout,
                api_key=api_key,
                base_url=base_url,
            )
            return _build_unstable_sample(stable, behavior, turns, args.model)

        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            futures = {executor.submit(do_job, s, b): (s["sample_id"], b) for s, b in work}
            for future in as_completed(futures):
                stable_id, behavior = futures[future]
                try:
                    result = future.result()
                    _append_checkpoint(ckpt_path, result)
                    key = f"{stable_id}|{behavior}"
                    ckpt_samples[key] = result
                except Exception as exc:
                    print(f"  [ERROR] {stable_id}/{behavior}: {exc}", flush=True)

                completed += 1
                elapsed = max(time.monotonic() - start, 1e-9)
                rate = completed / elapsed
                remaining = len(work) - completed
                eta = remaining / rate if rate > 0 else 0.0
                print(
                    f"\r[Expand] {completed}/{len(work)} "
                    f"elapsed={_format_dur(elapsed)} eta={_format_dur(eta)} "
                    f"({rate:.1f}/s)",
                    end="", flush=True,
                )

        print(flush=True)

    # Assemble final unstable samples ordered by (behavior, stable order)
    stable_order = {s["sample_id"]: i for i, s in enumerate(stable_samples)}
    unstable_samples: list[dict[str, Any]] = []
    for behavior in behaviors:
        for stable in stable_samples:
            key = f"{stable['sample_id']}|{behavior}"
            if key in ckpt_samples:
                unstable_samples.append(ckpt_samples[key])

    # Write benchmark_unstable_expanded.jsonl
    with open(unstable_path, "w", encoding="utf-8") as f:
        for s in unstable_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Write benchmark_full_expanded.jsonl (stable + all unstable)
    # Order: interleave stable + its 6 unstable variants together for readability
    full_samples: list[dict[str, Any]] = []
    for stable in stable_samples:
        full_samples.append(stable)
        for behavior in behaviors:
            key = f"{stable['sample_id']}|{behavior}"
            if key in ckpt_samples:
                full_samples.append(ckpt_samples[key])

    with open(full_path, "w", encoding="utf-8") as f:
        for s in full_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Summary
    from collections import Counter
    behavior_dist = Counter(s.get("unstable_behavior", "") for s in unstable_samples)
    print(f"\n[Expand] Done.")
    print(f"  Unstable samples total : {len(unstable_samples)}")
    print(f"  Full benchmark total   : {len(full_samples)}")
    print(f"  Unstable output        : {unstable_path}")
    print(f"  Full output            : {full_path}")
    print(f"  Behavior distribution:")
    for b in behaviors:
        print(f"    {b}: {behavior_dist.get(b, 0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
