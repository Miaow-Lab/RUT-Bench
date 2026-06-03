import json
from typing import Any


# ---------------------------------------------------------------------------
# Failure-mode classification prompt
#
# Classifies a single FAILED evaluation sample into one of five root causes:
#
#   R1a  Gave Up Early        – zero tool calls, or no write operations at all
#   R1b  Partial Completion   – some writes executed but ≥1 required write omitted
#   R2   Wrong Tool Order     – write-before-read precedence violated, or required
#                               read-equivalence match missing (and R1 does not apply)
#   R3   Wrong Params/Scope   – correct tools in correct order, but written values /
#                               target entity do not match the expected state diff
#   R4   Forbidden Tool       – an explicitly prohibited tool was invoked
#
# The classification replicates the deterministic Python classifier defined in
# the error-analysis pipeline.  The prompt is provided so that an LLM judge can
# reproduce the same label while also generating a human-readable rationale.
# ---------------------------------------------------------------------------


FAILURE_MODE_SYSTEM_PROMPT = """
You are a strict failure-mode classifier for tool-calling agent evaluations.

A sample has already been confirmed as FAILED.
Your task is to assign exactly ONE of the five root-cause labels below and
explain your reasoning.


CLASSIFICATION RULES  (apply in the order shown; stop at the first match)


R4  Unauthorized Tool Invocation
    Condition: forbidden_actions_passed == false
    The agent invoked at least one tool that was explicitly prohibited by the
    task constraints, regardless of whether the task was otherwise attempted.

R1a Premature Termination
    Condition: "missing required write tool usage" is in hard_failures
               AND (total_tool_calls == 0 OR write_tool_calls == 0)
    The agent made zero tool calls, or called only read-type tools and never
    executed any write operation.  The environment was not meaningfully modified.

R1b Partial Completion
    Condition: "missing required write tool usage" is in hard_failures
               AND write_tool_calls > 0
    The agent executed at least one write operation but omitted one or more
    additional required write tools.  The task was partially completed.

R2  Incorrect Tool Sequence
    Condition: "precedence constraints violated" is in hard_failures
               OR "missing required read-equivalence matches" is in hard_failures
               (and neither R4 nor R1a/R1b applies)
    The agent called a write-designated tool before satisfying required read-first
    constraints, or used a semantically equivalent but unacceptable lookup tool.
    Note: some environment tools that appear to be reads (e.g. names beginning
    with get_, search_, list_) are internally marked as write operations;
    calling them before the expected read steps triggers this label.

R3  Erroneous Parameter Assignment
    Condition: "missing required write matches" is in hard_failures
               (and none of R4, R1a, R1b, R2 applies)
    The agent followed the correct tool sequence and precedence order, but the
    parameter values written to the environment do not match the ground-truth
    expected state diff — e.g., wrong entity ID, date, time, or role scope.

INPUTS PROVIDED IN THE USER MESSAGE
  hard_failures: list of hard-failure keys from the evaluation engine
  forbidden_actions_passed: boolean
  total_tool_calls: total entries in actual_tool_trace
  write_tool_calls: entries where is_write == true
  tool_sequence: ordered list of {tool_name, is_write} for the run
  user_transcript: the conversation turns (user + assistant)
  task_goal:  one-line goal summary from the sample
  expected_state_diff: the ground-truth state changes required (if provided)
  forbidden_actions: list of prohibited actions (if any)

OUTPUT SCHEMA  (return JSON only, no markdown fences)
{
  "failure_mode": "R1a" | "R1b" | "R2" | "R3" | "R4",
  "label":        "Premature Termination" | "Partial Completion" | "Incorrect Tool Sequence"
                  | "Erroneous Parameter Assignment" | "Unauthorized Tool Invocation",
  "confidence":   "high" | "medium" | "low",
  "reason":       "one or two sentences citing the specific hard_failure key(s)
                   and the relevant tool call(s) that triggered this label",
  "key_evidence": "the single most diagnostic fact from the tool_sequence or
                   hard_failures that determined the label"
}

Important:
- Apply the rules strictly in the order listed; do not skip ahead.
- Do NOT invent failure modes beyond R1a, R1b, R2, R3, R4.
- If hard_failures is empty but the sample is marked failed, use R3 and set
  confidence to "low".
- Return JSON only.
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _format_transcript(transcript: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, item in enumerate(transcript, start=1):
        role    = str(item.get("role", "unknown")).strip() or "unknown"
        content = str(item.get("content", "")).strip()
        lines.append(f"{idx}. {role}: {content}")
    return "\n".join(lines) if lines else "<empty transcript>"


def _format_tool_sequence(actual_tool_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a compact ordered list of {step, tool_name, is_write} dicts."""
    return [
        {
            "step":      int(item.get("action_index", i + 1)),
            "tool_name": str(item.get("tool_name", "")).strip(),
            "is_write":  bool(item.get("is_write", False)),
        }
        for i, item in enumerate(actual_tool_trace)
    ]


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_failure_mode_user_prompt(
    sample: dict[str, Any],
    run_result: dict[str, Any],
    relaxed_completion: dict[str, Any],
) -> str:
    """
    Construct the user-turn payload for the failure-mode classifier.

    Parameters
    ----------
    sample              : the full sample dict (contains goal_summary,
                          expected_state_diff, forbidden_actions, …)
    run_result          : the run dict (transcript, actual_tool_trace, …)
    relaxed_completion  : the relaxed_completion sub-dict from the eval result
                          (contains original_hard_failures, forbidden_actions_passed, …)
    """
    transcript        = run_result.get("transcript", [])        if isinstance(run_result.get("transcript"), list)        else []
    actual_tool_trace = run_result.get("actual_tool_trace", []) if isinstance(run_result.get("actual_tool_trace"), list) else []

    hard_failures: list[str] = relaxed_completion.get("original_hard_failures") or []
    forbidden_actions_passed: bool = bool(relaxed_completion.get("forbidden_actions_passed", True))

    total_tool_calls = len(actual_tool_trace)
    write_tool_calls = sum(1 for t in actual_tool_trace if t.get("is_write", False))

    payload = {
        # ── Classification inputs ──────────────────────────────────────────
        "hard_failures":           hard_failures,
        "forbidden_actions_passed": forbidden_actions_passed,
        "total_tool_calls":        total_tool_calls,
        "write_tool_calls":        write_tool_calls,
        "tool_sequence":           _format_tool_sequence(actual_tool_trace),

        # ── Task context ───────────────────────────────────────────────────
        "sample_id":               sample.get("sample_id"),
        "variant":                 sample.get("variant"),
        "unstable_behavior":       sample.get("unstable_behavior") or "stable",
        "task_goal":               sample.get("goal_summary", ""),
        "expected_final_outcome":  sample.get("expected_final_outcome", ""),
        "expected_state_diff":     sample.get("expected_state_diff", {}),
        "forbidden_actions":       sample.get("forbidden_actions", []),

        # ── Conversation ───────────────────────────────────────────────────
        "user_transcript":         _format_transcript(transcript),

        # ── Relaxed-completion detail (for borderline R2/R3 cases) ─────────
        "relaxed_failures":        relaxed_completion.get("relaxed_failures") or [],
        "final_state_assertions_passed": relaxed_completion.get("final_state_assertions_passed"),
        "expected_state_diff_passed":    relaxed_completion.get("expected_state_diff_passed"),
    }

    return _json_block(payload)


# ---------------------------------------------------------------------------
# Convenience: deterministic (non-LLM) classifier for batch use
# ---------------------------------------------------------------------------

def classify_failure_mode_deterministic(
    relaxed_completion: dict[str, Any],
    actual_tool_trace: list[dict[str, Any]],
) -> dict[str, str]:
    """
    Rule-based classifier that exactly replicates the Python analysis pipeline.
    Returns {"failure_mode": "R1a"|…, "label": "…"} with no LLM call.

    Use this for bulk statistics; use the LLM prompt when a human-readable
    rationale is also required.
    """
    hf            = set(relaxed_completion.get("original_hard_failures") or [])
    forbidden_ok  = bool(relaxed_completion.get("forbidden_actions_passed", True))
    total_calls   = len(actual_tool_trace)
    write_calls   = sum(1 for t in actual_tool_trace if t.get("is_write", False))

    LABELS = {
        "R1a": "Gave Up Early",
        "R1b": "Partial Completion",
        "R2":  "Wrong Tool Order",
        "R3":  "Wrong Params or Scope",
        "R4":  "Forbidden Tool Called",
    }

    if not forbidden_ok:
        mode = "R4"
    elif "missing required write tool usage" in hf:
        mode = "R1a" if (total_calls == 0 or write_calls == 0) else "R1b"
    elif ("precedence constraints violated" in hf
          or "missing required read-equivalence matches" in hf):
        mode = "R2"
    elif "missing required write matches" in hf:
        mode = "R3"
    else:
        mode = "R3"   # fallback for edge cases with empty hard_failures

    return {"failure_mode": mode, "label": LABELS[mode]}
