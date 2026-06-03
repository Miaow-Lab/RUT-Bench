#!/usr/bin/env bash
# ===========================================================================
# eval.sh — Run NIU-Bench evaluation against a target model.
#
# This wraps eval/run_eval.py (API-based models) and eval/run_eval_local.py
# (local HuggingFace models). Edit the variables below or override them on the
# command line, e.g.:
#
#   MODEL=gpt-4o MAX_WORKERS=8 bash eval.sh
#   BACKEND=local MODEL_PATH=Qwen/Qwen3-8B-Instruct bash eval.sh
#
# Run from the final_code/ directory.
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Data paths — must match the folder you downloaded the benchmark into.
# ---------------------------------------------------------------------------
# BENCHMARK_FILE : the benchmark dataset, one JSON sample per line.
# BLUEPRINT_FILE : task blueprints (gold traces / expected state diffs) used by
#                  the completion + efficiency scorers.
BENCHMARK_FILE="${BENCHMARK_FILE:-eval/benchmark/RUT-Bench.jsonl}"
BLUEPRINT_FILE="${BLUEPRINT_FILE:-eval/benchmark/task_blueprints.jsonl}"

# OUTPUT_FILE : aggregated JSON report (metadata / summary / samples).
OUTPUT_FILE="${OUTPUT_FILE:-results/eval_results.json}"

# ---------------------------------------------------------------------------
# Backend selection.
# ---------------------------------------------------------------------------
# BACKEND : "api"   -> eval/run_eval.py        (OpenAI-compatible or Gemini)
#           "local" -> eval/run_eval_local.py  (local HuggingFace model)
BACKEND="${BACKEND:-api}"

# ---------------------------------------------------------------------------
# Model configuration.
# ---------------------------------------------------------------------------
# MODEL          : agent model name (api backend).
# AGENT_PROVIDER : "openai" (any OpenAI-compatible endpoint) or "gemini".
# MODEL_PATH     : HuggingFace model id / local dir (local backend only).
#                  ALWAYS use the *Instruct* variant, not the base model.
MODEL="${MODEL:-gpt-4o}"
AGENT_PROVIDER="${AGENT_PROVIDER:-openai}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B-Instruct}"

# ---------------------------------------------------------------------------
# Reliability judge (LLM-as-judge). Always an API model.
# ---------------------------------------------------------------------------
# JUDGE_MODEL      : model used to score faithfulness / clarification / tool use.
# SKIP_RELIABILITY : "1" to skip the judge entirely (faster, no judge cost).
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"
SKIP_RELIABILITY="${SKIP_RELIABILITY:-0}"

# ---------------------------------------------------------------------------
# Run controls.
# ---------------------------------------------------------------------------
# VARIANT      : sample filter — all / stable / unstable.
# MAX_SAMPLES  : cap number of samples (0 = all). Use a small value for smoke tests.
# MAX_STEPS    : max assistant tool-call steps per sample.
# MAX_WORKERS  : parallel workers (use 1 for a single local GPU).
# TEMPERATURE  : agent sampling temperature (0 = deterministic / greedy).
# THINKING_BUDGET : reasoning-token budget (-1 off; 0 disables thinking on Gemini 2.5+).
VARIANT="${VARIANT:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_STEPS="${MAX_STEPS:-50}"
MAX_WORKERS="${MAX_WORKERS:-4}"
TEMPERATURE="${TEMPERATURE:-0.0}"
THINKING_BUDGET="${THINKING_BUDGET:--1}"

# Local-model-only knobs.
# DTYPE        : bfloat16 / float16 / float32.
# LOAD_IN_4BIT : "1" to load in 4-bit via bitsandbytes (~5 GB VRAM for an 8B model).
# MAX_NEW_TOKENS : tokens generated per inference step.
DTYPE="${DTYPE:-bfloat16}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

# ---------------------------------------------------------------------------
# Build & run the command.
# ---------------------------------------------------------------------------
mkdir -p "$(dirname "$OUTPUT_FILE")"

RELIABILITY_FLAG=()
[ "$SKIP_RELIABILITY" = "1" ] && RELIABILITY_FLAG=(--skip-reliability)

if [ "$BACKEND" = "local" ]; then
    FOUR_BIT_FLAG=()
    [ "$LOAD_IN_4BIT" = "1" ] && FOUR_BIT_FLAG=(--load-in-4bit)

    python eval/run_eval_local.py \
        --model-path "$MODEL_PATH" \
        --benchmark-file "$BENCHMARK_FILE" \
        --blueprint-file "$BLUEPRINT_FILE" \
        --output-file "$OUTPUT_FILE" \
        --judge-model "$JUDGE_MODEL" \
        --variant "$VARIANT" \
        --max-samples "$MAX_SAMPLES" \
        --max-steps "$MAX_STEPS" \
        --max-workers "$MAX_WORKERS" \
        --temperature "$TEMPERATURE" \
        --dtype "$DTYPE" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        "${FOUR_BIT_FLAG[@]}" \
        "${RELIABILITY_FLAG[@]}"
else
    python eval/run_eval.py \
        --benchmark-file "$BENCHMARK_FILE" \
        --blueprint-file "$BLUEPRINT_FILE" \
        --output-file "$OUTPUT_FILE" \
        --model "$MODEL" \
        --agent-provider "$AGENT_PROVIDER" \
        --judge-model "$JUDGE_MODEL" \
        --variant "$VARIANT" \
        --max-samples "$MAX_SAMPLES" \
        --max-steps "$MAX_STEPS" \
        --max-workers "$MAX_WORKERS" \
        --temperature "$TEMPERATURE" \
        --thinking-budget "$THINKING_BUDGET" \
        "${RELIABILITY_FLAG[@]}"
fi
