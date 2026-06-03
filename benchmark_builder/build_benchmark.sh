#!/usr/bin/env bash
# ===========================================================================
# build_benchmark.sh — Benchmark construction pipeline (Stage 0 -> Stage 5).
#
# Turns the filtered environments produced by environment_builder into the
# packaged benchmark JSONL datasets. Each stage writes into
# benchmark_builder/output/ and the next stage reads it back, so order matters.
# LLM-driven stages read credentials from the project-root .env
# (OPENAI_API_KEY / OPENAI_BASE_URL).
#
# Run from the benchmark_builder/ directory:
#   bash build_benchmark.sh
#
# Override the upstream environment data with:
#   MERGED_ENV_DATA=/path/to/merged_env_data.json bash build_benchmark.sh
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY="python -u"
OUT="output"
mkdir -p "$OUT"

# Source environment metadata produced by environment_builder (Stage 3 output).
MERGED_ENV_DATA="${MERGED_ENV_DATA:-../environment_builder/stage3_check_env/final_result/merged_env_data.json}"

# ---------------------------------------------------------------------------
# Stage 0 — Normalize merged environment metadata into a stable manifest.
# ---------------------------------------------------------------------------
$PY stage0_prepare_env_manifest.py \
    --input-file "$MERGED_ENV_DATA" \
    --output-file "$OUT/env_manifest.json"

# ---------------------------------------------------------------------------
# Stage 1 — Sample valid initial states for each environment.
# ---------------------------------------------------------------------------
$PY stage1_generate_init_state_bank.py \
    --input-file "$OUT/env_manifest.json" \
    --output-file "$OUT/init_state_bank.json" \
    --configs-per-env 2 \
    --max-workers 4

# ---------------------------------------------------------------------------
# Stage 2 — Generate task blueprints + validated gold traces.
# ---------------------------------------------------------------------------
$PY stage2_generate_task_blueprints.py \
    --input-file "$OUT/init_state_bank.json" \
    --output-file "$OUT/task_blueprints.json" \
    --tasks-per-state 1 \
    --max-workers 4

# ---------------------------------------------------------------------------
# Stage 3 — Realize STABLE (collaborative) user dialogues.
# ---------------------------------------------------------------------------
$PY stage3_realize_stable_dialogues.py \
    --input-file "$OUT/task_blueprints.json" \
    --output-file "$OUT/stable_dialogues.json" \
    --max-workers 4

# ---------------------------------------------------------------------------
# Stage 4 — Rewrite into UNSTABLE (non-collaborative) user dialogues.
# Injects the six-behavior taxonomy on top of the stable dialogues.
# ---------------------------------------------------------------------------
$PY stage4_rewrite_unstable_dialogues.py \
    --input-file "$OUT/stable_dialogues.json" \
    --output-file "$OUT/unstable_dialogues.json" \
    --max-workers 4

# ---------------------------------------------------------------------------
# (Optional) Generate extra Impatience & Hostility unstable samples.
# Runs after Stage 5 packaging produces RUT-Bench.jsonl.
# ---------------------------------------------------------------------------
# $PY generate_ih_samples.py \
#     --input-file "$OUT/RUT-Bench.jsonl" \
#     --output-file "$OUT/benchmark_ih.jsonl"

# ---------------------------------------------------------------------------
# Stage 5 — Quality-check, pair, and package the benchmark JSONL datasets.
# Emits benchmark_stable.jsonl, benchmark_unstable.jsonl, RUT-Bench.jsonl
# (RUT-Bench.jsonl is the full packaged benchmark consumed by eval.sh).
# ---------------------------------------------------------------------------
$PY stage5_verify_and_package.py \
    --blueprint-file "$OUT/task_blueprints.json" \
    --stable-file "$OUT/stable_dialogues.json" \
    --unstable-file "$OUT/unstable_dialogues.json" \
    --output-stable-file "$OUT/benchmark_stable.jsonl" \
    --output-unstable-file "$OUT/benchmark_unstable.jsonl" \
    --output-full-file "$OUT/RUT-Bench.jsonl" \
    --report-file "$OUT/benchmark_package_report.json"

echo "[build_benchmark] Benchmark packaged at $OUT/RUT-Bench.jsonl"
