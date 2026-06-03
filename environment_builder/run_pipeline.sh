#!/usr/bin/env bash
# ===========================================================================
# run_pipeline.sh — Environment synthesis pipeline (Stage 1 -> Stage 3).
#
# Builds and filters executable tool-call environments from raw task data.
# Each step persists its output to disk and the next step reads it back, so
# the steps must run in order. Most steps drive an LLM; configure credentials
# in the project-root .env (OPENAI_API_KEY / OPENAI_BASE_URL) before running.
#
# Run from the environment_builder/ directory:
#   bash run_pipeline.sh
#
# Tip: comment out optional steps or steps you have already completed.
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY="python -u"

# ---------------------------------------------------------------------------
# Stage 1 — Collect tasks and infer environment descriptions.
# ---------------------------------------------------------------------------
# step0: extract & clean candidate tasks from API-Bank / ToolAce / Dolci.
# step1: keep tasks that depend on a queryable + mutable persistent state.
# step2: keep tasks suited to temporal/spatial dependency-conflict injection.
# step3: infer each environment's summary / introduction / usefulness.
# step4/step5 are OPTIONAL (embedding + dedup/cluster selection).
$PY stage1_collect_env_from_task/step0_collect_task.py
$PY stage1_collect_env_from_task/step1_judge_stateful_query.py
$PY stage1_collect_env_from_task/step2_judge_ts_query.py
$PY stage1_collect_env_from_task/step3_infer_env_topic.py
# $PY stage1_collect_env_from_task/step4_optional_get_embedding.py
# $PY stage1_collect_env_from_task/step5_optional_select_env.py

# ---------------------------------------------------------------------------
# Stage 2 — Synthesise the environment class code.
# ---------------------------------------------------------------------------
# step1: infer the state space (entities / attributes / constraints).
# step2: turn the state space into a Python class skeleton.
# step3: infer the operation set (query vs. state-change).
# step4: generate per-operation function code.
# step5: concatenate skeleton + methods, clean imports, AST + return checks.
# step6: parse the class and emit the tool schema (final env code package).
$PY stage2_syn_env/step1_infer_state.py
$PY stage2_syn_env/step2_infer_state_code.py
$PY stage2_syn_env/step3_infer_operation.py
$PY stage2_syn_env/step4_infer_func_code.py
$PY stage2_syn_env/step5_concat.py
$PY stage2_syn_env/step6_analysis_env_class_code.py

# ---------------------------------------------------------------------------
# Stage 3 — Validate environments by rollout and filter.
# ---------------------------------------------------------------------------
# step1: generate test init configs per environment.
# step2: multi-round automatic function-call rollout (white-box check).
# step3: filter environments by pass rate and emit final metadata.
$PY stage3_check_env/step1_gen_test_config.py
$PY stage3_check_env/step2_roll_check.py
$PY stage3_check_env/step3_filter_env_by_check_result.py

echo "[run_pipeline] Environment synthesis complete."
