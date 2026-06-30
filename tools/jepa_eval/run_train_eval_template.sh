#!/usr/bin/env bash
set -euo pipefail

# Server-side template for LeWM training, evaluation, and curve plotting.
# Edit the variables in this block on the server before running.

SERVER_REPO=${SERVER_REPO:-/data1/Johnny/challenge/wrf/homework}
PYTHON=${PYTHON:-python}
LEWM_OUTPUT_ROOT=${LEWM_OUTPUT_ROOT:-${SERVER_REPO}/outputs}

# train.py is launched from the le-wm directory.
LEWM_DIR=${LEWM_DIR:-${SERVER_REPO}/le-wm}

# StableWM cache/data root. Leave empty if your server environment already sets it.
STABLEWM_HOME=${STABLEWM_HOME:-}

# Training config.
DATA_CONFIG=${DATA_CONFIG:-pusht}
DATASET_NAME=${DATASET_NAME:-}
OUTPUT_MODEL_NAME=${OUTPUT_MODEL_NAME:-lewm}
SUBDIR=${SUBDIR:-manual_run}
MAX_EPOCHS=${MAX_EPOCHS:-}
BATCH_SIZE=${BATCH_SIZE:-}
NUM_WORKERS=${NUM_WORKERS:-}

# Evaluation config.
EVAL_CONFIG=${EVAL_CONFIG:-pusht.yaml}
POLICY=${POLICY:-<checkpoint-or-policy-path-relative-to-STABLEWM_HOME>}
SEED=${SEED:-42}
NUM_EVAL=${NUM_EVAL:-50}
EVAL_BUDGET=${EVAL_BUDGET:-300}

# Logging and plotting.
RUN_ROOT=${RUN_ROOT:-${LEWM_OUTPUT_ROOT}/jepa_eval}
TRAIN_LOG=${TRAIN_LOG:-${RUN_ROOT}/logs/train_${DATA_CONFIG}_${SUBDIR}.log}
EVAL_LOG=${EVAL_LOG:-${RUN_ROOT}/logs/eval_${DATA_CONFIG}_${SUBDIR}.log}
METRICS_LOG=${METRICS_LOG:-${TRAIN_LOG}}
PLOT_OUT=${PLOT_OUT:-${RUN_ROOT}/training}
WATCH_INTERVAL=${WATCH_INTERVAL:-60}

# MODE can be: train, eval, plot, watch, all.
MODE=${MODE:-train}

mkdir -p "${RUN_ROOT}/logs" "${PLOT_OUT}"

export LEWM_OUTPUT_ROOT
if [[ "${LEWM_RESPECT_EXTERNAL_CACHE:-0}" != "1" ]]; then
  export XDG_CACHE_HOME=${LEWM_OUTPUT_ROOT}/.cache
  export XDG_CONFIG_HOME=${LEWM_OUTPUT_ROOT}/.config
  export XDG_DATA_HOME=${LEWM_OUTPUT_ROOT}/.local
  export PIP_CACHE_DIR=${XDG_CACHE_HOME}/pip
  export UV_CACHE_DIR=${XDG_CACHE_HOME}/uv
  export HF_HOME=${XDG_CACHE_HOME}/huggingface
  export HF_HUB_CACHE=${HF_HOME}/hub
  export MPLCONFIGDIR=${XDG_CACHE_HOME}/matplotlib
  export TMPDIR=${LEWM_OUTPUT_ROOT}/tmp
  export STABLEWM_HOME=${LEWM_OUTPUT_ROOT}/stable-wm
else
  export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${LEWM_OUTPUT_ROOT}/.cache}
  export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-${LEWM_OUTPUT_ROOT}/.config}
  export XDG_DATA_HOME=${XDG_DATA_HOME:-${LEWM_OUTPUT_ROOT}/.local}
  export PIP_CACHE_DIR=${PIP_CACHE_DIR:-${XDG_CACHE_HOME}/pip}
  export UV_CACHE_DIR=${UV_CACHE_DIR:-${XDG_CACHE_HOME}/uv}
  export HF_HOME=${HF_HOME:-${XDG_CACHE_HOME}/huggingface}
  export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
  export MPLCONFIGDIR=${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}
  export TMPDIR=${TMPDIR:-${LEWM_OUTPUT_ROOT}/tmp}
  export STABLEWM_HOME=${STABLEWM_HOME:-${LEWM_OUTPUT_ROOT}/stable-wm}
fi
mkdir -p \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${XDG_DATA_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${UV_CACHE_DIR}" \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${MPLCONFIGDIR}" \
  "${TMPDIR}" \
  "${STABLEWM_HOME}"

if [[ -n "${STABLEWM_HOME}" ]]; then
  export STABLEWM_HOME
fi

run_train() {
  cd "${LEWM_DIR}"

  args=(
    "data=${DATA_CONFIG}"
    "output_model_name=${OUTPUT_MODEL_NAME}"
    "subdir=${SUBDIR}"
  )

  if [[ -n "${DATASET_NAME}" ]]; then
    args+=("data.dataset.name=${DATASET_NAME}")
  fi

  if [[ -n "${MAX_EPOCHS}" ]]; then
    args+=("trainer.max_epochs=${MAX_EPOCHS}")
  fi

  if [[ -n "${BATCH_SIZE}" ]]; then
    args+=("loader.batch_size=${BATCH_SIZE}")
  fi

  if [[ -n "${NUM_WORKERS}" ]]; then
    args+=("num_workers=${NUM_WORKERS}" "loader.num_workers=${NUM_WORKERS}")
  fi

  echo "Running training. Log: ${TRAIN_LOG}"
  "${PYTHON}" train.py "${args[@]}" 2>&1 | tee "${TRAIN_LOG}"
}

run_eval() {
  cd "${LEWM_DIR}"

  echo "Running evaluation. Log: ${EVAL_LOG}"
  "${PYTHON}" eval.py \
    --config-name="${EVAL_CONFIG}" \
    "policy=${POLICY}" \
    "seed=${SEED}" \
    "eval.num_eval=${NUM_EVAL}" \
    "eval.eval_budget=${EVAL_BUDGET}" \
    2>&1 | tee "${EVAL_LOG}"
}

run_plot_once() {
  cd "${SERVER_REPO}"

  echo "Writing training plots to ${PLOT_OUT}"
  "${PYTHON}" tools/jepa_eval/plot_training_curves.py \
    --log "${METRICS_LOG}" \
    --out "${PLOT_OUT}"
}

run_plot_watch() {
  cd "${SERVER_REPO}"

  echo "Watching metrics log: ${METRICS_LOG}"
  "${PYTHON}" tools/jepa_eval/plot_training_curves.py \
    --log "${METRICS_LOG}" \
    --out "${PLOT_OUT}" \
    --watch \
    --interval "${WATCH_INTERVAL}"
}

case "${MODE}" in
  train)
    run_train
    ;;
  eval)
    run_eval
    ;;
  plot)
    run_plot_once
    ;;
  watch)
    run_plot_watch
    ;;
  all)
    run_train
    run_plot_once
    run_eval
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use one of: train, eval, plot, watch, all." >&2
    exit 2
    ;;
esac
