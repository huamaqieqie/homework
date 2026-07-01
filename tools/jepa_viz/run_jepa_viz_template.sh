#!/usr/bin/env bash
set -euo pipefail

# Generic JEPA visualization template.
# Defaults write under this folder's output/ directory. Override paths as needed.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON=${PYTHON:-python}

JEPA_VIZ_OUTPUT_ROOT=${JEPA_VIZ_OUTPUT_ROOT:-${SCRIPT_DIR}/output}

# Inputs / outputs for training curves.
TRAIN_LOG=${TRAIN_LOG:-}
TRAINING_OUT=${TRAINING_OUT:-${JEPA_VIZ_OUTPUT_ROOT}/training}
WATCH_INTERVAL=${WATCH_INTERVAL:-60}

# Inputs / outputs for latent export. This path is LeWM-compatible, while the
# resulting latent files can be visualized for any JEPA-style model.
LATENT_CONFIG=${LATENT_CONFIG:-${REPO_ROOT}/le-wm/config/train/lewm.yaml}
LATENT_CHECKPOINT=${LATENT_CHECKPOINT:-}
LATENT_SPLIT=${LATENT_SPLIT:-val}
LATENT_MAX_SAMPLES=${LATENT_MAX_SAMPLES:-1024}
LATENT_OUT=${LATENT_OUT:-${JEPA_VIZ_OUTPUT_ROOT}/latents}
LATENT_FORMAT=${LATENT_FORMAT:-npz}
LATENT_DEVICE=${LATENT_DEVICE:-cuda}
DATA_CONFIG=${DATA_CONFIG:-pusht}
DATASET_NAME=${DATASET_NAME:-}
BATCH_SIZE=${BATCH_SIZE:-}
NUM_WORKERS=${NUM_WORKERS:-}

# Inputs / outputs for latent visualization.
LATENT_DIR=${LATENT_DIR:-${LATENT_OUT}}
LATENT_VIZ_OUT=${LATENT_VIZ_OUT:-${JEPA_VIZ_OUTPUT_ROOT}/latent_viz}
LATENT_COLOR_BY=${LATENT_COLOR_BY:-source}
LATENT_MAX_POINTS=${LATENT_MAX_POINTS:-5000}
LATENT_NN_QUERIES=${LATENT_NN_QUERIES:-8}
LATENT_TOP_K=${LATENT_TOP_K:-5}

# MODE can be: plot, watch, export_latents, visualize_latents, all.
MODE=${MODE:-visualize_latents}

mkdir -p "${JEPA_VIZ_OUTPUT_ROOT}" "${TRAINING_OUT}" "${LATENT_OUT}" "${LATENT_VIZ_OUT}"

if [[ "${JEPA_VIZ_RESPECT_EXTERNAL_CACHE:-0}" != "1" ]]; then
  export XDG_CACHE_HOME=${JEPA_VIZ_OUTPUT_ROOT}/.cache
  export XDG_CONFIG_HOME=${JEPA_VIZ_OUTPUT_ROOT}/.config
  export MPLCONFIGDIR=${XDG_CACHE_HOME}/matplotlib
  export TMPDIR=${JEPA_VIZ_OUTPUT_ROOT}/tmp
else
  export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${JEPA_VIZ_OUTPUT_ROOT}/.cache}
  export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-${JEPA_VIZ_OUTPUT_ROOT}/.config}
  export MPLCONFIGDIR=${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}
  export TMPDIR=${TMPDIR:-${JEPA_VIZ_OUTPUT_ROOT}/tmp}
fi
mkdir -p "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}" "${TMPDIR}"

run_plot() {
  if [[ -z "${TRAIN_LOG}" ]]; then
    echo "TRAIN_LOG is required for MODE=plot/watch." >&2
    exit 2
  fi

  args=(
    "--log" "${TRAIN_LOG}"
    "--out" "${TRAINING_OUT}"
  )
  if [[ "${1:-}" == "watch" ]]; then
    args+=("--watch" "--interval" "${WATCH_INTERVAL}")
  fi

  "${PYTHON}" "${SCRIPT_DIR}/plot_training_curves.py" "${args[@]}"
}

run_export_latents() {
  if [[ -z "${LATENT_CHECKPOINT}" ]]; then
    echo "LATENT_CHECKPOINT is required for MODE=export_latents/all." >&2
    exit 2
  fi

  args=(
    "--config" "${LATENT_CONFIG}"
    "--checkpoint" "${LATENT_CHECKPOINT}"
    "--split" "${LATENT_SPLIT}"
    "--max-samples" "${LATENT_MAX_SAMPLES}"
    "--out" "${LATENT_OUT}"
    "--format" "${LATENT_FORMAT}"
    "--device" "${LATENT_DEVICE}"
    "data=${DATA_CONFIG}"
  )

  if [[ -n "${DATASET_NAME}" ]]; then
    args+=("--dataset" "${DATASET_NAME}")
  fi
  if [[ -n "${BATCH_SIZE}" ]]; then
    args+=("--batch-size" "${BATCH_SIZE}")
  fi
  if [[ -n "${NUM_WORKERS}" ]]; then
    args+=("--num-workers" "${NUM_WORKERS}")
  fi

  "${PYTHON}" "${SCRIPT_DIR}/export_latents.py" "${args[@]}"
}

run_visualize_latents() {
  "${PYTHON}" "${SCRIPT_DIR}/visualize_latents.py" \
    --latent-dir "${LATENT_DIR}" \
    --out "${LATENT_VIZ_OUT}" \
    --color-by "${LATENT_COLOR_BY}" \
    --max-points "${LATENT_MAX_POINTS}" \
    --nn-queries "${LATENT_NN_QUERIES}" \
    --top-k "${LATENT_TOP_K}"
}

case "${MODE}" in
  plot)
    run_plot
    ;;
  watch)
    run_plot watch
    ;;
  export_latents)
    run_export_latents
    ;;
  visualize_latents)
    run_visualize_latents
    ;;
  all)
    run_export_latents
    run_visualize_latents
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use one of: plot, watch, export_latents, visualize_latents, all." >&2
    exit 2
    ;;
esac
