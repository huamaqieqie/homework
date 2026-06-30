export SERVER_REPO=${SERVER_REPO:-/data1/Johnny/challenge/wrf/homework}
export LEWM_OUTPUT_ROOT=${LEWM_OUTPUT_ROOT:-${SERVER_REPO}/outputs}

if [ "${LEWM_RESPECT_EXTERNAL_CACHE:-0}" != "1" ]; then
  export HOME=${LEWM_OUTPUT_ROOT}/home
  export XDG_CACHE_HOME=${LEWM_OUTPUT_ROOT}/.cache
  export XDG_CONFIG_HOME=${LEWM_OUTPUT_ROOT}/.config
  export XDG_DATA_HOME=${LEWM_OUTPUT_ROOT}/.local
  export STABLE_PRETRAINING_HOME=${XDG_CACHE_HOME}/stable-pretraining
  export STABLE_PRETRAINING_CACHE_DIR=${XDG_CACHE_HOME}/stable-pretraining
  export PIP_CACHE_DIR=${XDG_CACHE_HOME}/pip
  export UV_CACHE_DIR=${XDG_CACHE_HOME}/uv
  export HF_HOME=${XDG_CACHE_HOME}/huggingface
  export HF_HUB_CACHE=${HF_HOME}/hub
  export MPLCONFIGDIR=${XDG_CACHE_HOME}/matplotlib
  export TMPDIR=${LEWM_OUTPUT_ROOT}/tmp
  export STABLEWM_HOME=${LEWM_OUTPUT_ROOT}/stable-wm
else
  export HOME=${HOME:-${LEWM_OUTPUT_ROOT}/home}
  export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${LEWM_OUTPUT_ROOT}/.cache}
  export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-${LEWM_OUTPUT_ROOT}/.config}
  export XDG_DATA_HOME=${XDG_DATA_HOME:-${LEWM_OUTPUT_ROOT}/.local}
  export STABLE_PRETRAINING_HOME=${STABLE_PRETRAINING_HOME:-${XDG_CACHE_HOME}/stable-pretraining}
  export STABLE_PRETRAINING_CACHE_DIR=${STABLE_PRETRAINING_CACHE_DIR:-${XDG_CACHE_HOME}/stable-pretraining}
  export PIP_CACHE_DIR=${PIP_CACHE_DIR:-${XDG_CACHE_HOME}/pip}
  export UV_CACHE_DIR=${UV_CACHE_DIR:-${XDG_CACHE_HOME}/uv}
  export HF_HOME=${HF_HOME:-${XDG_CACHE_HOME}/huggingface}
  export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
  export MPLCONFIGDIR=${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}
  export TMPDIR=${TMPDIR:-${LEWM_OUTPUT_ROOT}/tmp}
  export STABLEWM_HOME=${STABLEWM_HOME:-${LEWM_OUTPUT_ROOT}/stable-wm}
fi

mkdir -p \
  "${HOME}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${XDG_DATA_HOME}" \
  "${STABLE_PRETRAINING_HOME}" \
  "${STABLE_PRETRAINING_CACHE_DIR}" \
  "${PIP_CACHE_DIR}" \
  "${UV_CACHE_DIR}" \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${MPLCONFIGDIR}" \
  "${TMPDIR}" \
  "${STABLEWM_HOME}"
