#!/usr/bin/env bash
set -e

cd /data1/Johnny/challenge/wrf/homework/le-wm

PYTHON=/data1/Johnny/challenge/wrf/.venv/bin/python
OUTDIR=/manifoldai-training/johnny/challenge/success_eval/lewm
mkdir -p ${OUTDIR}

# 默认先跑 baseline epoch15。需要跑更多模型时，在命令行覆盖 POLICIES。
POLICIES=${POLICIES:-"lewm_pusht/weights_epoch_15.pt"}
SEEDS=${SEEDS:-"0"}
NUM_EVAL=${NUM_EVAL:-50}
EVAL_BUDGET=${EVAL_BUDGET:-300}

for POLICY in ${POLICIES}; do
  SAFE_POLICY=$(echo ${POLICY} | sed 's|/|_|g' | sed 's|\.pt||g')

  for SEED in ${SEEDS}; do
    echo "============================================================"
    echo "Running LEWM success eval"
    echo "POLICY=${POLICY}"
    echo "SEED=${SEED}"
    echo "NUM_EVAL=${NUM_EVAL}"
    echo "EVAL_BUDGET=${EVAL_BUDGET}"
    echo "============================================================"

    LOG=${OUTDIR}/${SAFE_POLICY}_seed${SEED}_num${NUM_EVAL}_budget${EVAL_BUDGET}_$(date +%Y%m%d_%H%M%S).log

    HYDRA_FULL_ERROR=1 ${PYTHON} eval.py \
      --config-name=pusht.yaml \
      policy=${POLICY} \
      seed=${SEED} \
      eval.num_eval=${NUM_EVAL} \
      eval.eval_budget=${EVAL_BUDGET} \
      2>&1 | tee ${LOG}
  done
done
