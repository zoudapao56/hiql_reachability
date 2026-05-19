#!/usr/bin/env bash
set -euo pipefail

# Run this script from the repository root:
#   bash run_background_experiment.sh
#
# Edit the variables below before each launch. Every launch starts one background
# training job and writes logs/PID under background_runs/.

SEED=1
ENV_NAME="antmaze-ultra-diverse-v0"
RUN_GROUP="EXP"
ALGO_TAG="HIQL"
#ALGO_TAG="HIQL"
#ALGO_TAG="HIQL_R_no_filter"
#ALGO_TAG="HIQL_R_filter"

# GPU selection. Leave empty to use the default visible GPUs.
CUDA_DEVICES=""

# Reachability modes:
#   Original HIQL:          USE_REACHABILITY=0, REACHABILITY_FILTER_POLICY=0
#   R scoring only:         USE_REACHABILITY=1, REACHABILITY_FILTER_POLICY=0
#   R filtering/intervene:  USE_REACHABILITY=1, REACHABILITY_FILTER_POLICY=1
USE_REACHABILITY=0
REACHABILITY_FILTER_POLICY=0

# AntMaze ultra defaults.
PRETRAIN_STEPS=1000002
EVAL_INTERVAL=100000
SAVE_INTERVAL=250000
BATCH_SIZE=1024
WAY_STEPS=50
REACHABILITY_HORIZON=50

P_CURRGOAL=0.2
P_TRAJGOAL=0.5
P_RANDOMGOAL=0.3
HIGH_P_RANDOMGOAL=0.3
DISCOUNT=0.99
TEMPERATURE=1
HIGH_TEMPERATURE=1
PRETRAIN_EXPECTILE=0.7
GEOM_SAMPLE=1
USE_LAYER_NORM=1
VALUE_HIDDEN_DIM=512
VALUE_NUM_LAYERS=3
USE_REP=1
POLICY_TRAIN_REP=0
REP_DIM=10
REP_TYPE="concat"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="sd${SEED}_${ENV_NAME}_${ALGO_TAG}_${STAMP}"
LOG_DIR="${ROOT_DIR}/background_runs/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

CMD=(
  python main.py
  --run_group "${RUN_GROUP}"
  --seed "${SEED}"
  --env_name "${ENV_NAME}"
  --pretrain_steps "${PRETRAIN_STEPS}"
  --eval_interval "${EVAL_INTERVAL}"
  --save_interval "${SAVE_INTERVAL}"
  --p_currgoal "${P_CURRGOAL}"
  --p_trajgoal "${P_TRAJGOAL}"
  --p_randomgoal "${P_RANDOMGOAL}"
  --discount "${DISCOUNT}"
  --temperature "${TEMPERATURE}"
  --high_temperature "${HIGH_TEMPERATURE}"
  --pretrain_expectile "${PRETRAIN_EXPECTILE}"
  --geom_sample "${GEOM_SAMPLE}"
  --use_layer_norm "${USE_LAYER_NORM}"
  --value_hidden_dim "${VALUE_HIDDEN_DIM}"
  --value_num_layers "${VALUE_NUM_LAYERS}"
  --batch_size "${BATCH_SIZE}"
  --use_rep "${USE_REP}"
  --policy_train_rep "${POLICY_TRAIN_REP}"
  --rep_dim "${REP_DIM}"
  --rep_type "${REP_TYPE}"
  --algo_name "${ALGO_TAG}"
  --use_waypoints 1
  --way_steps "${WAY_STEPS}"
  --high_p_randomgoal "${HIGH_P_RANDOMGOAL}"
  --use_reachability "${USE_REACHABILITY}"
  --reachability_filter_policy "${REACHABILITY_FILTER_POLICY}"
  --reachability_horizon "${REACHABILITY_HORIZON}"
)

printf "%q " "${CMD[@]}" > "${LOG_DIR}/command.txt"
printf "\n" >> "${LOG_DIR}/command.txt"

echo "Launching background run:"
echo "  name: ${RUN_NAME}"
echo "  log : ${LOG_DIR}/train.log"
echo "  cmd : ${LOG_DIR}/command.txt"

if [[ -n "${CUDA_DEVICES}" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" nohup "${CMD[@]}" > "${LOG_DIR}/train.log" 2>&1 &
else
  nohup "${CMD[@]}" > "${LOG_DIR}/train.log" 2>&1 &
fi

PID=$!
echo "${PID}" > "${LOG_DIR}/pid.txt"

echo "Started PID ${PID}"
echo "Watch log:"
echo "  tail -f '${LOG_DIR}/train.log'"
echo "Stop run:"
echo "  kill \$(cat '${LOG_DIR}/pid.txt')"
