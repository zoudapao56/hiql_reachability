#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# HIQL / Reachability background launcher
# ============================================================
# Usage:
#   1. Edit the configuration block below.
#   2. Run:
#        bash run.sh
#
# Every launch starts one background process and writes:
#   background_runs/<run_name>/train.log
#   background_runs/<run_name>/pid.txt
#   background_runs/<run_name>/command.txt
#   background_runs/<run_name>/config.txt
#
# To run multiple experiments, edit SEED/CUDA_DEVICES/ALGO_TAG and run this
# script multiple times.

# ============================================================
# 1. Experiment identity
# ============================================================
RUN_GROUP="EXP"
SEED=0
ALGO_TAG="HIQL_R_decay_seed0"

# GPU id to use. Examples:
#   CUDA_DEVICES=""     # use default visible GPUs
#   CUDA_DEVICES="0"    # use only GPU 0
#   CUDA_DEVICES="1"    # use only GPU 1
CUDA_DEVICES="0"

# ============================================================
# 2. Environment selection
# ============================================================
# Common choices:
#   antmaze-ultra-diverse-v0
#   antmaze-large-diverse-v2
#   antmaze-large-diverse-v0  # needs antmaze_aux/antmaze-large-diverse-v0-aux.npz
ENV_NAME="antmaze-ultra-diverse-v0"

# Recommended settings:
#   ultra: PRETRAIN_STEPS=1500002, WAY_STEPS=50
#   large: PRETRAIN_STEPS=1000002, WAY_STEPS=25 or 50
PRETRAIN_STEPS=1500002
WAY_STEPS=50

# ============================================================
# 3. Basic training settings
# ============================================================
EVAL_INTERVAL=100000
SAVE_INTERVAL=250000
BATCH_SIZE=1024

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
USE_WAYPOINTS=1

# ============================================================
# 4. Reachability mode
# ============================================================
# Original HIQL:
#   USE_REACHABILITY=0
#   USE_REACHABILITY_MOD_ADV=0
#   REACHABILITY_FILTER_POLICY=0
#
# R scoring through high loss only:
#   USE_REACHABILITY=1
#   USE_REACHABILITY_MOD_ADV=1
#   REACHABILITY_FILTER_POLICY=0
#
# R hard filtering during evaluation/execution:
#   USE_REACHABILITY=1
#   REACHABILITY_FILTER_POLICY=1
USE_REACHABILITY=1
USE_REACHABILITY_MOD_ADV=1
REACHABILITY_FILTER_POLICY=0

# If >0, train only R for these steps before HIQL.
# Current preferred setup keeps this at 0.
REACHABILITY_PRETRAIN_STEPS=0
FREEZE_REACHABILITY_AFTER_PRETRAIN=0

# ============================================================
# 5. When to enable R
# ============================================================
# REACHABILITY_START_STEP has priority if >= 0.
# Otherwise start step = PRETRAIN_STEPS * REACHABILITY_START_FRAC.
#
# Examples:
#   start from beginning: REACHABILITY_START_FRAC=0
#   start halfway:        REACHABILITY_START_FRAC=0.5
REACHABILITY_START_FRAC=0
REACHABILITY_START_STEP=-1

# ============================================================
# 6. Alpha schedule
# ============================================================
# The high actor uses:
#   mod_adv = adv + alpha_effective * R * (adv > 0)
#
# Mode A: constant alpha
#   REACHABILITY_ALPHA=0.5
#   REACHABILITY_ALPHA_WARMUP_STEPS=0
#   REACHABILITY_ALPHA_DECAY_STEPS=0
#
# Mode B: alpha warmup from 0 to REACHABILITY_ALPHA
#   REACHABILITY_ALPHA=0.5
#   REACHABILITY_ALPHA_WARMUP_STEPS=200000
#   REACHABILITY_ALPHA_DECAY_STEPS=0
#
# Mode C: alpha decay from max to min
#   REACHABILITY_ALPHA_DECAY_STEPS=500000
#   REACHABILITY_ALPHA_MAX=1.0
#   REACHABILITY_ALPHA_MIN=0.1
#
# Current preferred setup: Mode C.
REACHABILITY_ALPHA=0.5
REACHABILITY_ALPHA_WARMUP_STEPS=0
REACHABILITY_ALPHA_MAX=1.0
REACHABILITY_ALPHA_MIN=0.1
REACHABILITY_ALPHA_DECAY_STEPS=500000

# ============================================================
# 7. Reachability sampling
# ============================================================
# Positive:
#   nearby anchor state -> future 1..REACHABILITY_HORIZON steps
#
# Hard negative:
#   nearby anchor state -> future K+1..REACHABILITY_NEGATIVE_HORIZON steps
#
# Random negative:
#   state farther than REACHABILITY_FAR_EPS in xy space
REACHABILITY_HORIZON=25
REACHABILITY_ANCHOR_EPS=0.75
REACHABILITY_FAR_EPS=5.0
REACHABILITY_NEGATIVE_HORIZON=200
REACHABILITY_HARD_NEGATIVE_PROB=0.75
REACHABILITY_ANCHOR_SAMPLE_ATTEMPTS=16

# ============================================================
# 8. Launch
# ============================================================
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
  --use_waypoints "${USE_WAYPOINTS}"
  --way_steps "${WAY_STEPS}"
  --high_p_randomgoal "${HIGH_P_RANDOMGOAL}"
  --use_reachability "${USE_REACHABILITY}"
  --reachability_filter_policy "${REACHABILITY_FILTER_POLICY}"
  --reachability_pretrain_steps "${REACHABILITY_PRETRAIN_STEPS}"
  --freeze_reachability_after_pretrain "${FREEZE_REACHABILITY_AFTER_PRETRAIN}"
  --reachability_start_frac "${REACHABILITY_START_FRAC}"
  --reachability_start_step "${REACHABILITY_START_STEP}"
  --use_reachability_mod_adv "${USE_REACHABILITY_MOD_ADV}"
  --reachability_alpha "${REACHABILITY_ALPHA}"
  --reachability_alpha_warmup_steps "${REACHABILITY_ALPHA_WARMUP_STEPS}"
  --reachability_alpha_max "${REACHABILITY_ALPHA_MAX}"
  --reachability_alpha_min "${REACHABILITY_ALPHA_MIN}"
  --reachability_alpha_decay_steps "${REACHABILITY_ALPHA_DECAY_STEPS}"
  --reachability_horizon "${REACHABILITY_HORIZON}"
  --reachability_anchor_eps "${REACHABILITY_ANCHOR_EPS}"
  --reachability_far_eps "${REACHABILITY_FAR_EPS}"
  --reachability_negative_horizon "${REACHABILITY_NEGATIVE_HORIZON}"
  --reachability_hard_negative_prob "${REACHABILITY_HARD_NEGATIVE_PROB}"
  --reachability_anchor_sample_attempts "${REACHABILITY_ANCHOR_SAMPLE_ATTEMPTS}"
)

printf "%q " "${CMD[@]}" > "${LOG_DIR}/command.txt"
printf "\n" >> "${LOG_DIR}/command.txt"

cat > "${LOG_DIR}/config.txt" <<EOF
RUN_NAME=${RUN_NAME}
RUN_GROUP=${RUN_GROUP}
SEED=${SEED}
ENV_NAME=${ENV_NAME}
ALGO_TAG=${ALGO_TAG}
CUDA_DEVICES=${CUDA_DEVICES}

PRETRAIN_STEPS=${PRETRAIN_STEPS}
EVAL_INTERVAL=${EVAL_INTERVAL}
SAVE_INTERVAL=${SAVE_INTERVAL}
BATCH_SIZE=${BATCH_SIZE}
WAY_STEPS=${WAY_STEPS}

USE_REACHABILITY=${USE_REACHABILITY}
USE_REACHABILITY_MOD_ADV=${USE_REACHABILITY_MOD_ADV}
REACHABILITY_FILTER_POLICY=${REACHABILITY_FILTER_POLICY}
REACHABILITY_START_FRAC=${REACHABILITY_START_FRAC}
REACHABILITY_START_STEP=${REACHABILITY_START_STEP}

REACHABILITY_ALPHA=${REACHABILITY_ALPHA}
REACHABILITY_ALPHA_WARMUP_STEPS=${REACHABILITY_ALPHA_WARMUP_STEPS}
REACHABILITY_ALPHA_MAX=${REACHABILITY_ALPHA_MAX}
REACHABILITY_ALPHA_MIN=${REACHABILITY_ALPHA_MIN}
REACHABILITY_ALPHA_DECAY_STEPS=${REACHABILITY_ALPHA_DECAY_STEPS}

REACHABILITY_HORIZON=${REACHABILITY_HORIZON}
REACHABILITY_ANCHOR_EPS=${REACHABILITY_ANCHOR_EPS}
REACHABILITY_FAR_EPS=${REACHABILITY_FAR_EPS}
REACHABILITY_NEGATIVE_HORIZON=${REACHABILITY_NEGATIVE_HORIZON}
REACHABILITY_HARD_NEGATIVE_PROB=${REACHABILITY_HARD_NEGATIVE_PROB}
REACHABILITY_ANCHOR_SAMPLE_ATTEMPTS=${REACHABILITY_ANCHOR_SAMPLE_ATTEMPTS}
EOF

echo "Launching background run:"
echo "  name: ${RUN_NAME}"
echo "  gpu : ${CUDA_DEVICES:-default visible GPUs}"
echo "  log : ${LOG_DIR}/train.log"
echo "  cmd : ${LOG_DIR}/command.txt"
echo "  cfg : ${LOG_DIR}/config.txt"

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
echo "Check process:"
echo "  ps -p ${PID} -o pid,etime,%cpu,%mem,cmd"
echo "Stop run:"
echo "  kill \$(cat '${LOG_DIR}/pid.txt')"
