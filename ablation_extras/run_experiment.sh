#!/usr/bin/env bash
# Train one ablation for 50 epochs, evaluate its final-epoch checkpoint on
# the OpenLane validation split and the six scenario subsets, and collect the
# results into ablation_extras/return_back/<exp>_return/.
#
# Usage (from anywhere):
#   bash ablation_extras/run_experiment.sh e1
#   GPUS=2,3 PORT=29622 bash ablation_extras/run_experiment.sh e2a
#   SMOKE=1 bash ablation_extras/run_experiment.sh e1   # ~10 min check:
#       1 epoch on every 400th frame, validation only, separate folders
#   QUARTER=1 bash ablation_extras/run_experiment.sh e1 # reduced protocol:
#       train on every 4th training frame (same 50-epoch schedule, about a
#       quarter of the iterations), evaluate on the full validation split
#
# Experiments: e0 e1 e2a e2b e2c e5 (see ablation_extras/README.md).

set -u
EXP="${1:-}"
case "${EXP}" in
  e0)  CONFIG=e0_full_50e.py ;;
  e1)  CONFIG=e1_straight_templates_50e.py ;;
  e2a) CONFIG=e2a_history_image_only_50e.py ;;
  e2b) CONFIG=e2b_history_lidar_only_50e.py ;;
  e2c) CONFIG=e2c_mean_instead_of_tconv_50e.py ;;
  e5)  CONFIG=e5_three_decoder_layers_50e.py ;;
  *) echo "Usage: bash ablation_extras/run_experiment.sh {e0|e1|e2a|e2b|e2c|e5}"; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}" || exit 2
export PYTHONPATH="$(dirname "${REPO}"):${PYTHONPATH:-}"
GPUS="${GPUS:-0,1}"
NGPU="$(echo "${GPUS}" | awk -F, '{print NF}')"
PORT="${PORT:-$((29500 + RANDOM % 400))}"
CONFIG_PATH="ablation_extras/configs/${CONFIG}"
WORK_ROOT="work_dirs/ablation_${CONFIG%.py}"
RETURN_DIR="ablation_extras/return_back/${EXP}_return"
TEST_CASE=all
if [ "${SMOKE:-0}" = 1 ]; then
  SMOKE_CONFIG="ablation_extras/configs/_smoke_${CONFIG}"
  cat > "${SMOKE_CONFIG}" <<SMOKE_EOF
_base_ = ['./${CONFIG}']
work_dir = './work_dirs/smoke_${EXP}'
train_cfg = dict(max_epochs=1)
subset_override = dict(dataset=dict(dataset_cfg=dict(load_interval=400)))
train_dataloader = subset_override
val_dataloader = subset_override
test_dataloader = subset_override
SMOKE_EOF
  CONFIG_PATH="${SMOKE_CONFIG}"
  WORK_ROOT="work_dirs/smoke_${EXP}"
  RETURN_DIR="ablation_extras/return_back/${EXP}_smoke_return"
  TEST_CASE=validation
elif [ "${QUARTER:-0}" = 1 ]; then
  QUARTER_CONFIG="ablation_extras/configs/_quarter_${CONFIG}"
  cat > "${QUARTER_CONFIG}" <<QUARTER_EOF
_base_ = ['./${CONFIG}']
work_dir = './work_dirs/quarter_${EXP}'
train_cfg = dict(val_interval=25)
train_dataloader = dict(dataset=dict(dataset_cfg=dict(load_interval=4)))
QUARTER_EOF
  CONFIG_PATH="${QUARTER_CONFIG}"
  WORK_ROOT="work_dirs/quarter_${EXP}"
  RETURN_DIR="ablation_extras/return_back/${EXP}_quarter_return"
fi
mkdir -p "${RETURN_DIR}"
STATUS_FILE="${RETURN_DIR}/status.txt"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
note() { echo "[$(ts)] $*" | tee -a "${STATUS_FILE}"; }

finish() {
  local outcome="$1" detail="$2"
  note "${outcome}: ${detail}"
  echo
  echo "================================================================"
  if [ "${outcome}" = "SUCCESS" ]; then
    echo "[$(ts)] ${EXP}: SUCCESS."
  else
    echo "[$(ts)] ${EXP}: FAILED (${detail})."
  fi
  echo "Zip this folder and send it to Claude (LaneLoom conversation):"
  echo "  ${REPO}/${RETURN_DIR}"
  echo "================================================================"
  [ "${outcome}" = "SUCCESS" ] && exit 0 || exit 1
}

note "experiment ${EXP}, config ${CONFIG_PATH}, GPUs ${GPUS}, port ${PORT}"
note "git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
nvidia-smi -L > "${RETURN_DIR}/gpus.txt" 2>&1 || true
cp "ablation_extras/configs/${CONFIG}" "${RETURN_DIR}/"

note "training started (SMOKE=${SMOKE:-0}, QUARTER=${QUARTER:-0}). MMEngine prints an 'eta' field in every log line."
TRAIN_START=$(date +%s)
CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --nproc_per_node="${NGPU}" \
  --master_port="${PORT}" tools/train.py "${CONFIG_PATH}" \
  --work-dir "${WORK_ROOT}" --launcher pytorch
TRAIN_CODE=$?
note "training finished with exit code ${TRAIN_CODE} after $(( ($(date +%s) - TRAIN_START) / 3600 )) h"

EXP_DIR="$(ls -dt "${WORK_ROOT}"/exp_date_* 2>/dev/null | head -1)"
if [ -n "${EXP_DIR}" ]; then
  for log in "${EXP_DIR}"/logs/*.log; do
    [ -f "${log}" ] && cp "${log}" "${RETURN_DIR}/train_$(basename "${log}")"
  done
fi
[ "${TRAIN_CODE}" -eq 0 ] || finish FAILED "training exited with code ${TRAIN_CODE}"
[ -n "${EXP_DIR}" ] || finish FAILED "no experiment directory under ${WORK_ROOT}"

CKPT="$(ls "${EXP_DIR}"/checkpoints/epoch_*.pth 2>/dev/null | sort -V | tail -1)"
[ -n "${CKPT}" ] || CKPT="$(ls -t "${EXP_DIR}"/checkpoints/best_*.pth 2>/dev/null | head -1)"
[ -n "${CKPT}" ] || finish FAILED "no checkpoint found in ${EXP_DIR}/checkpoints"
note "evaluating ${CKPT} (case: ${TEST_CASE})"
echo "${CKPT}" > "${RETURN_DIR}/checkpoint_path.txt"

CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --nproc_per_node="${NGPU}" \
  --master_port="${PORT}" tools/test.py "${CONFIG_PATH}" "${CKPT}" \
  --case "${TEST_CASE}" --launcher pytorch 2>&1 | tee "${RETURN_DIR}/test_output.txt"
TEST_CODE=${PIPESTATUS[0]}
cp "${EXP_DIR}/evaluation/evaluation_results.json" "${RETURN_DIR}/" 2>/dev/null
[ "${TEST_CODE}" -eq 0 ] || finish FAILED "evaluation exited with code ${TEST_CODE}"
[ -f "${RETURN_DIR}/evaluation_results.json" ] || \
  finish FAILED "evaluation_results.json was not written"
finish SUCCESS "results in ${RETURN_DIR}/evaluation_results.json"
