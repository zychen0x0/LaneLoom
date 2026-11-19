#!/usr/bin/env bash
# Run an inference-only analysis on one GPU and report where the result is.
#
#   bash ablation_extras/run_analysis.sh e3 CONFIG CHECKPOINT TAG   # speed
#   bash ablation_extras/run_analysis.sh e4 CONFIG CHECKPOINT TAG   # Q0 coverage
#   bash ablation_extras/run_analysis.sh e6 CONFIG none train       # shape modes (CPU)
#
# GPU selection: GPU=1 bash ablation_extras/run_analysis.sh ...

set -u
KIND="${1:-}"; CONFIG="${2:-}"; CKPT="${3:-}"; TAG="${4:-}"
if [ -z "${TAG}" ] || { [ "${KIND}" != e3 ] && [ "${KIND}" != e4 ] && [ "${KIND}" != e6 ]; }; then
  echo "Usage: bash ablation_extras/run_analysis.sh {e3|e4|e6} CONFIG CHECKPOINT TAG"
  exit 2
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}" || exit 2
export PYTHONPATH="$(dirname "${REPO}"):${PYTHONPATH:-}"
mkdir -p ablation_extras/return_back
LOG="ablation_extras/return_back/${KIND}_${TAG}_console.txt"
if [ "${KIND}" = e3 ]; then
  SCRIPT=ablation_extras/tools/e3_benchmark_fps.py
  OUTPUT="ablation_extras/return_back/e3_fps_${TAG}.json"
elif [ "${KIND}" = e4 ]; then
  SCRIPT=ablation_extras/tools/e4_q0_coverage.py
  OUTPUT="ablation_extras/return_back/e4_q0coverage_${TAG}"
else
  SCRIPT=ablation_extras/tools/e6_shape_modes.py
  OUTPUT="ablation_extras/return_back/e6_shape_modes_${TAG}.json"
fi
if [ "${KIND}" = e6 ]; then
  python "${SCRIPT}" "${CONFIG}" --split "${TAG}" 2>&1 | tee "${LOG}"
else
  CUDA_VISIBLE_DEVICES="${GPU:-0}" python "${SCRIPT}" "${CONFIG}" "${CKPT}" \
    --tag "${TAG}" 2>&1 | tee "${LOG}"
fi
CODE=${PIPESTATUS[0]}
echo
echo "================================================================"
if [ "${CODE}" -eq 0 ] && [ -e "${OUTPUT}" ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${KIND} ${TAG}: SUCCESS."
  echo "Send this file or folder to Claude (LaneLoom conversation):"
  echo "  ${REPO}/${OUTPUT}"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${KIND} ${TAG}: FAILED (exit code ${CODE})."
  echo "Send this log to Claude (LaneLoom conversation):"
  echo "  ${REPO}/${LOG}"
fi
echo "================================================================"
[ "${CODE}" -eq 0 ] && [ -e "${OUTPUT}" ] || exit 1
