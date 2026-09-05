#!/usr/bin/env bash
set -Eeuo pipefail

# AXHome-MM-v1 seven-model, 105-run launcher. This is deliberately separate
# from the historical 75-run launcher so the first-round matrix cannot be
# selected accidentally.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/pytorch/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/AXHome-MM-v1}"
CSI_CACHE_ROOT="${CSI_CACHE_ROOT:-/root/autodl-tmp/axhome-cache}"
VIDEO_CACHE_ROOT="${VIDEO_CACHE_ROOT:-/root/autodl-tmp/axhome-video-cache}"
RESULTS_ROOT="${RESULTS_ROOT:-/root/autodl-tmp/runs/axhome-multiarc-v2}"
FIRST_ROUND_ROOT="${FIRST_ROUND_ROOT:-/root/autodl-tmp/runs/axhome-multiarc-new}"

CSI_SOURCE="${PROJECT_ROOT}/baseline-csi-2dcnn"
VIDEO_SOURCE="${PROJECT_ROOT}/baseline-video-r3d18"
SEEDS=(2026 2027 2028 2029 2030)
PEOPLE=(P01 P02 P03 P04 P05 P06 P07 P08 P09 P10)

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "required file is missing: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "required directory is missing: $1"
}

if [[ "${CONFIRM_AXHOME_MULTIRUN:-}" != "RUN_105_V2" ]]; then
  cat >&2 <<'EOF'
Training is locked. Start only the second-round 105-run matrix with:

  CONFIRM_AXHOME_MULTIRUN=RUN_105_V2 bash scripts/autodl/run_multiarc_baselines_v2.sh

The historical 75-run result root is read-only and is never used as output.
EOF
  exit 64
fi

[[ "${RESULTS_ROOT}" != "${FIRST_ROUND_ROOT}" ]] || die "v2 and first-round roots must differ"
source "${SCRIPT_DIR}/activate_pytorch.sh"
[[ -x "${PYTHON_BIN}" ]] || die "Python is not executable: ${PYTHON_BIN}"
require_file "${DATASET_ROOT}/manifests/archive_index.csv"
require_dir "${CSI_CACHE_ROOT}"
require_dir "${VIDEO_CACHE_ROOT}"
require_dir "${FIRST_ROUND_ROOT}"
for source in \
  "${CSI_SOURCE}/run_in_domain.py" "${CSI_SOURCE}/run_loso.py" \
  "${VIDEO_SOURCE}/run_in_domain.py" "${VIDEO_SOURCE}/run_loso.py"; do
  require_file "${source}"
done

mkdir -p "$(dirname -- "${RESULTS_ROOT}")"
exec 8>"${RESULTS_ROOT}.lock"
flock -n 8 || die "another launcher owns the v2 result root"
if [[ -e "${RESULTS_ROOT}" ]] && find "${RESULTS_ROOT}" -mindepth 1 -print -quit | grep -q .; then
  die "v2 result root must be absent or empty: ${RESULTS_ROOT}"
fi

mkdir -p "${RESULTS_ROOT}/logs"
STATUS_LOG="${RESULTS_ROOT}/execution_status.log"
FIRST_ROUND_BEFORE="${RESULTS_ROOT}/first_round_inventory_before.tsv"
FIRST_ROUND_AFTER="${RESULTS_ROOT}/first_round_inventory_after.tsv"

inventory_first_round() {
  (
    cd -- "${FIRST_ROUND_ROOT}"
    find . -type f -printf '%P\t%s\t%T@\n' | LC_ALL=C sort
  )
}

inventory_first_round > "${FIRST_ROUND_BEFORE}"
CURRENT_GROUP=initialization
trap 'code=$?; if (( code != 0 )); then printf "%s FAILED %s exit=%s\n" "$(date --iso-8601=seconds)" "${CURRENT_GROUP}" "${code}" | tee -a "${STATUS_LOG}"; fi' EXIT

"${PYTHON_BIN}" -B "${SCRIPT_DIR}/verify_video_weights.py"
"${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; do not start the fixed-budget runs")
print(torch.cuda.get_device_name(0), flush=True)
PY

run_logged() {
  local label="$1"
  local workdir="$2"
  shift 2
  CURRENT_GROUP="${label}"
  local started=${SECONDS}
  printf '%s START %s\n' "$(date --iso-8601=seconds)" "${label}" | tee -a "${STATUS_LOG}"
  (
    cd -- "${workdir}"
    "$@"
  ) 2>&1 | tee -a "${RESULTS_ROOT}/logs/${label}.log"
  "${PYTHON_BIN}" -B "${SCRIPT_DIR}/verify_multiarc_results.py" \
    --matrix v2 --results-root "${RESULTS_ROOT}" --group "${label}"
  printf '%s COMPLETE %s elapsed_seconds=%s\n' \
    "$(date --iso-8601=seconds)" "${label}" "$((SECONDS - started))" | tee -a "${STATUS_LOG}"
}

run_csi_formal() {
  local arch="$1"
  local learning_rate="$2"
  local dropout="$3"

  run_logged "in_domain_${arch}" "${CSI_SOURCE}" \
    "${PYTHON_BIN}" -u -B run_in_domain.py \
      --dataset-root "${DATASET_ROOT}" \
      --cache-dir "${CSI_CACHE_ROOT}" \
      --output-dir "${RESULTS_ROOT}/in_domain_${arch}" \
      --arch "${arch}" \
      --seeds "${SEEDS[@]}" \
      --epochs 50 \
      --batch-size 32 \
      --learning-rate "${learning_rate}" \
      --weight-decay 0.0 \
      --dropout "${dropout}" \
      --patience 10 \
      --num-workers 8 \
      --device cuda

  run_logged "loso_${arch}" "${CSI_SOURCE}" \
    "${PYTHON_BIN}" -u -B run_loso.py \
      --dataset-root "${DATASET_ROOT}" \
      --cache-dir "${CSI_CACHE_ROOT}" \
      --output-dir "${RESULTS_ROOT}/loso_${arch}" \
      --arch "${arch}" \
      --test-people "${PEOPLE[@]}" \
      --epochs 50 \
      --batch-size 32 \
      --learning-rate "${learning_rate}" \
      --weight-decay 0.0 \
      --dropout "${dropout}" \
      --patience 10 \
      --num-workers 8 \
      --device cuda
}

run_video_formal() {
  local arch="$1"
  local reference_root="${RESULTS_ROOT}/in_domain_cnn2d"

  for seed in "${SEEDS[@]}"; do
    require_file "${reference_root}/seed_${seed}/split_assignments.csv"
  done

  run_logged "video_in_domain_${arch}" "${VIDEO_SOURCE}" \
    "${PYTHON_BIN}" -u -B run_in_domain.py \
      --dataset-root "${DATASET_ROOT}" \
      --cache-dir "${VIDEO_CACHE_ROOT}" \
      --reference-assignment-root "${reference_root}" \
      --output-dir "${RESULTS_ROOT}/video_in_domain_${arch}" \
      --arch "${arch}" \
      --seeds "${SEEDS[@]}" \
      --epochs 50 \
      --batch-size 16 \
      --learning-rate 1e-4 \
      --weight-decay 1e-4 \
      --patience 10 \
      --num-workers 8 \
      --device cuda \
      --amp \
      --pretrained

  run_logged "video_loso_${arch}" "${VIDEO_SOURCE}" \
    "${PYTHON_BIN}" -u -B run_loso.py \
      --dataset-root "${DATASET_ROOT}" \
      --cache-dir "${VIDEO_CACHE_ROOT}" \
      --output-dir "${RESULTS_ROOT}/video_loso_${arch}" \
      --arch "${arch}" \
      --test-people "${PEOPLE[@]}" \
      --epochs 50 \
      --batch-size 16 \
      --learning-rate 1e-4 \
      --weight-decay 1e-4 \
      --patience 10 \
      --num-workers 8 \
      --device cuda \
      --amp \
      --pretrained
}

# Four CSI architectures (60 runs), then three Video architectures (45 runs).
# cnn2d in-domain is deliberately first because all Video in-domain groups
# must use this round's assignments as their reference.
run_csi_formal cnn2d 1e-3 0.5
run_csi_formal resnet18 1e-3 0.5
run_csi_formal bilstm 1e-3 0.5
run_csi_formal transformer 3e-4 0.1
run_video_formal r3d_18
run_video_formal mc3_18
run_video_formal r2plus1d_18

CURRENT_GROUP=final_acceptance
"${PYTHON_BIN}" -B "${SCRIPT_DIR}/verify_multiarc_results.py" \
  --matrix v2 --results-root "${RESULTS_ROOT}" --bundle
inventory_first_round > "${FIRST_ROUND_AFTER}"
cmp -s -- "${FIRST_ROUND_BEFORE}" "${FIRST_ROUND_AFTER}" \
  || die "first-round result inventory changed during v2 execution"
printf '%s FIRST_ROUND_UNCHANGED files=%s\n' \
  "$(date --iso-8601=seconds)" "$(wc -l < "${FIRST_ROUND_AFTER}")" | tee -a "${STATUS_LOG}"
printf '%s ALL_COMPLETE runs=105\n' "$(date --iso-8601=seconds)" | tee -a "${STATUS_LOG}"
