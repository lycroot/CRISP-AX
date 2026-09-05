#!/usr/bin/env bash
set -Eeuo pipefail

# AXHome-MM-v1 multi-architecture baseline launcher.
# This file only orchestrates the existing CLIs. It does not build data caches,
# change splits or tune hyperparameters. The only supported recovery point is
# after all three CSI smokes and before either video smoke has trained.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/pytorch/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/AXHome-MM-v1}"
CSI_CACHE_ROOT="${CSI_CACHE_ROOT:-/root/autodl-tmp/axhome-cache}"
VIDEO_CACHE_ROOT="${VIDEO_CACHE_ROOT:-/root/autodl-tmp/axhome-video-cache}"
REFERENCE_ASSIGNMENT_ROOT="${REFERENCE_ASSIGNMENT_ROOT:-/root/autodl-tmp/reference/csi_in_domain_cnn2d}"
RESULTS_ROOT="${RESULTS_ROOT:-/root/autodl-tmp/runs/axhome-multiarc-new}"
RESUME_FROM="${RESUME_FROM:-}"
ATTEMPT_ID="$(date +%Y%m%dT%H%M%S)-$$"

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

if [[ "${CONFIRM_AXHOME_MULTIRUN:-}" != "RUN_75" ]]; then
  cat >&2 <<'EOF'
Training is locked. After the dataset and both existing caches are complete,
start the 75 formal runs explicitly with:

  CONFIRM_AXHOME_MULTIRUN=RUN_75 bash scripts/autodl/run_multiarc_baselines.sh

The launcher first performs five one-fold/one-epoch smoke runs, then runs the
fixed 10-group matrix from docs/2026-08-30-多架构基线-Codex交接说明.md.
EOF
  exit 64
fi

source "${SCRIPT_DIR}/activate_pytorch.sh"
[[ -x "${PYTHON_BIN}" ]] || die "Python is not executable: ${PYTHON_BIN}"
require_file "${DATASET_ROOT}/manifests/archive_index.csv"
require_dir "${CSI_CACHE_ROOT}"
require_dir "${VIDEO_CACHE_ROOT}"
require_file "${CSI_SOURCE}/run_in_domain.py"
require_file "${CSI_SOURCE}/run_loso.py"
require_file "${CSI_SOURCE}/train_fold.py"
require_file "${VIDEO_SOURCE}/run_in_domain.py"
require_file "${VIDEO_SOURCE}/run_loso.py"
require_file "${VIDEO_SOURCE}/train_fold.py"

for seed in "${SEEDS[@]}"; do
  require_file "${REFERENCE_ASSIGNMENT_ROOT}/seed_${seed}/split_assignments.csv"
done

mkdir -p "$(dirname -- "${RESULTS_ROOT}")"
exec 8>"${RESULTS_ROOT}.lock"
flock -n 8 || die "another launcher owns this result root"
if [[ -z "${RESUME_FROM}" ]]; then
  if [[ -e "${RESULTS_ROOT}" ]] && find "${RESULTS_ROOT}" -mindepth 1 -print -quit | grep -q .; then
    die "RESULTS_ROOT must be absent or empty to prevent mixed runs: ${RESULTS_ROOT}"
  fi
elif [[ "${RESUME_FROM}" == video_smoke ]]; then
  require_dir "${RESULTS_ROOT}"
  [[ ! -L "${RESULTS_ROOT}" ]] || die "recovery result root must not be a symlink"
  for group in in_domain_resnet18 loso_resnet18 in_domain_bilstm loso_bilstm \
    in_domain_transformer loso_transformer video_in_domain_mc3_18 \
    video_loso_mc3_18 video_in_domain_r2plus1d_18 video_loso_r2plus1d_18; do
    [[ ! -e "${RESULTS_ROOT}/${group}" ]] || die "formal group already exists: ${group}"
  done
  for arch in mc3_18 r2plus1d_18; do
    smoke_dir="${RESULTS_ROOT}/_smoke/video_${arch}"
    [[ ! -L "${smoke_dir}" ]] || die "video smoke must not be a symlink"
    if [[ -e "${smoke_dir}" ]]; then
      require_dir "${smoke_dir}"
      while IFS= read -r -d '' artifact; do
        [[ "${artifact}" == "${smoke_dir}/split_assignments.csv" && -f "${artifact}" && ! -L "${artifact}" ]] \
          || die "video smoke already contains training artifacts: ${artifact}"
      done < <(find "${smoke_dir}" -mindepth 1 -print0)
    fi
  done
else
  die "unsupported RESUME_FROM: ${RESUME_FROM}"
fi

# Verify the complete cached files before any training, including cache hits
# for which torch.hub itself does not repeat its download-time hash check.
"${PYTHON_BIN}" -B "${SCRIPT_DIR}/verify_video_weights.py"

"${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; do not start the fixed-budget runs")
print(torch.cuda.get_device_name(0), flush=True)
PY

mkdir -p "${RESULTS_ROOT}/logs" "${RESULTS_ROOT}/_smoke"
STATUS_LOG="${RESULTS_ROOT}/execution_status.log"
CURRENT_GROUP=initialization
trap 'code=$?; if (( code != 0 )); then printf "%s FAILED %s exit=%s\n" "$(date --iso-8601=seconds)" "${CURRENT_GROUP}" "${code}" | tee -a "${STATUS_LOG}"; fi' EXIT

if [[ "${RESUME_FROM}" == video_smoke ]]; then
  CURRENT_GROUP=recovery_acceptance
  recovery_dir="${RESULTS_ROOT}/_recovery/${ATTEMPT_ID}"
  mkdir -p "${recovery_dir}"
  "${PYTHON_BIN}" -B "${SCRIPT_DIR}/verify_multiarc_results.py" \
    --results-root "${RESULTS_ROOT}" --smoke \
    --group _smoke/csi_resnet18 --group _smoke/csi_bilstm --group _smoke/csi_transformer \
    --report "${recovery_dir}/existing_csi_smokes_acceptance.json"
  for arch in mc3_18 r2plus1d_18; do
    smoke_dir="${RESULTS_ROOT}/_smoke/video_${arch}"
    if [[ -d "${smoke_dir}" ]]; then
      cp -a -- "${smoke_dir}" "${recovery_dir}/video_${arch}_before_retry"
    fi
  done
  printf '%s RESUME video_smoke attempt=%s preserved_csi_smokes=3\n' \
    "$(date --iso-8601=seconds)" "${ATTEMPT_ID}" | tee -a "${STATUS_LOG}"
fi

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
  if [[ "${label}" == smoke_* ]]; then
    "${PYTHON_BIN}" -B "${SCRIPT_DIR}/verify_multiarc_results.py" \
      --results-root "${RESULTS_ROOT}" --smoke --group "_smoke/${label#smoke_}"
  else
    "${PYTHON_BIN}" -B "${SCRIPT_DIR}/verify_multiarc_results.py" \
      --results-root "${RESULTS_ROOT}" --group "${label}"
  fi
  printf '%s COMPLETE %s elapsed_seconds=%s\n' "$(date --iso-8601=seconds)" "${label}" "$((SECONDS - started))" | tee -a "${STATUS_LOG}"
}

run_csi_smoke() {
  local arch="$1"
  local learning_rate="$2"
  local dropout="$3"
  run_logged "smoke_csi_${arch}" "${CSI_SOURCE}" \
    "${PYTHON_BIN}" -u -B train_fold.py \
      --dataset-root "${DATASET_ROOT}" \
      --cache-dir "${CSI_CACHE_ROOT}" \
      --output-dir "${RESULTS_ROOT}/_smoke/csi_${arch}" \
      --arch "${arch}" \
      --test-person P01 \
      --val-person P02 \
      --epochs 1 \
      --batch-size 32 \
      --learning-rate "${learning_rate}" \
      --weight-decay 0.0 \
      --dropout "${dropout}" \
      --patience 1 \
      --num-workers 8 \
      --device cuda
}

run_video_smoke() {
  local arch="$1"
  run_logged "smoke_video_${arch}" "${VIDEO_SOURCE}" \
    "${PYTHON_BIN}" -u -B train_fold.py \
      --dataset-root "${DATASET_ROOT}" \
      --cache-dir "${VIDEO_CACHE_ROOT}" \
      --output-dir "${RESULTS_ROOT}/_smoke/video_${arch}" \
      --arch "${arch}" \
      --test-person P01 \
      --val-person P02 \
      --epochs 1 \
      --batch-size 16 \
      --learning-rate 1e-4 \
      --weight-decay 1e-4 \
      --patience 1 \
      --num-workers 8 \
      --device cuda \
      --amp \
      --pretrained
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

  run_logged "video_in_domain_${arch}" "${VIDEO_SOURCE}" \
    "${PYTHON_BIN}" -u -B run_in_domain.py \
      --dataset-root "${DATASET_ROOT}" \
      --cache-dir "${VIDEO_CACHE_ROOT}" \
      --reference-assignment-root "${REFERENCE_ASSIGNMENT_ROOT}" \
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

# Five mandatory one-fold/one-epoch smoke runs.
if [[ -z "${RESUME_FROM}" ]]; then
  run_csi_smoke resnet18 1e-3 0.5
  run_csi_smoke bilstm 1e-3 0.5
  run_csi_smoke transformer 3e-4 0.1
fi
run_video_smoke mc3_18
run_video_smoke r2plus1d_18

# Ten formal groups: 45 CSI runs followed by 30 Video runs.
run_csi_formal resnet18 1e-3 0.5
run_csi_formal bilstm 1e-3 0.5
run_csi_formal transformer 3e-4 0.1
run_video_formal mc3_18
run_video_formal r2plus1d_18

CURRENT_GROUP=final_acceptance
"${PYTHON_BIN}" -B "${SCRIPT_DIR}/verify_multiarc_results.py" \
  --results-root "${RESULTS_ROOT}" --bundle
printf '%s ALL_COMPLETE\n' "$(date --iso-8601=seconds)" | tee -a "${STATUS_LOG}"
