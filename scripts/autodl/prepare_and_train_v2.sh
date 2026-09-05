#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
export DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/AXHome-MM-v1}"
export CSI_CACHE_ROOT="${CSI_CACHE_ROOT:-/root/autodl-tmp/axhome-cache}"
export VIDEO_CACHE_ROOT="${VIDEO_CACHE_ROOT:-/root/autodl-tmp/axhome-video-cache}"
export RESULTS_ROOT="${RESULTS_ROOT:-/root/autodl-tmp/runs/axhome-multiarc-v2}"
export FIRST_ROUND_ROOT="${FIRST_ROUND_ROOT:-/root/autodl-tmp/runs/axhome-multiarc-new}"
export OPS_ROOT="${OPS_ROOT:-/root/autodl-tmp/axhome-multiarc-v2-ops-20260901}"

if [[ "${CONFIRM_AXHOME_MULTIRUN:-}" != RUN_105_V2 ]]; then
  printf 'Explicit CONFIRM_AXHOME_MULTIRUN=RUN_105_V2 is required\n' >&2
  exit 64
fi

mkdir -p "${OPS_ROOT}"
exec 9>"${OPS_ROOT}/pipeline.lock"
flock -n 9 || { printf 'Another v2 pipeline owns the lock\n' >&2; exit 1; }
exec > >(tee -a "${OPS_ROOT}/pipeline.log") 2>&1
PHASE=environment
trap 'code=$?; printf "%s EXIT phase=%s code=%s\n" "$(date --iso-8601=seconds)" "${PHASE}" "${code}"; printf "%s\n" "${code}" > "${OPS_ROOT}/exit_code.txt"' EXIT

source "${SCRIPT_DIR}/activate_pytorch.sh"
printf '%s START environment=%s python=%s git_commit=%s\n' \
  "$(date --iso-8601=seconds)" "${CONDA_DEFAULT_ENV}" "${PYTHON_BIN}" \
  "${AXHOME_GIT_COMMIT:-unknown}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${PYTHON_BIN}" -B - <<'PY'
import sys
import numpy, torch, torchvision, matplotlib, PIL, av

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print("python", sys.version, flush=True)
for module in (numpy, torch, torchvision, matplotlib, PIL, av):
    print(module.__name__, module.__version__, flush=True)
print("cuda_runtime", torch.version.cuda, flush=True)
print("cudnn", torch.backends.cudnn.version(), flush=True)
print("GPU", torch.cuda.get_device_name(0), flush=True)
PY
"${PYTHON_BIN}" -m pip freeze > "${OPS_ROOT}/environment_packages.txt"

PHASE=source_guard
cmp -s -- "${PROJECT_ROOT}/baseline-csi-2dcnn/axhome_csi/model.py" \
  "/root/autodl-tmp/axhome-multiarc-src-20260830/baseline-csi-2dcnn/axhome_csi/model.py"
cmp -s -- "${PROJECT_ROOT}/baseline-video-r3d18/axhome_video/model.py" \
  "/root/autodl-tmp/axhome-multiarc-src-20260830/baseline-video-r3d18/axhome_video/model.py"
sha256sum \
  "${PROJECT_ROOT}/baseline-csi-2dcnn/axhome_csi/model.py" \
  "${PROJECT_ROOT}/baseline-video-r3d18/axhome_video/model.py" \
  > "${OPS_ROOT}/model_definition_sha256.txt"

PHASE=tests
(
  cd -- "${PROJECT_ROOT}/baseline-csi-2dcnn"
  "${PYTHON_BIN}" -B -m unittest discover -s tests -v
) 2>&1 | tee "${OPS_ROOT}/csi_tests.log"
(
  cd -- "${PROJECT_ROOT}/baseline-video-r3d18"
  "${PYTHON_BIN}" -B -m unittest discover -s tests -v
) 2>&1 | tee "${OPS_ROOT}/video_tests.log"

PHASE=cache_acceptance
"${PYTHON_BIN}" -u -B "${SCRIPT_DIR}/verify_multiarc_caches.py" \
  --dataset-root "${DATASET_ROOT}" \
  --csi-cache-root "${CSI_CACHE_ROOT}" \
  --video-cache-root "${VIDEO_CACHE_ROOT}" \
  2>&1 | tee "${OPS_ROOT}/cache_acceptance.log"

PHASE=formal_training
bash "${SCRIPT_DIR}/run_multiarc_baselines_v2.sh"
PHASE=complete
printf '%s COMPLETE runs=105\n' "$(date --iso-8601=seconds)"
