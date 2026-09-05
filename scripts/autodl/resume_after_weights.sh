#!/usr/bin/env bash
set -Eeuo pipefail

# Recovery only: reuse complete caches and accepted CSI smokes, preserving the
# original failed attempt. The launcher rejects any existing formal group.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPS_ROOT="${OPS_ROOT:-/root/autodl-tmp/axhome-multiarc-ops-20260831/retry-$(date +%Y%m%dT%H%M%S)-$$}"
export RESUME_FROM=video_smoke
if [[ "${CONFIRM_AXHOME_MULTIRUN:-}" != RUN_75 ]]; then
  printf 'Explicit CONFIRM_AXHOME_MULTIRUN=RUN_75 is required\n' >&2
  exit 64
fi
[[ ! -e "${OPS_ROOT}" ]] || { printf 'Retry OPS_ROOT must be new\n' >&2; exit 1; }
mkdir -p "${OPS_ROOT}"
exec 9>"${OPS_ROOT}/../pipeline.lock"
flock -n 9 || { printf 'Another pipeline owns the lock\n' >&2; exit 1; }
exec > >(tee -a "${OPS_ROOT}/pipeline.log") 2>&1
trap 'code=$?; printf "%s EXIT recovery code=%s\n" "$(date --iso-8601=seconds)" "${code}"; printf "%s\n" "${code}" > "${OPS_ROOT}/exit_code.txt"' EXIT
printf '%s START recovery=video_smoke\n' "$(date --iso-8601=seconds)"
bash "${SCRIPT_DIR}/run_multiarc_baselines.sh"
