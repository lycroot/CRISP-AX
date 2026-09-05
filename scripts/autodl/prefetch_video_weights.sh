#!/usr/bin/env bash
set -Eeuo pipefail

# Download binary artifacts only; no Python or model execution. Each range has
# bounded retries, and only the fully verified file receives the final name.
STAGING_ROOT="${STAGING_ROOT:-/root/autodl-tmp/axhome-weight-repair-20260831}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/.cache/torch/hub/checkpoints}"
mkdir -p "${STAGING_ROOT}" "${CHECKPOINT_ROOT}"
exec 7>"${STAGING_ROOT}/download.lock"
flock -n 7 || { printf 'Another checkpoint download is active\n' >&2; exit 1; }

fetch_weight() {
  local filename="$1" size="$2" expected_hash="$3"
  local url="https://download.pytorch.org/models/${filename}"
  local destination="${CHECKPOINT_ROOT}/${filename}"
  if [[ -f "${destination}" ]]; then
    [[ "$(stat -c %s "${destination}")" == "${size}" ]]
    printf '%s  %s\n' "${expected_hash}" "${destination}" | sha256sum --check -
    return
  fi
  local parts_dir="${STAGING_ROOT}/${filename}.parts"
  mkdir -p "${parts_dir}"
  local chunk_size=4194304 start=0 end index=0 part pid failed=0
  local -a parts=() pids=()
  while (( start < size )); do
    end=$((start + chunk_size - 1))
    (( end < size )) || end=$((size - 1))
    printf -v part '%s/%04d.part' "${parts_dir}" "${index}"
    parts+=("${part}")
    (
      # Reuse complete ranges after an interrupted transfer; the assembled
      # checkpoint must still pass its full SHA-256 below.
      if [[ ! -f "${part}" ]] || [[ "$(stat -c %s "${part}")" != "$((end - start + 1))" ]]; then
        curl --fail --location --silent --show-error \
          --connect-timeout 15 --max-time 120 --speed-limit 65536 --speed-time 20 \
          --retry 4 --retry-delay 2 --retry-all-errors \
          --range "${start}-${end}" --output "${part}" "${url}"
      fi
      [[ "$(stat -c %s "${part}")" == "$((end - start + 1))" ]]
      printf '%s RANGE_COMPLETE %s index=%s bytes=%s\n' \
        "$(date --iso-8601=seconds)" "${filename}" "${index}" "$((end - start + 1))"
    ) &
    pids+=("$!")
    if (( ${#pids[@]} == 8 )); then
      for pid in "${pids[@]}"; do
        wait "${pid}" || failed=1
      done
      (( failed == 0 )) || return 1
      pids=()
    fi
    start=$((end + 1))
    index=$((index + 1))
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  (( failed == 0 )) || return 1
  local assembled="${CHECKPOINT_ROOT}/${filename}.verified-download"
  [[ ! -e "${assembled}" ]] || { printf 'Assembly target already exists\n' >&2; return 1; }
  cat -- "${parts[@]}" > "${assembled}"
  [[ "$(stat -c %s "${assembled}")" == "${size}" ]]
  printf '%s  %s\n' "${expected_hash}" "${assembled}" | sha256sum --check -
  [[ ! -e "${destination}" ]] || return 1
  mv -- "${assembled}" "${destination}"
  printf '%s WEIGHT_READY %s sha256=%s bytes=%s\n' \
    "$(date --iso-8601=seconds)" "${filename}" "${expected_hash}" "${size}"
}

fetch_weight mc3_18-a90a0ba3.pth 46841888 \
  a90a0ba35ca1242d15b77511ff28bfb29cc596988b5ea36081042f8e2f54212b
fetch_weight r2plus1d_18-91a641e6.pth 126162996 \
  91a641e6c2ab531d1aca5f4321b4d802ec5c3babc15df855cdb6e39c6a1107c8
