#!/usr/bin/env bash
set -Eeuo pipefail

# Poll the AutoDL host from the local machine and power it off only after the
# second-round matrix has passed its own final acceptance gate.
#
# Run locally (the local machine must remain powered on):
#   CONFIRM_REMOTE_SHUTDOWN=SHUTDOWN_AFTER_105 \
#     bash scripts/autodl/wait_for_v2_and_shutdown.sh

SSH_TARGET="${SSH_TARGET:-autodl}"
RESULTS_ROOT="${RESULTS_ROOT:-/root/autodl-tmp/runs/axhome-multiarc-v2}"
EXPECTED_RUNS="${EXPECTED_RUNS:-105}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-15}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ "${CONFIRM_REMOTE_SHUTDOWN:-}" == "SHUTDOWN_AFTER_105" ]] \
  || die "set CONFIRM_REMOTE_SHUTDOWN=SHUTDOWN_AFTER_105 to enable shutdown"
[[ "${EXPECTED_RUNS}" =~ ^[1-9][0-9]*$ ]] \
  || die "EXPECTED_RUNS must be a positive integer"
[[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
  || die "POLL_SECONDS must be a positive integer"
[[ "${SSH_CONNECT_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] \
  || die "SSH_CONNECT_TIMEOUT must be a positive integer"
[[ "${RESULTS_ROOT}" =~ ^/[A-Za-z0-9._/-]+$ ]] \
  || die "RESULTS_ROOT contains unsupported characters"

STATUS_LOG="${RESULTS_ROOT}/execution_status.log"
SSH_OPTIONS=(
  -c aes128-gcm@openssh.com
  -o "ConnectTimeout=${SSH_CONNECT_TIMEOUT}"
  -o ConnectionAttempts=2
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=2
  -o IPQoS=none
  -o ControlMaster=no
  -o ControlPath=none
)

timestamp() {
  date '+%F %T %Z'
}

read_remote_state() {
  ssh "${SSH_OPTIONS[@]}" "${SSH_TARGET}" \
    "root='${RESULTS_ROOT}'; status='${STATUS_LOG}'; \
     if [ ! -d \"\$root\" ]; then exit 20; fi; \
     count=\$(find \"\$root\" -type f -name metrics.json | wc -l); \
     accepted=0; \
     if [ -f \"\$status\" ] && grep -Fq ' ALL_COMPLETE runs=${EXPECTED_RUNS}' \"\$status\"; then accepted=1; fi; \
     last='NO_STATUS'; \
     if [ -f \"\$status\" ]; then last=\$(tail -n 1 \"\$status\" | tr '\t' ' '); fi; \
     printf '%s\t%s\t%s\n' \"\$count\" \"\$accepted\" \"\$last\""
}

printf '[%s] Monitoring %s:%s for %s accepted runs.\n' \
  "$(timestamp)" "${SSH_TARGET}" "${RESULTS_ROOT}" "${EXPECTED_RUNS}"

while true; do
  if state="$(read_remote_state 2>/dev/null)"; then
    IFS=$'\t' read -r run_count accepted last_status <<<"${state}"
    [[ "${run_count}" =~ ^[0-9]+$ ]] \
      || die "remote metric count is invalid: ${run_count}"

    printf '[%s] runs=%s/%s accepted=%s status=%s\n' \
      "$(timestamp)" "${run_count}" "${EXPECTED_RUNS}" \
      "${accepted}" "${last_status}"

    if (( run_count > EXPECTED_RUNS )); then
      die "found ${run_count} metrics files; refusing automatic shutdown"
    fi

    if (( run_count == EXPECTED_RUNS )) && [[ "${accepted}" == "1" ]]; then
      printf '[%s] Final acceptance confirmed; requesting immediate remote shutdown.\n' \
        "$(timestamp)"
      ssh "${SSH_OPTIONS[@]}" "${SSH_TARGET}" "shutdown -h now"
      printf '[%s] Shutdown command accepted by %s.\n' \
        "$(timestamp)" "${SSH_TARGET}"
      exit 0
    fi
  else
    printf '[%s] Remote host unavailable or result root not ready; retrying.\n' \
      "$(timestamp)" >&2
  fi

  sleep "${POLL_SECONDS}"
done
