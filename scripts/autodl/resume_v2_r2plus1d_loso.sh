#!/usr/bin/env bash
set -Eeuo pipefail

# Recovery for the interrupted 2026-09-01 v2 run. Execute as SSH commands on
# the ORIGINAL instance, not as a new full-matrix launch. No power action.
[[ "${CONFIRM_AXHOME_V2_RESUME:-}" == "RESUME_8_LOSO" ]] || {
  echo 'Set CONFIRM_AXHOME_V2_RESUME=RESUME_8_LOSO' >&2
  exit 64
}
[[ "$(hostname)" == "autodl-container-b42e48ba7c-4919ed9a" ]] || {
  echo 'Refusing recovery on a different instance' >&2
  exit 1
}
ops=/root/autodl-tmp/axhome-multiarc-v2-resume-20260902
[[ ! -e "$ops" ]] || {
  echo "Recovery directory already exists; inspect it instead of relaunching: $ops" >&2
  exit 1
}
mkdir -- "$ops"

IFS= read -r -d '' job <<'JOB' || true
set -Eeuo pipefail
root=/root/autodl-tmp/runs/axhome-multiarc-v2
first=/root/autodl-tmp/runs/axhome-multiarc-new
project=/root/autodl-tmp/axhome-multiarc-v2-src-20260901
ops=/root/autodl-tmp/axhome-multiarc-v2-resume-20260902
group="$root/video_loso_r2plus1d_18"
status="$root/execution_status.log"
exec >>"$ops/pipeline.log" 2>&1
trap 'code=$?; printf "%s\n" "$code" >"$ops/exit_code.txt"; if (( code != 0 )); then printf "%s FAILED resume_video_loso_r2plus1d_18 exit=%s\n" "$(date --iso-8601=seconds)" "$code" >>"$status"; fi' EXIT
exec 8>"$root.lock"
flock -n 8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 AXHOME_GIT_COMMIT=383cc48
source "$project/scripts/autodl/activate_pytorch.sh"
cd "$project/baseline-video-r3d18"

[[ "$(realpath -- "$root")" == /root/autodl-tmp/runs/axhome-multiarc-v2 ]]
[[ "$(realpath -- "$first")" == /root/autodl-tmp/runs/axhome-multiarc-new ]]
[[ "$(find "$root" -type f -name metrics.json | wc -l)" -eq 97 ]]
[[ -f "$group/test_P01_val_P02/metrics.json" ]]
[[ -f "$group/test_P02_val_P03/metrics.json" ]]
partial="$group/test_P03_val_P04"
[[ -d "$partial" && ! -L "$partial" && ! -e "$partial/metrics.json" ]]
[[ "$(realpath -- "$partial")" == "$group/test_P03_val_P04" ]]
for pair in P04:P05 P05:P06 P06:P07 P07:P08 P08:P09 P09:P10 P10:P01; do
  [[ ! -e "$group/test_${pair%:*}_val_${pair#*:}" ]]
done
cmp -s "$root/first_round_inventory_before.tsv" <(
  cd "$first"
  find . -type f -printf '%P\t%s\t%T@\n' | LC_ALL=C sort
)

# A byte-level manifest protects every artifact in the 97 completed run dirs.
find "$root" -type f -name metrics.json -print0 | sort -z |
  while IFS= read -r -d '' metric; do
    find "${metric%/metrics.json}" -type f -print0
  done | sort -z | xargs -0 sha256sum >"$ops/completed97.sha256"
mkdir "$ops/summary_before_resume"
find "$group" -maxdepth 1 -type f -exec cp -p -- {} "$ops/summary_before_resume/" \;
# Preserve the incomplete fold outside the formal result tree. Nothing deleted.
[[ ! -e "$ops/interrupted_test_P03_val_P04" ]]
mv -T -- "$partial" "$ops/interrupted_test_P03_val_P04"
printf '%s RESUME_START video_loso_r2plus1d_18 remaining=P03-P10 completed_runs=97\n' \
  "$(date --iso-8601=seconds)" | tee -a "$status"
started=$SECONDS

"$PYTHON_BIN" -u -B run_loso.py \
  --dataset-root /root/autodl-tmp/AXHome-MM-v1 \
  --cache-dir /root/autodl-tmp/axhome-video-cache \
  --output-dir "$group" --arch r2plus1d_18 \
  --test-people P03 P04 P05 P06 P07 P08 P09 P10 \
  --epochs 50 --batch-size 16 --learning-rate 1e-4 --weight-decay 1e-4 \
  --patience 10 --num-workers 8 --seed 2026 --device cuda --amp --pretrained \
  2>&1 | tee -a "$root/logs/video_loso_r2plus1d_18.log"

# The subset runner summarizes only P03-P10. Recompute from all TEN folds;
# never combine means or pretend an eight-fold summary is the full protocol.
"$PYTHON_BIN" -B - <<'PY'
import json
from pathlib import Path
from axhome_video.aggregate import write_loso_summary
from axhome_video.splits import HUMAN_ACTIONS

group = Path('/root/autodl-tmp/runs/axhome-multiarc-v2/video_loso_r2plus1d_18')
people = [f'P{i:02d}' for i in range(1, 11)]
runs = []
for i, person in enumerate(people):
    val = people[(i + 1) % 10]
    path = group / f'test_{person}_val_{val}' / 'metrics.json'
    run = json.loads(path.read_text(encoding='utf-8'))
    assert run['test_person'] == person and run['val_person'] == val
    assert run['arch'] == 'r2plus1d_18' and run['seed'] == 2026
    runs.append(run)
summary = write_loso_summary(
    group, runs, requested_people=people,
    failed_folds=[], class_names=HUMAN_ACTIONS,
)
assert summary['complete'] and summary['num_completed'] == 10
print('TEN_FOLD_SUMMARY_REBUILT', flush=True)
PY
"$PYTHON_BIN" -B "$project/scripts/autodl/verify_multiarc_results.py" \
  --matrix v2 --results-root "$root" --group video_loso_r2plus1d_18
printf '%s COMPLETE video_loso_r2plus1d_18 resume_elapsed_seconds=%s\n' \
  "$(date --iso-8601=seconds)" "$((SECONDS - started))" | tee -a "$status"

sha256sum --status -c "$ops/completed97.sha256"
echo COMPLETED_97_ARTIFACTS_UNCHANGED
(
  cd "$first"
  find . -type f -printf '%P\t%s\t%T@\n' | LC_ALL=C sort
) >"$root/first_round_inventory_after.tsv"
cmp -s "$root/first_round_inventory_before.tsv" "$root/first_round_inventory_after.tsv"
"$PYTHON_BIN" -B "$project/scripts/autodl/verify_multiarc_results.py" \
  --matrix v2 --results-root "$root" --bundle
printf '%s FIRST_ROUND_UNCHANGED files=%s\n' \
  "$(date --iso-8601=seconds)" "$(wc -l < "$root/first_round_inventory_after.tsv")" | tee -a "$status"
printf '%s ALL_COMPLETE runs=105\n' "$(date --iso-8601=seconds)" | tee -a "$status"
JOB

screen -dmS axhome-multiarc-v2-resume-20260902 bash -c "$job"
printf 'Recovery dispatched. Log: %s/pipeline.log\n' "$ops"
