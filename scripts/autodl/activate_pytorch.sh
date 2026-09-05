#!/usr/bin/env bash
# Source this helper so all project Python commands run in the pytorch environment.
MAMBA_EXE="${MAMBA_EXE:-/root/autodl-tmp/axhome-tools/mamba/bin/mamba}"
export MAMBA_EXE
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/root/miniconda3}"
if [[ ! -x "${MAMBA_EXE}" ]]; then
  printf 'Mamba is missing: %s\n' "${MAMBA_EXE}" >&2
  return 1
fi
eval "$("${MAMBA_EXE}" shell hook --shell bash)"
mamba activate pytorch
if [[ "${CONDA_DEFAULT_ENV:-}" != "pytorch" ]]; then
  printf 'Failed to activate pytorch environment\n' >&2
  return 1
fi
export PYTHON_BIN="${CONDA_PREFIX}/bin/python"
