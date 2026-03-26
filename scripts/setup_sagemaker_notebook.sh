#!/usr/bin/env bash
# Create/update the lam_sagemaker conda env from conda/lam-sagemaker.yml, install LAM
# in editable mode, and register a Jupyter kernel — for SageMaker *Notebook Instances*
# (classic Jupyter on EC2). After running, restart Jupyter (or the instance) so the
# new kernel appears in the kernel picker.
#
# Usage (from a clone of this repo):
#   bash scripts/setup_sagemaker_notebook.sh
#
# Optional environment variables:
#   LAM_ROOT          — repo root (default: parent of scripts/)
#   LAM_CONDA_ENV     — env name from lam-sagemaker.yml (default: lam_sagemaker)
#   LAM_KERNEL_NAME   — Jupyter kernel id (default: same as LAM_CONDA_ENV)
#   LAM_KERNEL_DISPLAY — shown name in Jupyter (default: LAM (lam_sagemaker))
#   LAM_PYTORCH_CUDA  — Linux only: after conda env, remove conda-forge pytorch/torchvision and
#                       install CUDA wheels from PyTorch’s pip index (e.g. 12.4 → cu124).
#                       Pick 12.x ≤ nvidia-smi "CUDA Version". Ignored on macOS (keeps conda torch).
#   LAM_PYTORCH_WHL   — optional override for the pip index segment (e.g. cu124, cu121). If set,
#                       LAM_PYTORCH_CUDA mapping is skipped; still Linux-only.
#   LAM_SKIP_PIP      — if 1, skip pip install -e (conda + kernel only); GPU pip torch still runs
#                       when LAM_PYTORCH_CUDA or LAM_PYTORCH_WHL is set on Linux.

set -euo pipefail

LAM_ROOT="${LAM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LAM_CONDA_ENV="${LAM_CONDA_ENV:-lam_sagemaker}"
LAM_KERNEL_NAME="${LAM_KERNEL_NAME:-$LAM_CONDA_ENV}"
LAM_KERNEL_DISPLAY="${LAM_KERNEL_DISPLAY:-LAM ($LAM_CONDA_ENV)}"
YAML="${LAM_ROOT}/conda/lam-sagemaker.yml"

if [[ ! -f "$YAML" ]]; then
  echo "error: missing $YAML (set LAM_ROOT to your LAM repo root)" >&2
  exit 1
fi

if [[ ! -f "${LAM_ROOT}/pyproject.toml" ]]; then
  echo "error: ${LAM_ROOT}/pyproject.toml not found" >&2
  exit 1
fi

# shellcheck disable=SC1091
_init_conda() {
  local base
  if [[ -n "${CONDA_EXE:-}" ]]; then
    base="$(dirname "$(dirname "$CONDA_EXE")")"
    if [[ -f "$base/etc/profile.d/conda.sh" ]]; then
      # shellcheck source=/dev/null
      source "$base/etc/profile.d/conda.sh"
      return 0
    fi
  fi
  for base in "${HOME}/anaconda3" "${HOME}/miniconda3" "/home/ec2-user/anaconda3" "/opt/conda"; do
    if [[ -f "$base/etc/profile.d/conda.sh" ]]; then
      # shellcheck source=/dev/null
      source "$base/etc/profile.d/conda.sh"
      return 0
    fi
  done
  if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    return 0
  fi
  echo "error: could not find conda (tried CONDA_EXE, ~/anaconda3, /home/ec2-user/anaconda3, /opt/conda)" >&2
  return 1
}

_init_conda
command -v conda &>/dev/null || {
  echo "error: conda not on PATH after init" >&2
  exit 1
}

_env_exists() {
  local base
  base="$(conda info --base)"
  [[ -d "$base/envs/$LAM_CONDA_ENV" ]]
}

echo "==> Conda env: $LAM_CONDA_ENV (from $YAML)"
if _env_exists; then
  conda env update -f "$YAML" --prune -n "$LAM_CONDA_ENV"
else
  conda env create -f "$YAML"
fi

echo "==> Installing ipykernel (Jupyter kernel support)"
conda install -n "$LAM_CONDA_ENV" -y -c conda-forge ipykernel

# conda-forge pytorch on Linux is often CPU-only; conda "pytorch-cuda" does not reliably swap it.
# Pip CUDA wheels match what works on SageMaker GPU hosts (see PyTorch get-started).
if [[ -n "${LAM_PYTORCH_CUDA:-}" || -n "${LAM_PYTORCH_WHL:-}" ]]; then
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "==> Skipping CUDA pip wheels (LAM_PYTORCH_CUDA / LAM_PYTORCH_WHL are Linux-only)"
  else
    WHL="${LAM_PYTORCH_WHL:-}"
    if [[ -z "$WHL" ]]; then
      case "${LAM_PYTORCH_CUDA:-}" in
        11.8 | 11.8.*) WHL=cu118 ;;
        12.1 | 12.1.*) WHL=cu121 ;;
        12.4 | 12.4.*) WHL=cu124 ;;
        12.6 | 12.6.*) WHL=cu126 ;;
        12.8 | 12.8.*)
          WHL=cu124
          echo "note: LAM_PYTORCH_CUDA=12.8 → using pip index cu124 (wheels run on newer drivers)" >&2
          ;;
        *)
          echo "warning: LAM_PYTORCH_CUDA='${LAM_PYTORCH_CUDA:-}' not mapped; using cu124 (set LAM_PYTORCH_WHL to override)" >&2
          WHL=cu124
          ;;
      esac
    fi
    echo "==> Linux GPU: conda remove pytorch/torchvision, then pip install torch (index ${WHL})"
    conda run -n "$LAM_CONDA_ENV" conda remove -y pytorch torchvision --force || true
    conda run -n "$LAM_CONDA_ENV" pip install -U pip
    conda run -n "$LAM_CONDA_ENV" pip install -U torch torchvision --index-url "https://download.pytorch.org/whl/${WHL}"
  fi
fi

if [[ "${LAM_SKIP_PIP:-0}" != "1" ]]; then
  echo "==> pip install -e ${LAM_ROOT} --no-deps"
  conda run -n "$LAM_CONDA_ENV" pip install -U pip
  conda run -n "$LAM_CONDA_ENV" pip install -e "${LAM_ROOT}" --no-deps
fi

echo "==> Registering Jupyter kernel: $LAM_KERNEL_NAME ($LAM_KERNEL_DISPLAY)"
conda run -n "$LAM_CONDA_ENV" python -m ipykernel install --user \
  --name "$LAM_KERNEL_NAME" \
  --display-name "$LAM_KERNEL_DISPLAY"

echo
echo "Done. Restart Jupyter from the Notebook Instance menu (or reboot) if the kernel does not show up."
echo "Select kernel: ${LAM_KERNEL_DISPLAY}"
