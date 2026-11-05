#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /home/yukun80/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec python -u -m sar_groundlm.cli "$@"
