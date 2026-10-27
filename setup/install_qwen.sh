#!/usr/bin/env bash
# Qwen Image Edit runtime. Example: bash setup/install_qwen.sh --prefix adam
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
MODULE=qwen
PYTHON_VERSION=3.12
parse_args "$@"
prepare_env "$PROJECT_DIR/third_party/Qwen-Image"
install_torch
# Upstream recommends Diffusers from GitHub. Package dependencies use the mirror;
# GitHub source has no configured trusted domestic mirror, so report direct use.
warn 'Diffusers source is fetched from official GitHub; PyPI dependencies still use domestic-first fallback'
pip_install 'transformers>=4.51.3' accelerate safetensors pillow requests sentencepiece \
    'git+https://github.com/huggingface/diffusers.git'
run python -c 'from diffusers import QwenImageEditPlusPipeline; import transformers; print("Qwen imports ready")'
finish
