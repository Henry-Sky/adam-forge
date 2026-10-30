#!/usr/bin/env bash
# Qwen Image Edit runtime. Example: bash setup/install_qwen.sh --prefix adam
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
MODULE=qwen
PYTHON_VERSION=3.12
parse_args "$@"
prepare_env "$PROJECT_DIR/third_party/Qwen-Image"
install_torch
# This release includes QwenImageEditPlusPipeline without a Git source checkout.
pip_install 'transformers>=4.51.3,<5' accelerate safetensors pillow requests sentencepiece \
    'diffusers==0.37.0'
run python -c 'import inspect; from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel; import transformers; assert "zero_cond_t" in inspect.signature(QwenImageTransformer2DModel.__init__).parameters; print("Qwen 2511 imports ready")'
finish
