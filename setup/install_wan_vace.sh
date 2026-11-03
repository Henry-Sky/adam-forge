#!/usr/bin/env bash
# Wan2.1-VACE inference runtime with DiffSynth-Studio.
# Example: bash setup/install_wan_vace.sh --prefix adam
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
MODULE=wan-vace
PYTHON_VERSION=3.10
parse_args "$@"
prepare_env "$PROJECT_DIR/third_party/DiffSynth-Studio"
install_torch
pip_install -e "$PROJECT_DIR/third_party/DiffSynth-Studio" pillow
run python -c 'import torch, torchvision, diffsynth, modelscope; from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig; assert torch.version.cuda == "12.8", torch.version.cuda; print("Wan VACE imports ready:", torch.__version__, diffsynth.__file__)'
finish
