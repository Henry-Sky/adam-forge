#!/usr/bin/env bash
# Wan2.1-VACE inference runtime with DiffSynth-Studio.
# Example: bash setup/install_wan_vace.sh --prefix adam
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
MODULE=wan-vace
PYTHON_VERSION=3.10
parse_args "$@"
prepare_env "$PROJECT_DIR/third_party/DiffSynth-Studio"
conda_install install ffmpeg
install_torch
pip_install -e "$PROJECT_DIR/third_party/DiffSynth-Studio" pillow 'transformers==4.57.6' 'peft>=0.17,<0.19' 'sentencepiece>=0.2,<0.3' 'protobuf>=5,<6' 'PyYAML>=6,<7'
run python -c 'import shutil, torch, torchvision, diffsynth, modelscope, sentencepiece, yaml; from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig; assert torch.version.cuda == "12.8", torch.version.cuda; assert shutil.which("ffmpeg") and shutil.which("ffprobe"); print("Wan VACE imports ready:", torch.__version__, diffsynth.__file__)'
finish
