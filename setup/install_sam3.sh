#!/usr/bin/env bash
# SAM3 image/video runtime. Example: bash setup/install_sam3.sh --prefix adam
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
MODULE=sam3
PYTHON_VERSION=3.12
parse_args "$@"
prepare_env "$PROJECT_DIR/third_party/sam3"
install_torch
pip_install -e "$PROJECT_DIR/third_party/sam3" einops scipy pillow \
    'numpy>=1.26,<2' 'opencv-python<4.12' decord pycocotools psutil
run python -c 'from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor; print("SAM3 imports ready")'
finish
