#!/usr/bin/env bash
# GEM-X pose/motion runtime. Example: bash setup/install_gem_x.sh --prefix adam
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
MODULE=gem-x
PYTHON_VERSION=3.12
parse_args "$@"
prepare_env "$PROJECT_DIR/third_party/GEM-X"
install_torch
conda_install install ffmpeg
# GEM pins py-soma-x==0.3.3; keep this environment separate from SOMA-X source.
pip_install -e "$PROJECT_DIR/third_party/GEM-X[video,hand]"
# Exact triangle proximity queries for the optional web Gaussian skinning worker.
pip_install 'trimesh>=4.6,<6' 'rtree>=1.3,<2'
run python -c 'import trimesh, rtree; print("Gaussian binding dependencies ready")'
run python -c 'import gem, soma, onnxruntime; from importlib.metadata import version; assert version("py-soma-x") == "0.3.3"; print("GEM-X imports ready")'
run gem-infer --help
run ffmpeg -version
finish
