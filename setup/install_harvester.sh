#!/usr/bin/env bash
# Asset Harvester runtime. Example: bash setup/install_harvester.sh --prefix adam
# Requires Linux, an NVIDIA CUDA 12.8 compatible driver, and a CUDA-compatible GCC/G++.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
MODULE=harvester
PYTHON_VERSION=3.10
parse_args "$@"
prepare_env "$PROJECT_DIR/third_party/asset-harvester"
install_torch
# conda-forge provides a domestic-mirrored CUDA toolkit, including nvcc.
conda_install install 'cuda-toolkit=12.8' 'gcc_linux-64=12' 'gxx_linux-64=12'
if (( ! DRY_RUN )); then
    export CUDA_HOME="$CONDA_PREFIX"
    command -v nvcc >/dev/null || fail 'CUDA toolkit did not provide nvcc'
    command -v "${CC:-gcc}" >/dev/null || fail 'Install a CUDA-compatible GCC (upstream tests GCC 10–13)'
    command -v "${CXX:-g++}" >/dev/null || fail 'Install a CUDA-compatible G++'
fi
pip_install ninja packaging
warn 'Pinned gsplat source is fetched from official GitHub; PyPI dependencies use domestic-first fallback'
pip_install --no-build-isolation \
    'git+https://github.com/nerfstudio-project/gsplat.git@b60e917c95afc449c5be33a634f1f457e116ff5e'
run python -c 'from gsplat.cuda._backend import _C; assert _C is not None; print("gsplat CUDA extension ready")'
pip_install -e "$PROJECT_DIR/third_party/asset-harvester[ncore-parser,multiview_diffusion,tokengs,camera-estimator]"
fix_decord_metadata
run python -c 'import asset_harvester; from asset_harvester.multiview_diffusion.utils.model_builder import get_models; print("Harvester imports ready")'
finish
