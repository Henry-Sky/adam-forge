#!/usr/bin/env bash
# SOMA-X body/skeleton runtime. Example: bash setup/install_soma_x.sh --prefix adam
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
MODULE=soma-x
PYTHON_VERSION=3.12
parse_args "$@"
prepare_env "$PROJECT_DIR/third_party/SOMA-X"
install_torch
pip_install -e "$PROJECT_DIR/third_party/SOMA-X"
# Import only: constructing a layer can automatically download model assets.
run python -c 'from soma import SOMALayer; from importlib.metadata import version; print("SOMA-X imports ready:", version("py-soma-x"))'
finish
