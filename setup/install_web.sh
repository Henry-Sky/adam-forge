#!/usr/bin/env bash
# Web registration service; no GPU/model dependencies.
# Optional skinning jobs use the separate GEM-X environment (see setup/README.md).
# Examples (from repository root, or use this script's absolute path):
#   bash setup/install_web.sh --prefix adam
#   bash setup/install_web.sh --prefix demo --dry-run
#   bash setup/install_web.sh --reuse
# Options: --prefix NAME sets <prefix>-web; --dry-run prints commands only;
#          --reuse explicitly permits installing into an existing environment.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
MODULE=web
PYTHON_VERSION=3.12
parse_args "$@"
prepare_env "$PROJECT_DIR/web_server"
pip_install 'numpy>=1.26,<3' 'scipy>=1.11,<2'
run python "$PROJECT_DIR/tools/install_web_vendor.py"
run env PYTHONPATH="$PROJECT_DIR" python -c 'from web_server.api import create_server; import numpy, scipy; print("Web service imports ready")'
run python "$PROJECT_DIR/web_server/api.py" --help
finish
