#!/usr/bin/env bash
# Shared installer helpers; source this file from install_*.sh.
set -euo pipefail

SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SETUP_DIR/.." && pwd)"
PREFIX=adam
DRY_RUN=0
REUSE=0
PIP_MIRROR="${ADAM_PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_MIRRORS=("$PIP_MIRROR" "${ADAM_PIP_MIRROR_2:-https://mirrors.aliyun.com/pypi/simple}")
TORCH_MIRRORS=("${ADAM_TORCH_MIRROR:-https://mirrors.nju.edu.cn/pytorch/whl/cu128}" "${ADAM_TORCH_MIRROR_2:-https://mirrors.aliyun.com/pytorch-wheels/cu128}")
CONDA_MIRRORS=("${ADAM_CONDA_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge}" "${ADAM_CONDA_MIRROR_2:-https://mirrors.bfsu.edu.cn/anaconda/cloud/conda-forge}")

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
run() {
    printf '+'; printf ' %q' "$@"; printf '\n'
    if (( DRY_RUN )); then return 0; fi
    "$@"
}

parse_args() {
    while (( $# )); do
        case "$1" in
            --prefix)
                (( $# >= 2 )) || fail '--prefix requires a value'
                PREFIX="$2"; shift 2 ;;
            --dry-run) DRY_RUN=1; shift ;;
            --reuse) REUSE=1; shift ;;
            -h|--help)
                printf 'Usage: bash %s [--prefix NAME] [--dry-run] [--reuse]\n' "${0##*/}"
                printf 'Environment: <prefix>-%s (default: adam-%s); Python %s\n' "$MODULE" "$MODULE" "$PYTHON_VERSION"
                printf 'Domestic mirrors first; official fallback emits WARNING.\n'
                printf '%s\n' '--reuse explicitly allows modifying an existing environment.'
                exit 0 ;;
            *) fail "Unknown argument: $1" ;;
        esac
    done
    [[ "$PREFIX" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || fail 'Invalid prefix; use letters, digits, underscores or hyphens'
    ENV_NAME="$PREFIX-$MODULE"
}

# HEAD checks avoid downloading package indexes; installation still tests usability.
reachable() {
    local url="$1"
    if run curl --head --location --fail --silent --show-error \
        --connect-timeout 5 --max-time 15 "$url"; then return 0; fi
    warn "Source connectivity check failed: $url"
    return 1
}

conda_install() {
    local action="$1"; shift
    local mirror
    for mirror in "${CONDA_MIRRORS[@]}"; do
        if ! reachable "$mirror/noarch/repodata.json"; then continue; fi
        if run conda "$action" -y -n "$ENV_NAME" --override-channels -c "$mirror" "$@"; then return 0; fi
        warn "Conda install failed via $mirror; trying next source"
    done
    warn "Both domestic Conda sources failed; falling back to official conda-forge: $*"
    reachable https://conda.anaconda.org/conda-forge/noarch/repodata.json || fail 'Official Conda source is unreachable'
    run conda "$action" -y -n "$ENV_NAME" --override-channels -c https://conda.anaconda.org/conda-forge "$@" || fail 'Both Conda sources failed'
}

# Disable inherited pip indexes/configuration so each attempt uses its declared sources.
pip_at() {
    local index="$1"; shift
    run env -u PIP_EXTRA_INDEX_URL -u PIP_NO_INDEX -u PIP_FIND_LINKS \
        PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL="$index" \
        python -m pip --disable-pip-version-check --timeout 30 --retries 1 install "$@"
}
pip_install() {
    local mirror
    for mirror in "${PIP_MIRRORS[@]}"; do
        if ! reachable "$mirror/"; then continue; fi
        if pip_at "$mirror" "$@"; then return 0; fi
        warn "pip install failed via $mirror; trying next source"
    done
    warn "Both domestic PyPI sources failed; falling back to official PyPI: $*"
    reachable https://pypi.org/simple/ || fail 'Official PyPI source is unreachable'
    pip_at https://pypi.org/simple "$@" || fail 'Both PyPI installation attempts failed'
}

prepare_env() {
    local source_dir="$1"
    [[ -f "$source_dir/README.md" ]] || fail "Missing submodule: $source_dir; run git submodule update --init --recursive"
    if (( ! DRY_RUN )); then
        command -v curl >/dev/null || fail 'Install curl for source connectivity checks'
        command -v conda >/dev/null || fail 'Install Miniconda or Miniforge first'
        local conda_base
        conda_base="$(conda info --base)"
        set +u
        source "$conda_base/etc/profile.d/conda.sh"
        set -u
        local envs
        envs="$(conda env list --json)"
        if printf '%s' "$envs" | "$conda_base/bin/python" -c \
            'import json,sys; from pathlib import Path; sys.exit(not any(Path(p).name == sys.argv[1] for p in json.load(sys.stdin)["envs"]))' "$ENV_NAME"; then
            (( REUSE )) || fail "Environment $ENV_NAME exists; use --reuse to modify it"
            printf 'Reusing environment: %s\n' "$ENV_NAME"
        else
            conda_install create "python=$PYTHON_VERSION" pip
        fi
        set +u
        conda activate "$ENV_NAME"
        set -u
        python -c 'import sys; expected=tuple(map(int,sys.argv[1].split("."))); assert sys.version_info[:2] == expected, f"Expected Python {expected}, got {sys.version}"' "$PYTHON_VERSION"
    else
        conda_install create "python=$PYTHON_VERSION" pip
        run conda activate "$ENV_NAME"
    fi
    # SAM3 still imports pkg_resources; retain a setuptools version providing it.
    pip_install --upgrade pip 'setuptools<81' wheel
}

install_torch() {
    local mirror pip_mirror installed=0
    # Try both ordinary PyPI mirrors too: CUDA wheels have dependencies hosted there.
    for mirror in "${TORCH_MIRRORS[@]}"; do
        if ! reachable "$mirror/torch/"; then continue; fi
        for pip_mirror in "${PIP_MIRRORS[@]}"; do
            if ! reachable "$pip_mirror/"; then continue; fi
            if pip_at "$mirror" 'torch==2.10.0+cu128' 'torchvision==0.25.0+cu128' --extra-index-url "$pip_mirror"; then
                installed=1; break
            fi
            warn "PyTorch install failed via $mirror with $pip_mirror; trying next source"
        done
        if (( installed )); then break; fi
    done
    if (( ! installed )); then
        warn 'Both domestic PyTorch sources failed; falling back to official PyTorch cu128 and PyPI'
        reachable https://download.pytorch.org/whl/cu128/torch/ || fail 'Official PyTorch source is unreachable'
        reachable https://pypi.org/simple/ || fail 'Official PyPI source is unreachable'
        pip_at https://download.pytorch.org/whl/cu128 'torch==2.10.0+cu128' 'torchvision==0.25.0+cu128' \
            --extra-index-url https://pypi.org/simple || fail 'Both PyTorch installation attempts failed'
    fi
    # Keep later dependency resolution from upgrading Torch and breaking CUDA extensions.
    if (( ! DRY_RUN )); then
        printf 'torch==2.10.0+cu128\ntorchvision==0.25.0+cu128\n' > "$CONDA_PREFIX/adam-torch-constraints.txt"
        export PIP_CONSTRAINT="$CONDA_PREFIX/adam-torch-constraints.txt"
    fi
    run python -c 'import torch, torchvision; assert torch.version.cuda == "12.8", torch.version.cuda; print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)'
}

finish() {
    run python -m pip check
    printf 'Installation complete: %s\nActivate: conda activate %s\n' "$ENV_NAME" "$ENV_NAME"
}
