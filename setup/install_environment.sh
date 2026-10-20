#!/usr/bin/env bash
# Install only Wan VACE inference dependencies into an existing Python runtime,
# optionally adding the standalone Blender application. No model weights, SAM3,
# Qwen, Conda bootstrap, Torch replacement, CUDA toolkit or attention extensions.
#
# Parameters (also shown by --help):
#   --stage wan-vace|blender|verify|all  Default: wan-vace; all adds Blender.
#   --python PATH                      Existing interpreter; default: python3.
#   --conda-env NAME                    Use an existing Conda env instead.
#   --with-blender                     Add Blender to the selected stage.
#   --blender-version X.Y.Z             Default: 4.5.14 LTS.
#   --blender-prefix PATH               Default: $HOME/.local/opt.
#   --dry-run                          Print plan; no network/download/install.
# Environment: PYPI_INDEX_URL (preferred domestic index; default Tsinghua),
#   BLENDER_MIRROR_URL (default Aliyun release mirror),
#   INSTALL_LOG_DIR (default repository logs/).
# Domestic sources are checked by HTTP GET with timeouts; official fallbacks
# are used on failure. Blender archives require official SHA-256 verification.
# pip older than 23.1 is upgraded in the selected interpreter before using
# --report. The existing Python interpreter and Torch installation are reused.
# Examples:
#   bash setup/install_environment.sh --dry-run
#   bash setup/install_environment.sh --python /usr/local/bin/python3
#   bash setup/install_environment.sh --conda-env wan-vace --with-blender
#   bash setup/install_environment.sh --stage blender --blender-prefix /opt
#   bash setup/install_environment.sh --stage verify
# Blender here means the optional application install, not unspecified add-ons.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
stage=wan-vace
python_path="${PYTHON:-python3}"
conda_env=""
with_blender=0
blender_version=4.5.14
blender_prefix="${HOME}/.local/opt"
preferred_index="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
blender_mirror="${BLENDER_MIRROR_URL:-https://mirrors.aliyun.com/blender/release}"
log_dir="${INSTALL_LOG_DIR:-${repo_root}/logs}"
dry_run=0
explicit_python=0
work_dir=""

usage() {
    sed -n '2,/^set -Eeuo pipefail/{ /^set -Eeuo pipefail/d; s/^# \{0,1\}//; p; }' "${BASH_SOURCE[0]}"
}
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
while (($#)); do
    case "$1" in
        --stage) stage="${2:?--stage requires a value}"; shift 2 ;;
        --python) python_path="${2:?--python requires a value}"; explicit_python=1; shift 2 ;;
        --conda-env) conda_env="${2:?--conda-env requires a value}"; shift 2 ;;
        --with-blender) with_blender=1; shift ;;
        --blender-version) blender_version="${2:?--blender-version requires a value}"; shift 2 ;;
        --blender-prefix) blender_prefix="${2:?--blender-prefix requires a value}"; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; fail "Unknown argument: $1" ;;
    esac
done
case "$stage" in wan-vace|blender|verify|all) ;; *) fail "Unknown stage: $stage" ;; esac
[[ "$blender_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail 'Invalid Blender version'
if [[ -n "$conda_env" ]] && ((explicit_python)); then fail 'Use either --python or --conda-env'; fi
if [[ -n "$conda_env" ]]; then
    command -v conda >/dev/null || fail 'Conda is not available; use --python instead'
    python_command=(conda run --no-capture-output -n "$conda_env" python)
else
    python_command=("$python_path")
fi
run() {
    printf '+ '; printf '%q ' "$@"; printf '\n'
    if ((!dry_run)); then "$@"; fi
}
cleanup() { if [[ -n "$work_dir" ]]; then rm -rf -- "$work_dir"; fi; }
trap cleanup EXIT
trap 'printf "ERROR: stage=%s line=%s\n" "$stage" "$LINENO" >&2' ERR
if ((!dry_run)); then
    mkdir -p "$log_dir"
    work_dir="$(mktemp -d)"
    log_file="${log_dir}/install-environment-$(date -u +%Y%m%dT%H%M%S)-$$.log"
    exec > >(tee -a "$log_file") 2>&1
    printf 'Installation log: %s\n' "$log_file"
fi

reachable() {
    printf 'Checking connectivity: %s\n' "$1" >&2
    curl --fail --silent --show-error --location --connect-timeout 5 --max-time 15 \
        --output /dev/null "$1"
}
choose_index() {
    local candidate
    for candidate in "${preferred_index%/}" https://pypi.org/simple; do
        if reachable "${candidate}/pip/"; then
            selected_index="$candidate"
            printf 'Using Python package index: %s\n' "$selected_index"
            return
        fi
    done
    fail 'Both preferred and official Python indexes are unreachable'
}
ensure_pip() {
    "${python_command[@]}" -m pip --version
    if ! "${python_command[@]}" -c 'from importlib.metadata import version; import re; import sys; parts = tuple(map(int, re.match(r"(\d+)\.(\d+)", version("pip")).groups())); sys.exit(0 if parts >= (23, 1) else 1)'; then
        printf 'Upgrading pip in the selected interpreter for installation reports and editable builds.\n'
        run "${python_command[@]}" -m pip install --index-url "$selected_index" --upgrade 'pip>=23.1'
    fi
    local install_help
    install_help="$("${python_command[@]}" -m pip help install)"
    [[ "$install_help" == *--report* ]] || fail 'Selected interpreter still has no pip --report support after upgrade'
}
verify_wan() {
    if ((dry_run)); then
        printf '+ Verify editable DiffSynth source and Wan VACE imports; no weights loaded.\n'
        return
    fi
    "${python_command[@]}" -m pip check
    "${python_command[@]}" - "$repo_root" <<'PY'
import sys
from pathlib import Path
import torch
import torchvision
import diffsynth
import imageio_ffmpeg
import sentencepiece
import google.protobuf
from PIL import Image
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import VideoData, save_video
expected = Path(sys.argv[1]) / 'third_party/DiffSynth-Studio/diffsynth/__init__.py'
if Path(diffsynth.__file__).resolve() != expected.resolve():
    raise SystemExit(f'DiffSynth is not imported from this checkout: {diffsynth.__file__}')
print('Wan VACE imports: OK')
print(f'Torch={torch.__version__}; torchvision={torchvision.__version__}; CUDA={torch.cuda.is_available()}')
print('FFmpeg:', imageio_ffmpeg.get_ffmpeg_exe())
if not torch.cuda.is_available():
    print('GPU inference not verified: CUDA is unavailable.')
PY
}
install_wan() {
    local source="${repo_root}/third_party/DiffSynth-Studio"
    [[ -f "${source}/pyproject.toml" ]] || fail 'Initialize the DiffSynth-Studio submodule first'
    if ((dry_run)); then
        printf '+ Check preferred PyPI connectivity: %s/pip/; fall back to pypi.org.\n' "${preferred_index%/}"
        printf '+ Check existing Python >=3.10.1, pip, Torch and torchvision; preserve installed Torch/NumPy versions.\n'
        printf '+ Upgrade pip to >=23.1 in the selected interpreter if needed, then verify --report support.\n'
        run "${python_command[@]}" -m pip install --index-url "$preferred_index" \
            -e "$source" Pillow 'imageio[ffmpeg]' sentencepiece protobuf
        verify_wan
        return
    fi
    command -v curl >/dev/null || fail 'curl is required for mirror connectivity checks'
    # Supplement the selected runtime rather than silently replacing its CUDA stack.
    "${python_command[@]}" - "${work_dir}/constraints.txt" <<'PY'
import sys
from importlib import metadata
from pathlib import Path
if sys.version_info < (3, 10, 1):
    raise SystemExit('Python >=3.10.1 is required')
import pip
import torch
import torchvision
if int(torch.__version__.split('.')[0]) < 2:
    raise SystemExit('Install a compatible PyTorch >=2.0 / torchvision pair first')
constraints = []
for name in ('torch', 'torchvision', 'numpy'):
    try:
        constraints.append(f'{name}=={metadata.version(name)}\n')
    except metadata.PackageNotFoundError:
        pass
Path(sys.argv[1]).write_text(''.join(constraints))
print(f'Using {sys.executable}; existing Torch={torch.__version__}; torchvision={torchvision.__version__}')
PY
    choose_index
    ensure_pip
    run "${python_command[@]}" -m pip install --index-url "$selected_index" \
        --constraint "${work_dir}/constraints.txt" \
        --report "${log_dir}/wan-vace-install-$(date -u +%Y%m%dT%H%M%S)-$$.json" \
        -e "$source" Pillow 'imageio[ffmpeg]' sentencepiece protobuf
    verify_wan
}

install_blender() {
    local series="${blender_version%.*}"
    local name="blender-${blender_version}-linux-x64"
    local target="${blender_prefix}/blender-${blender_version}"
    local official="https://download.blender.org/release/Blender${series}"
    local mirror="${blender_mirror%/}/Blender${series}"
    local base checksum="" actual output
    [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail 'Blender archive installation supports Linux x86_64 only'
    if ((dry_run)); then
        printf '+ Download Blender %s: try %s, then %s (GET with timeouts).\n' "$blender_version" "$mirror" "$official"
        printf '+ Verify archive against official SHA-256; install to %s; run headless smoke test.\n' "$target"
        return
    fi
    for executable in curl tar xz sha256sum; do
        command -v "$executable" >/dev/null || fail "Required command not found: $executable"
    done
    if [[ -e "$target" ]]; then
        [[ -x "${target}/blender" ]] || fail "Unexpected existing path: $target"
    else
        # Small checksum GET is both the connectivity probe and integrity source.
        if ! curl --fail --location --connect-timeout 5 --max-time 30 \
            --output "${work_dir}/blender.sha256" "${official}/blender-${blender_version}.sha256"; then
            fail 'Official Blender checksum source is unreachable; no unverified archive will be installed'
        fi
        checksum="$(awk -v name="${name}.tar.xz" '$2 == name || $2 == "*" name {print $1}' "${work_dir}/blender.sha256")"
        [[ "$checksum" =~ ^[[:xdigit:]]{64}$ ]] || fail 'Requested Blender archive is absent from the official checksum list'
        for base in "$mirror" "$official"; do
            printf 'Checking/downloading Blender source: %s\n' "$base"
            if curl --fail --location --retry 2 --connect-timeout 10 --max-time 1800 \
                --speed-time 30 --speed-limit 1024 --output "${work_dir}/blender.tar.xz" "${base}/${name}.tar.xz"; then
                actual="$(sha256sum "${work_dir}/blender.tar.xz")"
                if [[ "${actual%% *}" == "$checksum" ]]; then break; fi
                printf 'Checksum mismatch; trying next source.\n' >&2
            fi
        done
        [[ "${actual:-}" == "$checksum "* ]] || fail 'All Blender sources failed or produced an invalid checksum'
        run tar -xf "${work_dir}/blender.tar.xz" -C "$work_dir"
        [[ -x "${work_dir}/${name}/blender" ]] || fail 'Archive lacks the expected Blender executable'
        # Verify runtime libraries before moving into the final installation path.
        run "${work_dir}/${name}/blender" --background --factory-startup --python-expr 'import bpy; print("Blender headless:", bpy.app.version_string)'
        run mkdir -p "$blender_prefix"
        run mv "${work_dir}/${name}" "$target"
    fi
    output="$("${target}/blender" --version)"
    [[ "${output%%$'\n'*}" == "Blender ${blender_version}"* ]] || fail 'Installed Blender version mismatch'
    run "${target}/blender" --background --factory-startup --python-expr 'import bpy; print("Blender headless:", bpy.app.version_string)'
    printf 'Blender executable: %s/blender\n' "$target"
}

printf 'Wan VACE setup: stage=%s; dry_run=%s\n' "$stage" "$dry_run"
case "$stage" in
    wan-vace) install_wan ;;
    blender) install_blender ;;
    all) install_wan; install_blender ;;
    verify) verify_wan ;;
esac
if ((with_blender)) && [[ "$stage" != all && "$stage" != blender ]]; then install_blender; fi
printf 'Completed stage: %s\n' "$stage"
