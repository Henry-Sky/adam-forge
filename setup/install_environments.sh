#!/usr/bin/env bash
# 统一安装各模块的 Conda 环境，默认前缀为 adam。
# 命令示例（在仓库根目录执行，也支持用绝对路径从任意目录运行）：
#   bash setup/install_environments.sh qwen
#   bash setup/install_environments.sh harvester sam3
#   bash setup/install_environments.sh all --prefix demo
#   bash setup/install_environments.sh all --dry-run
#   bash setup/install_environments.sh sam3 --reuse
# 参数：qwen / harvester / sam3 / gem_x / soma_x / web / all 选择模块；--prefix 修改环境名前缀；
#       --dry-run 只打印安装命令；--reuse 允许向已有同名环境安装。
# 多模块按选择顺序串行安装；任何模块失败立即退出。
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODULES=()
OPTIONS=()
SELECTED=' '

usage() {
    cat <<'EOF'
Usage: bash setup/install_environments.sh MODULE [MODULE ...] [OPTIONS]

Modules:
  qwen        安装 <prefix>-qwen（默认 adam-qwen）
  harvester   安装 <prefix>-harvester（默认 adam-harvester）
  sam3        安装 <prefix>-sam3（默认 adam-sam3）
  gem_x       安装 <prefix>-gem-x（默认 adam-gem-x）
  soma_x      安装 <prefix>-soma-x（默认 adam-soma-x）
  web         安装 <prefix>-web（默认 adam-web）
  all         依次安装 qwen、harvester、sam3、gem_x、soma_x、web

Options:
  --prefix NAME   环境名前缀，默认 adam
  --dry-run       只打印安装命令，不创建环境、不联网
  --reuse         允许向已有环境安装
  -h, --help      显示帮助
EOF
}

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
add_module() {
    if [[ "$SELECTED" != *" $1 "* ]]; then
        MODULES+=("$1")
        SELECTED+="$1 "
    fi
}

while (( $# )); do
    case "$1" in
        qwen|harvester|sam3|gem_x|soma_x|web) add_module "$1"; shift ;;
        all) add_module qwen; add_module harvester; add_module sam3; add_module gem_x; add_module soma_x; add_module web; shift ;;
        --prefix)
            (( $# >= 2 )) || fail '--prefix requires a value'
            [[ "$2" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || fail 'Invalid prefix; use letters, digits, underscores or hyphens'
            OPTIONS+=(--prefix "$2"); shift 2 ;;
        --dry-run|--reuse) OPTIONS+=("$1"); shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown module or option: $1" ;;
    esac
done

if (( ${#MODULES[@]} == 0 )); then
    usage >&2
    fail 'Select at least one module: qwen, harvester, sam3, gem_x, soma_x, web or all'
fi

# Validate the complete selection before starting any installation.
for module in "${MODULES[@]}"; do
    [[ -f "$PROJECT_DIR/setup/install_$module.sh" ]] || fail "Missing installer: setup/install_$module.sh"
done
for module in "${MODULES[@]}"; do
    printf '\nInstalling module: %s\n' "$module"
    if bash "$PROJECT_DIR/setup/install_$module.sh" "${OPTIONS[@]}"; then
        printf 'Module completed: %s\n' "$module"
    else
        status=$?
        printf 'ERROR: Module %s failed (exit %s); stopping.\n' "$module" "$status" >&2
        exit "$status"
    fi
done
