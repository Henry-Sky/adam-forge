#!/usr/bin/env bash

# Reconstruction 启动前由脚本 source。默认权重目录由部署环境定义。
# 每个赋值都会保留调用方已设置的值；部署环境可先 export 自定义路径再执行重建命令。
checkpoint_root="/mnt/data/asset-forge/checkpoints"

# Qwen-Image-Edit-2511 本地 CLI 权重目录。安装脚本和 CLI 文档均使用此规范路径；
# 如权重挂载在其他位置，可在 source 本文件前导出 QWEN_IMAGE_EDIT_MODEL_DIR 覆盖。
export QWEN_IMAGE_EDIT_MODEL_DIR="${QWEN_IMAGE_EDIT_MODEL_DIR:-${checkpoint_root}/Qwen-Image-Edit-2511}"

# Wan VACE local inference weights; the tokenizer lives under google/umt5-xxl.
# Install and tests share this path; export WAN_VACE_MODEL_DIR to override it.
export WAN_VACE_MODEL_DIR="${WAN_VACE_MODEL_DIR:-${checkpoint_root}/Wan2.1-VACE-14B}"

# SAM3 checkpoints already mounted under the shared checkpoint root.
export SAM3_CHECKPOINT_DIR="${SAM3_CHECKPOINT_DIR:-${checkpoint_root}/sam3}"
export SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-${SAM3_CHECKPOINT_DIR}/sam3.pt}"
export SAM3_1_CHECKPOINT="${SAM3_1_CHECKPOINT:-${SAM3_CHECKPOINT_DIR}/sam3.1_multiplex.pt}"
