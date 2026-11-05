# 模型权重目录

此目录用于通过软链接访问外部模型权重，不要将权重文件提交到 Git。原有模型位于 `/mnt/data/asset-forge/checkpoints`，新增 GEM-X 和 SOMA-X 位于 `/mnt/data/adam-forge/checkpoints`。下表记录仓库内链接路径及目标目录。

已记录的软链接目标：

| 模型 | 仓库内路径 | 外部目标目录 |
| --- | --- | --- |
| Qwen | `checkpoint/Qwen-Image-Edit-2511` | `/mnt/data/asset-forge/checkpoints/Qwen-Image-Edit-2511` |
| Harvester | `checkpoint/asset-harvester` | `/mnt/data/asset-forge/checkpoints/asset-harvester` |
| Wan2.1-VACE | `checkpoint/Wan2.1-VACE-14B` | `/mnt/data/asset-forge/checkpoints/Wan2.1-VACE-14B` |
| SAM3 | `checkpoint/sam3` | `/mnt/data/asset-forge/checkpoints/sam3` |
| GEM-X | `checkpoint/GEM-X` | `/mnt/data/adam-forge/checkpoints/GEM-X` |
| SOMA-X | `checkpoint/SOMA-X` | `/mnt/data/adam-forge/checkpoints/SOMA-X` |

当前工作区已创建以上六个软链接，目标目录均存在。不要把模型文件复制进仓库。

软链接及模型权重由 `.gitignore` 排除，此目录仅跟踪本 README。以上链接依赖本机外部存储；请将目标目录作为只读模型来源使用。

## Wan2.1-VACE-14B 资源

2026-11-05 补齐 `checkpoint/Wan2.1-VACE-14B` 软链接，复用既有 `/mnt/data/asset-forge/checkpoints/Wan2.1-VACE-14B`，没有重新下载或修改权重。目录包含七个 `diffusion_pytorch_model-0000*-of-00007.safetensors`、索引 JSON、`models_t5_umt5-xxl-enc-bf16.pth`、`Wan2.1_VAE.pth` 及 `google/umt5-xxl/` tokenizer。与官方 [Wan-AI/Wan2.1-VACE-14B](https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B) 布局对应，本次未追溯既有缓存的下载 revision 或上游哈希。

本地结构检查核对全部 1,331 个张量的分片索引、shape、dtype、数据区偏移与文件完整长度，不等同于上游 SHA-256 校验。现有索引的 `metadata.total_size` 为全 FP32 等价大小 69,350,371,584 字节；实际前六片为 FP32，末片为 BF16，真实张量合计 63,252,168,960 字节，两者差值来自末片精度；所有索引键均匹配。保留上游文件原样，并在测试报告中记录该差异。

使用本地 `ModelConfig(path=...)` 并禁用自动下载，避免上游默认示例额外拉取通用转换权重。环境、配置和 `tests/` 冒烟推理命令见 [`setup/README.md`](../setup/README.md#wan21-vace-本地配置与测试)。

## GEM-X 与 SOMA-X 资源

GEM-X 人体推理权重来自官方 [nvidia/GEM-X](https://huggingface.co/nvidia/GEM-X)，固定版本 `4fce057963671fd95cd4154297944397e7b5a9a0`：

- `body_s240000.ckpt`、`body_config.json`：人体姿态模型。
- `motion_native77_e774_s292175.ckpt`、`motion_stats_raw.npz`、`motion_native77.yaml`：77 关键点时序动作模型与配套配置。
- `vitpose.pth`、`body_video_native77.yaml`：RGB 视频动作推理配套模型与配置。
- `yolox_humanart.onnx`：人体检测模型，从 [OpenMMLab 官方 ZIP](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip) 提取；原 ZIP 一并保留。
- `soma-v0.3.3/`：GEM-X 要求的 [nvidia/SOMA-X v0.3.3](https://huggingface.co/nvidia/SOMA-X/tree/v0.3.3) 资源，包括人体 rig、形状、程序变换、correctives 和手部/MHR 基础资源；与独立 SOMA-X 当前资源分开保存。

SOMA-X 目录从完整检出且通过 `git lfs fsck` 的 `third_party/SOMA-X/assets/` 复制，源码版本 `56a3d957a549c66425b500fddf7f3183bc6fbdff`（0.3.4），保留上游目录布局，包括 `MHR/`、rig、形状、correctives 和示例资源。SMPL/SMPL-X 需另行授权下载的模型不在此清单中。

Hugging Face 下载优先使用 `https://hf-mirror.com` 并为该命令绕过默认代理；不可用时回退官方源代理。大文件按上游 LFS SHA-256 校验，小文件按 Git blob 校验。两个模型目录的 `download-manifest.json` 记录文件大小与 SHA-256。

可选 EgoHand 手部姿态检查点不属于本次人体推理权重安装；该检查点需要单独的 Hugging Face 访问授权。环境安装命令见 [`setup/README.md`](../setup/README.md)。

从仓库根目录使用这些路径，例如 GEM-X 人体姿态推理：

```bash
conda activate adam-gem-x
gem-infer pose-image --task body --input /path/to/image.jpg \
    --release body --checkpoint checkpoint/GEM-X/body_s240000.ckpt \
    --body-detector checkpoint/GEM-X/yolox_humanart.onnx \
    --soma-assets checkpoint/GEM-X/soma-v0.3.3 \
    --device cuda --output outputs/body.pt
```

独立 SOMA-X 使用 `SOMALayer(data_root="checkpoint/SOMA-X", ...)` 指定资源路径。