# Conda 环境安装

在仓库根目录执行（脚本也支持从其他工作目录运行）：

```bash
# 统一入口：选择单个、多个或全部模块
bash setup/install_environments.sh qwen
bash setup/install_environments.sh harvester sam3 --prefix adam
bash setup/install_environments.sh all --dry-run

# 也可单独运行各模块安装脚本
bash setup/install_qwen.sh
bash setup/install_harvester.sh
bash setup/install_sam3.sh
```

统一入口支持 `--prefix`、`--dry-run`、`--reuse` 和 `--help`，
按选择顺序串行安装，重复模块只执行一次；任一模块失败立即停止。
未选择模块时显示用法并退出。

| 脚本 | 默认环境名 | Python |
| --- | --- | --- |
| `install_qwen.sh` | `adam-qwen` | 3.12 |
| `install_harvester.sh` | `adam-harvester` | 3.10 |
| `install_sam3.sh` | `adam-sam3` | 3.12 |

三个环境使用 PyTorch 2.10.0、Torchvision 0.25.0、CUDA 12.8 wheels。
需要预先安装 Conda、curl、Git 和支持 CUDA 12.8 的 NVIDIA 驱动。
先运行 `git submodule update --init --recursive` 准备源码。
Harvester 还需要 CUDA 兼容的 GCC/G++（上游测试 GCC 10–13），脚本会在其环境中安装 CUDA Toolkit 12.8 并编译固定版本 gsplat。

## 参数

```bash
# 自定义前缀，创建 demo-harvester
bash setup/install_harvester.sh --prefix demo

# 只打印安装命令，不创建环境、不联网
bash setup/install_sam3.sh --prefix demo --dry-run

# 明确允许向已有环境安装；默认遇到同名环境会退出
bash setup/install_qwen.sh --reuse
```

`--help` 显示参数说明。复用环境时，脚本会校验 Python 主次版本。
安装会执行 `pip check`、Torch CUDA 构建检查和模块导入检查；
Harvester 额外验证 gsplat CUDA 扩展。不会加载或下载模型权重，GPU 推理需另行验证。

## 源与回退

| 类别 | 国内源 1 | 国内源 2 | 官方回退 |
| --- | --- | --- | --- |
| Conda | 清华 conda-forge | 北外 conda-forge | conda.anaconda.org/conda-forge |
| PyPI | 清华 PyPI | 阿里云 PyPI | pypi.org |
| PyTorch cu128 | 南京大学 | 阿里云 | download.pytorch.org |

每次尝试前使用 curl HEAD 检查对应索引的连通性（连接超时 5 秒，总超时 15 秒）。
无法连接或安装失败会输出 `WARNING` 并尝试下一个源；
两个国内源均失败后，输出明确警告，再检查并尝试官方源；最终失败会非零退出。
连通性通过不代表包齐全或版本可用，因此安装失败也会触发回退。
PyTorch 安装同时尝试两个国内 PyPI 源来解析其依赖。

可通过以下环境变量覆盖国内源，不修改全局 `.condarc` 或 pip 配置：

- `ADAM_CONDA_MIRROR`、`ADAM_CONDA_MIRROR_2`：完整 conda-forge channel URL。
- `ADAM_PIP_MIRROR`、`ADAM_PIP_MIRROR_2`：完整 PyPI simple URL，不带末尾 `/`。
- `ADAM_TORCH_MIRROR`、`ADAM_TORCH_MIRROR_2`：完整 PyTorch cu128 索引 URL，不带末尾 `/`。

Diffusers 和固定提交的 gsplat 从官方 GitHub 拉取源码；脚本会明确警告，
其 Python 包依赖仍遵循上述国内源优先策略。未配置未经验证的第三方 GitHub 代理。
本次安装的后续步骤通过环境目录中的 constraints 文件固定 Torch/Torchvision，防止扩展 ABI 被隐式升级破坏。
这些脚本只配置环境，权重继续使用仓库现有 `checkpoint/` 路径。
