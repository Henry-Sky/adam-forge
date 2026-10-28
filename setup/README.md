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
bash setup/install_gem_x.sh
bash setup/install_soma_x.sh
bash setup/install_web.sh
# 同时准备两个独立环境
bash setup/install_environments.sh gem_x soma_x
```

统一入口支持 `--prefix`、`--dry-run`、`--reuse` 和 `--help`，
按选择顺序串行安装，重复模块只执行一次；任一模块失败立即停止。
未选择模块时显示用法并退出。

| 脚本 | 默认环境名 | Python |
| --- | --- | --- |
| `install_qwen.sh` | `adam-qwen` | 3.12 |
| `install_harvester.sh` | `adam-harvester` | 3.10 |
| `install_sam3.sh` | `adam-sam3` | 3.12 |
| `install_gem_x.sh` | `adam-gem-x` | 3.12 |
| `install_soma_x.sh` | `adam-soma-x` | 3.12 |
| `install_web.sh` | `adam-web` | 3.12 |

模型环境使用 PyTorch 2.10.0、Torchvision 0.25.0、CUDA 12.8 wheels。
模型环境需要预先安装 Conda、curl、Git 和支持 CUDA 12.8 的 NVIDIA 驱动。
先运行 `git submodule update --init --recursive` 准备源码。
Harvester 还需要 CUDA 兼容的 GCC/G++（上游测试 GCC 10–13），脚本会在其环境中安装 GCC/G++ 12、CUDA Toolkit 12.8 并编译固定版本 gsplat。

GEM-X 安装本地 `third_party/GEM-X` 的 `video,hand` 依赖及 FFmpeg，包含人体/手部 ONNX 检测运行库；不包含可选 Open3D 渲染或训练依赖。
GEM-X 固定依赖 `py-soma-x==0.3.3`，独立 SOMA-X 环境安装 `third_party/SOMA-X` 当前源码（0.3.4），请勿合并两个环境。
SOMA-X 脚本安装基础人体与骨架依赖，不安装可选 SMPL/Anny 后端或演示依赖。
两者只执行导入检查，不构造模型、不自动下载模型资源。权重路径和来源见 [`checkpoint/README.md`](../checkpoint/README.md)。

网页服务独立环境为 `<prefix>-web`（默认 `adam-web`），仅安装 Python 3.12、NumPy、SciPy，不需要 Torch、CUDA 或模型权重。
`install_web.sh` 同时调用 `install_web_vendor.py` 将固定版本 Three.js / Spark 安装到忽略的 `web_server/src/static/.vendor/`，优先直连国内 npm 镜像，失败或 SHA-512 不匹配时回退官方源。已准备的前端包会复用，不重复下载。

```bash
bash setup/install_web.sh --prefix adam
bash setup/install_environments.sh web --prefix demo --dry-run
conda activate adam-web
python web_server/api.py --port 8766
```

网页参数和 Python 接口见 [`web_server/README.md`](../web_server/README.md)。

GPU 诊断、CUDA 扩展编译与验证须在 Codex 沙箱外运行，遵循根目录 `AGENTS.md`；沙箱内 GPU 不可见不能作为驱动异常的依据。

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

Qwen 使用发布版 `diffusers==0.37.0`（支持 2511 权重的 `zero_cond_t` 配置），并将 Transformers 限制在 4.x，避免依赖 Git 克隆及未经验证的大版本升级。安装检查同时确认该参数存在；仅能导入编辑管线不足以验证 2511 兼容性。
固定提交的 gsplat 从官方 GitHub 拉取源码；脚本会明确警告，
其 Python 包依赖仍遵循上述国内源优先策略。脚本保留调用环境的代理设置，未配置第三方 GitHub 代理。
本次安装的后续步骤通过环境目录中的 constraints 文件固定 Torch/Torchvision，防止扩展 ABI 被隐式升级破坏。
这些脚本只配置环境，权重继续使用仓库现有 `checkpoint/` 路径。

## 安装修复与工具验证

Conda 安装期间可能重新激活环境；共享 helper 在调用 Conda 时暂时关闭 Bash nounset，并保留失败退出码，兼容 CUDA Toolkit 的上游激活脚本。

decord 0.6.0 的 Linux wheel 文件名为 `py3-none-manylinux2010_x86_64`，内部却声明旧的 Python 3.6 标签（[上游问题 #356](https://github.com/dmlc/decord/issues/356)）。SAM3 和 Harvester 安装后运行 `fix_decord_metadata.py`，仅修正这一已知标签及对应 RECORD 校验值，不更改解码库。可用目标环境的 Python 单独运行：

```bash
python setup/fix_decord_metadata.py
```

安装 helper 的回归测试使用 Python 标准库，所有文件夹具写入临时目录：

```bash
python -B -m unittest discover -s tests -p test_setup_helpers.py -v
bash -n setup/common.sh setup/install_qwen.sh setup/install_harvester.sh setup/install_sam3.sh
bash -n setup/install_gem_x.sh setup/install_soma_x.sh setup/install_web.sh setup/install_environments.sh
bash setup/install_environments.sh all --dry-run
```
