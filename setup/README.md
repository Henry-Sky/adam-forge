# Conda 环境安装

`setup/` 统一保留 Bash 安装入口与共享函数；Python 辅助工具位于 `tools/`，由安装脚本使用目标环境的 Python 调用。

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
bash setup/install_wan_vace.sh
bash setup/install_web.sh
# 同时准备两个独立环境
bash setup/install_environments.sh gem_x soma_x
# Wan2.1-VACE 推理环境
bash setup/install_environments.sh wan_vace
```

统一入口支持 `--prefix`、`--dry-run`、`--reuse` 和 `--help`，
按选择顺序串行安装，重复模块只执行一次；任一模块失败立即停止。
未选择模块时显示用法并退出。

### 安装到仓库 `.conda/`

在仓库根目录设置以下变量后运行安装入口，环境将位于 `.conda/adam-<模块>`，包缓存位于 `.conda/pkgs`；该目录已被 Git 忽略。`--prefix` 仅修改环境名称，不是安装目录参数。

```bash
source /root/miniconda3/etc/profile.d/conda.sh  # 按实际 Miniconda 路径调整
export CONDA_ENVS_PATH="$PWD/.conda"
export CONDA_PKGS_DIRS="$PWD/.conda/pkgs"
export ADAM_CONDA_MIRROR=https://mirrors.nju.edu.cn/anaconda/cloud/conda-forge
export ADAM_CONDA_MIRROR_2=https://mirrors.sustech.edu.cn/anaconda/cloud/conda-forge
export ADAM_PIP_MIRROR=https://mirrors.aliyun.com/pypi/simple
export ADAM_PIP_MIRROR_2=https://repo.huaweicloud.com/repository/pypi/simple
export NO_PROXY="${NO_PROXY:-},mirrors.nju.edu.cn,mirrors.sustech.edu.cn,mirrors.aliyun.com,repo.huaweicloud.com"
export no_proxy="$NO_PROXY"
export OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64
export MAX_JOBS=64 CMAKE_BUILD_PARALLEL_LEVEL=64
# 仅准备源码，避免下载 SOMA-X 等子模块的 LFS 资产
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive --jobs 8
bash setup/install_environments.sh all
conda activate "$PWD/.conda/adam-web"
```

以上镜像也是安装脚本的默认国内源，后续安装或修复时可用同名变量覆盖。对可直连的国内镜像，为安装命令设置 `NO_PROXY` / `no_proxy` 绕过代理；不要修改全局代理配置。Harvester 在 A800 上可另外设置 `TORCH_CUDA_ARCH_LIST=8.0`，只编译当前设备架构；更换 GPU 时应按目标设备调整或取消该变量。安装检查不需要多 GPU 推理。

| 脚本 | 默认环境名 | Python |
| --- | --- | --- |
| `install_qwen.sh` | `adam-qwen` | 3.12 |
| `install_harvester.sh` | `adam-harvester` | 3.10 |
| `install_sam3.sh` | `adam-sam3` | 3.12 |
| `install_gem_x.sh` | `adam-gem-x` | 3.12 |
| `install_soma_x.sh` | `adam-soma-x` | 3.12 |
| `install_wan_vace.sh` | `adam-wan-vace` | 3.10 |
| `install_web.sh` | `adam-web` | 3.12 |

模型环境使用 PyTorch 2.10.0、Torchvision 0.25.0、CUDA 12.8 wheels。
模型环境需要预先安装 Conda、curl、Git 和支持 CUDA 12.8 的 NVIDIA 驱动。
先运行 `git submodule update --init --recursive` 准备源码。
Harvester 还需要 CUDA 兼容的 GCC/G++（上游测试 GCC 10–13），脚本会在其环境中安装 GCC/G++ 12、CUDA Toolkit 12.8 并编译固定版本 gsplat。

GEM-X 安装本地 `third_party/GEM-X` 的 `video,hand` 依赖及 FFmpeg，包含人体/手部 ONNX 检测运行库；不包含可选 Open3D 渲染或训练依赖。安装同时准备 `trimesh>=4.6,<6`、`rtree>=1.3,<2`，供网页的 Gaussian 蒙皮绑定后台任务进行最近三角面查询。
GEM-X 固定依赖 `py-soma-x==0.3.3`，独立 SOMA-X 环境安装 `third_party/SOMA-X` 当前源码（0.3.4），请勿合并两个环境。
SOMA-X 脚本安装基础人体与骨架依赖，不安装可选 SMPL/Anny 后端或演示依赖。
Wan2.1-VACE 使用独立 `<prefix>-wan-vace` 环境（Python 3.10），安装 CUDA 12.8 PyTorch、FFmpeg 和本地 `third_party/DiffSynth-Studio` 基础依赖，不安装其量化、训练或日志 extras。固定 `transformers==4.57.6`，限定 `peft>=0.17,<0.19`，显式安装 `sentencepiece>=0.2,<0.3`、`protobuf>=5,<6`，用于本地 UMT5 tokenizer，并显式安装 `PyYAML>=6,<7` 读取带注释的 VACE 配置。源码固定于子模块提交 `98a9e478e259a76b473f5dfdcc8442dca8d5bb41`（DiffSynth 2.2.0）。模型权重由 `checkpoint/Wan2.1-VACE-14B` 提供；本地配置与显式 GPU 测试见下文。安装本身只验证依赖导入，不下载或运行 14B 权重。
两者只执行导入检查，不构造模型、不自动下载模型资源。权重路径和来源见 [`checkpoint/README.md`](../checkpoint/README.md)。

网页服务独立环境为 `<prefix>-web`（默认 `adam-web`），仅安装 Python 3.12、NumPy、SciPy，不需要 Torch、CUDA 或模型权重。
`install_web.sh` 同时调用 `tools/install_web_vendor.py` 将固定版本 Three.js / Spark 安装到忽略的 `web_server/src/static/.vendor/`，优先直连国内 npm 镜像，失败或 SHA-512 不匹配时回退官方源。已准备的前端包会复用，不重复下载。

```bash
bash setup/install_web.sh --prefix adam
bash setup/install_environments.sh web --prefix demo --dry-run
conda activate adam-web
python web_server/api.py --port 8766
```

可选 Gaussian 蒙皮绑定由独立的 `<prefix>-gem-x` 环境计算，`adam-web` 不安装 Torch/SOMA。先安装 `setup/install_gem_x.sh` 并准备匹配的 `checkpoint/GEM-X/soma-v0.3.3`（来源见 checkpoint README）。网页默认寻找同级 `<prefix>-gem-x/bin/python`，可用 `ADAM_BINDING_PYTHON` 指定计算环境的 Python、`ADAM_SOMA_ASSETS` 指定 SOMA v0.3.3 资产目录，再启动网页服务。后台任务使用 64 CPU 线程，每次一个；使用 CPU 完整 SOMA 重建和表面查询，无需占用 GPU，不支持通过复制单人体任务使用多 GPU。

网页鼠标 LBS 测试使用已固定的 Three.js / Spark 2.4.0 与 WebGL2 covariance splat 渲染路径，不增加前端包或 Python 依赖；支持全部已保存影响骨骼数，并遵循左侧显隐、显示模式和点云 opacity 阈值。

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
| Conda | 南京大学 conda-forge | 南科大 conda-forge | conda.anaconda.org/conda-forge |
| PyPI | 阿里云 PyPI | 华为云 PyPI | pypi.org |
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
- `ADAM_PIP_WHEELHOUSE`：可选本地 wheel 缓存目录，作为 `pip --find-links` 的附加来源；依赖和固定版本约束仍照常校验，缺失的包继续从所选镜像获取。

设置 `ADAM_PIP_WHEELHOUSE` 时，Torch/Torchvision 先尝试从该目录离线解析完整依赖，成功则跳过远程 Torch 索引，避免相同版本的远程 wheel 优先于本地文件造成重复下载。缓存不完整时仍按国内源、官方源顺序回退。当前机器可复用 `outputs/setup/wheels/` 中的 Python 3.10 cu128 wheels。

Qwen 使用发布版 `diffusers==0.37.0`（支持 2511 权重的 `zero_cond_t` 配置），并将 Transformers 限制在 4.x，避免依赖 Git 克隆及未经验证的大版本升级。安装检查同时确认该参数存在；仅能导入编辑管线不足以验证 2511 兼容性。
固定提交的 gsplat 从官方 GitHub 拉取源码；脚本会明确警告，
其 Python 包依赖仍遵循上述国内源优先策略。脚本保留调用环境的代理设置，未配置第三方 GitHub 代理。
本次安装的后续步骤通过环境目录中的 constraints 文件固定 Torch/Torchvision，防止扩展 ABI 被隐式升级破坏。
这些脚本只配置环境，权重继续使用仓库现有 `checkpoint/` 路径。

## 安装修复与工具验证

Conda 安装期间可能重新激活环境；共享 helper 在调用 Conda 时暂时关闭 Bash nounset，并保留失败退出码，兼容 CUDA Toolkit 的上游激活脚本。

decord 0.6.0 的 Linux wheel 文件名为 `py3-none-manylinux2010_x86_64`，内部却声明旧的 Python 3.6 标签（[上游问题 #356](https://github.com/dmlc/decord/issues/356)）。SAM3 和 Harvester 安装后运行 `tools/fix_decord_metadata.py`，仅修正这一已知标签及对应 RECORD 校验值，不更改解码库。可用目标环境的 Python 单独运行：

```bash
python tools/fix_decord_metadata.py
```

安装 helper 的回归测试使用 Python 标准库，所有文件夹具写入临时目录：

```bash
python -B -m unittest discover -s tests -p test_setup_helpers.py -v
bash -n setup/common.sh setup/install_qwen.sh setup/install_harvester.sh setup/install_sam3.sh
bash -n setup/install_gem_x.sh setup/install_soma_x.sh setup/install_wan_vace.sh setup/install_web.sh setup/install_environments.sh
bash setup/install_environments.sh all --dry-run
```

人物配对退化脚本 `tools/degrade_person.py` 使用 `adam-gem-x` 环境的 NumPy、SciPy、OpenCV 和 `PyYAML>=6,<7`；GEM-X 安装脚本显式安装并检查 YAML 依赖。该脚本不需要 GPU 或模型权重；已有 Mask 为必需输入。配置见 `configs/person_degradation.yaml`，使用方法见 `tools/README.md`。

## Wan2.1-VACE 本地配置与测试

`configs/wan_vace_smoke.yaml` 使用本地七个 diffusion 分片、UMT5、VAE 和 tokenizer，不需要额外权重。测试通过 `ModelConfig(path=...)` 指定绝对路径，并禁用通用文件重定向与在线下载；直接运行上游示例则可能下载其他仓库的转换权重。Mask 白色表示需要生成的区域，黑色表示上下文。

```bash
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init third_party/DiffSynth-Studio
# 设置前文的 Conda/PyPI 国内镜像和仓库环境目录变量；如有缓存：
export ADAM_PIP_WHEELHOUSE="$PWD/outputs/setup/wheels"
bash setup/install_wan_vace.sh

# 输入预检（不加载模型；依赖导入可能探测 CUDA，建议也在沙箱外执行）。
.conda/adam-wan-vace/bin/python -B tests/test_wan_vace.py \
    --config configs/wan_vace_smoke.yaml --output tests/wan_vace_preflight

# 必须在沙箱外执行。标准单 GPU 管线用一张空闲 A800，CPU 64 线程。
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 \
    .conda/adam-wan-vace/bin/python -B tests/test_wan_vace.py \
    --config configs/wan_vace_smoke.yaml --output tests/wan_vace_smoke --run-inference
```

输出目录必须不存在。按用户要求，此集成测试的实际视频和报告写入忽略的 `tests/wan_vace_*` 目录；临时单元测试仍只用临时目录。输入是 `adam-00000-00057/scene_liner.mp4` 与 `scene_mask.mp4`，原数据只读。默认 384×256、9 帧、4 步，仅验证本地模型加载与 VACE 视频/遮罩条件能完成推理，不作为生成质量评估；正式参数需另行设置。分辨率须为 16 的倍数，帧数须为 4n+1。标准管线不自动跨 GPU 分片；本次短样本不安装 xFuser/USP 或启动重复作业。

`report.json` 记录本地资源清单、依赖版本、GPU、峰值显存、输入/输出参数及异常；推理成功时另有 `result.mp4`、`result_first_frame.png`，输入缩略视频及逐帧无损 Mask PNG 也会保留。

二值 Mask 缩放回归测试只写临时目录：`.conda/adam-wan-vace/bin/python -B -m unittest discover -s tests -p test_wan_vace.py -v`。Mask 先由 YUV 解码为全范围灰度再最近邻缩放，避免先缩放 YUV 再转 RGB 引入接近白色的中间值。

2026-11-05 实际验证完成：`.conda/adam-wan-vace` 的管线导入及 `pip check` 通过；输入预检见 `tests/wan_vace_preflight_verified_20261105/report.json`。GPU 7（A800 80GB）以 BF16 完成默认 384×256、9 帧、4 步 VACE 推理，含加载耗时 57.2 秒，峰值分配显存 44.20 GiB；`tests/wan_vace_smoke_20261105/result.mp4` 已解码验证 H.264 High、yuv420p、10 FPS、9 帧。完整报告及首帧对比见同目录 `report.json`、`comparison.jpg`。安装日志、推理日志及依赖快照分别为 `outputs/setup/wan-vace-install-resume.log`、`wan-vace-smoke.log`、`adam-wan-vace-pip-freeze.txt`。
