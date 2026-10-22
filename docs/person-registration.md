# 人物注册接口

统一入口为 `python interface/register_person.py`，实现 skill 的分阶段流程：3D 框预览 → 确认位置和选择 FBX → 检查整段动作并固定等比拟合 → 模型注册/DWPose → VACE → 回贴和对比。原来 `tests/clip-00610-00/results/rerun_vace_new_reference.py` 的固定实验逻辑已迁入公共模块，任务代码不再放在测试输出目录。

## 环境

从项目根目录安装媒体依赖：

```bash
python -m pip install -r setup/requirements-tools.txt
bash setup/install_environment.sh --help
```

Blender 可在 `--blender` 显式指定，或通过 `BLENDER_BIN`、PATH、本机安装目录发现。需要 VACE 时按 `setup/install_environment.sh` 配置本地 DiffSynth/PyTorch 和已下载权重；`preview`、`prepare`、`compose` 不需要加载 VACE 或访问 GPU。注册配置为 `configs/global.yaml` 与 `configs/render.yaml`，无需相邻项目 checkout。

所有命令支持 `--help`。`--work-dir` 为输出根目录，每次 `preview` 自动创建独立任务子目录，所有任务输出和日志均写入该子目录，数据集保持只读。

人物注册任务参数可在 `configs/register_person.yaml` 中集中设置，分为 `common` 和各阶段配置，阶段值覆盖公共值，显式命令行参数优先。相对路径均相对于运行时工作目录，请从仓库根目录执行：

```bash
python interface/register_person.py preview --config configs/register_person.yaml
# 查看框预览、确认位置后再执行 prepare；配置不能代替显式确认。
python interface/register_person.py prepare --config configs/register_person.yaml --confirm-position
python interface/register_person.py run --config configs/register_person.yaml --dry-run
```

当前配置使用 `common.work_dir: outputs`、`preview.source: data/waymo-source-data` 和 `preview.clip: 610`。clip 按数值匹配，整数 `610` 与字符串 `"00610"` 均选中 metadata 中的 `00610`。

任务目录命名为 `{clip}-{x}_{y}_{z}`，clip 至少补齐五位；FBX 和参考图不参与命名。例如 XYZ 为 `[1, 2, 3]` 时，首次生成 `outputs/00610-1_2_3/`，重名时依次生成 `outputs/00610-1_2_3-1/`、`outputs/00610-1_2_3-2/`。`registration.json`、`inputs/` 和框预览位于任务目录内。坐标使用数值文本命名，完整 XYZ/LWH/heading 仍记录在 manifest 中。

`prepare/run/compose` 默认接续同一根目录中最近一次成功的预览，根目录的 `.person-registration-current.json` 记录该选择。可通过 `--task-dir 00610-1_2_3` 显式选择已有任务；该参数只对后续阶段生效。接口也兼容将已有任务目录直接作为 `--work-dir` 传入。直接复用已有 DWPose 输入的 `run` 仍使用显式指定的工作目录。

先按实际场景修改示例坐标；核对 dry-run 结果后去掉 `--dry-run` 运行推理。各阶段保持相同输出根目录，处理多个任务时使用 `--task-dir` 选定任务。人物资产通过 `data/3d-person-assets` 访问，输入保持只读。

配置解析和原始数据提取测试（不加载模型，临时文件位于系统临时目录）：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_person_*.py'
```

## 1. 画框确认位置

```bash
python interface/register_person.py preview \
  --source data/waymo-source-data --clip 610 \
  --xyz 0.11610073 26.53805042 1.80765938 --lwh 5 5 10 --heading 0 \
  --work-dir outputs
```

`--source` 默认是 `data/waymo-source-data`，可以省略。`--clip` 对应该目录 `metadata.csv` 的 `clip_id`（例如 `00610`），选取唯一的 FRONT 相机记录；编号不存在、重复或源文件缺失时明确报错。可通过 `--frame-start 0 --frame-count 17` 选择原始帧范围。`frame_count` 必须至少为 5 且满足 `4n+1`（5、9、13、17 等），否则报错。省略或设为 `null` 时，将剩余帧数向下裁为满足约束的最大值（例如 200 → 197），三路标定和图像同步裁剪；不足 5 帧则报错。

这组 XYZ/LWH 仅是示例，需按选定的原始 clip 和起始帧重新检查。XYZ 是世界框中心，首帧自车为原点，X 右、Y 前、Z 上，单位米；heading 绕 Z，单位弧度。先查看任务目录中的 `box_preview.mp4` 和 `box_preview.png`。修改框时重新执行预览生成新任务，不能把旧确认套到新参数上。

接口校验 TFRecord CRC，提取原生分辨率 FRONT 图像并按相机内参去畸变，在工作目录 `inputs/` 下生成对齐的 `origin.mp4`、`source-data/images`、3×3 intrinsics、4×4 extrinsics（ego→CV camera）和 ego2global。原始数据及软链接保持只读，无需额外提供 source-data 或原视频。视频 FPS 根据原始时间戳计算并取两位小数，以匹配编码器精度；原始时间戳和帧范围记录在 `inputs/source-data/manifest.json`。预览只画指定框，不附带车辆轨迹框。

## 2. 选择模型并生成注册/姿态输入

确认框后执行：

```bash
python interface/register_person.py prepare \
  --work-dir outputs --confirm-position \
  --fbx data/3d-person-assets/fbx/waving-1.fbx \
  --fit-bounds clip \
  --reference-image data/3d-person-assets/references/traffic-officer-reference.png
```

`--confirm-position` 表示已经看过同一目录的框预览。缺少该参数时，在启动 Blender 前退出。

- 默认 `--fit-bounds clip` 按视频真实 FPS/时长采样整段 FBX，以全动作并集计算一次固定等比缩放与中心，避免抬手、跨步时越框。`fbx_bounds.json` 记录首帧拟合是否超框，以及全片拟合参数。会明确输出首帧尺度下需要的 LWH；复核最终人物尺度后再执行 `run`。
- `--fit-bounds first-frame` 保持原首帧拟合行为，但检测到后续动作超框就停止。可以修订参数后重新 `prepare`。
- `--animation-fps` 覆盖 FBX 帧率，检查与注册同步使用。默认按 FBX 导入 FPS 推进，并在动画短于原视频时循环。
- 带网格模型生成 `registered.mp4` 和 `dwpose_box.mp4`；纯骨骼只生成姿态，明确说明没有实体网格。`--skip-mesh` 可省略可选的实体预览。CPU 为渲染默认设备，可通过 `--render-device` 选择已配置的 GPU 后端。
- DWPose 来源于真实三维关节的相机投影；支持当前 Mixamo 骨架命名。`box_mask.mp4` 与 pose JSON 的投影矩形逐帧阈值校验一致。
- 已符合视频画布大小的参考图保持原构图，RGBA 合成中性灰底。其他图像等比放入首个可见人物框，`--reference-scale` 可调整大小，出画时拒绝裁切。接口不自动抠图或重画人物；应事先提供合格的去背景参考图。
- `--reference-image` 也可以留到 `run` 时提供。启动 VACE 前必须具备参考图。

## 3. VACE 融合

```bash
python interface/register_person.py run --work-dir outputs \
  --model-gpus 0 1 2 3 --steps 50 --seed 1
```

先用 `nvidia-smi` 查看实际空闲显存，选择空闲卡。多卡分片至少四卡，或使用单卡 `--device cuda:0`；显存紧张可选 `--low-vram`，不能与多卡分片同时使用。

模型统一优先加载完整的 `/tmp/vace-model-cache`。缓存缺失或不完整时，从 `setup/checkpoint_environment.sh` 中的 `WAN_VACE_MODEL_DIR` 复制/补齐缓存，再从缓存加载；可用 `--model-dir` 指定补齐来源。缓存位置由 `WAN_VACE_CACHE_DIR` 覆盖。完整缓存包含七个 diffusion shards、UMT5、VAE 和 tokenizer，且不能以指向挂载目录的软链接冒充本地文件。采用进程锁、临时文件和原子替换，已有完整文件不重复复制；来源目录保持只读，不下载模型。

默认提示词内置于 `utils/person_prompts.py`，强调照片级真人和参考图外观，负向包含动画、卡通、动漫和 CGI。可通过 `--prompt`、`--negative-prompt` 或显式指定的 `--prompt-config JSON` 覆盖；实际使用的完整文本保存在 `registration.json`。

`--dry-run` 验证媒体、参考图、帧补齐和本地权重路径并保存计划，不导入 Torch、不加载模型、不复制大权重。缺少缓存时报告来源和待建立的缓存路径，实际 `run` 才执行复制。它会生成所需的预处理媒体与任务记录，适合先检查再执行实际 `run`。

当前接口保持原画布，宽高需为 16 的倍数。帧数自动同步补控制和 mask 至 `4n+1`（至少 5），VACE 原始输出可能带补帧；最终回贴视频裁回原帧数，FPS 不变。需要局部高分辨率推理时使用 skill 中的 crop 工具，当前统一接口没有隐式裁剪或缩放原场景。

同一任务已有推理输出时需要显式 `--overwrite`，或重新执行预览创建新任务。运行参数、源文件哈希、标定哈希、日志和状态均有记录；输入变化会要求重新准备。`--dry-run` 不代表已完成模型推理。

## 直接复用已有 DWPose 输入

这对应原 tests 实验脚本的用途，需同时提供原视频、控制、mask 和 pose JSON，以验证尺寸、时间和编辑区域一致。使用新的工作目录：

```bash
python interface/register_person.py run --work-dir outputs/waving-existing-inputs \
  --origin-video data/waymo-video-data/clip-00610-00/videos/video_origin.mp4 \
  --control-video tests/clip-00610-00/waving_dwpose_box.mp4 \
  --mask-video tests/clip-00610-00/waving_box_mask.mp4 \
  --pose-json tests/clip-00610-00/waving_dwpose_box_pose.json \
  --reference-image person-assets/references/traffic-officer-reference.png \
  --model-gpus 0 1 2 3
```

复用输入不会重新注册 FBX，也不会补做历史动作范围检查；应使用已确认合格的 pose 控制。

## 回贴已有生成结果

准备或导入输入后，可单独执行：

```bash
python interface/register_person.py compose --work-dir outputs \
  --generated-video /path/to/generated.mp4
```

生成视频必须与原视频尺寸/FPS一致，帧数为原帧数或对应 VACE 补帧数。回贴在编码前只改变 pose JSON 矩形内的像素；最终 H.264 编码有损。上下对比上方为控制、下方为回贴结果，按帧号对齐。这里复用了 tests 实验脚本的逐帧拼接逻辑，并增加原视频 mask 回贴。

## 输出

| 路径 | 内容 |
| --- | --- |
| `registration.json` | 状态、输入校验、参数、完整提示词、命令和输出索引 |
| `inputs/` | 从 source/clip 提取的原视频、去畸变图像、逐帧标定和来源记录 |
| `box_preview.mp4/.png/.json` | 待确认的三维框和坐标 |
| `fbx_bounds.json` | 全动作范围、超框诊断与固定拟合 |
| `registered.mp4` | 可选的实体注册预览 |
| `dwpose_box.mp4`、`dwpose_box_pose.json` | 姿态控制与逐帧矩形/关节 |
| `box_mask.mp4`、`reference.png` | VACE 编辑区域和参考图 |
| `inference_control.mp4`、`inference_mask.mp4` | 仅需要补帧时产生 |
| `results/vace_raw.mp4` | VACE 原始生成，可能带补帧 |
| `results/vace_result.mp4` | 按框回贴的最终兼容视频 |
| `results/dwpose_vs_vace_vertical.mp4` | 上下对比 |
| `logs/`、`registration_logs/` | 命令、Blender 和推理日志 |

## 验证输出

通过 `preview` 查看三维框位置，通过 `prepare` 检查全动作范围、实体注册和姿态。运行前可用 `run --dry-run` 核验输入、补帧与模型路径；该命令不会完成模型推理。

完成后核对原视频与最终视频的尺寸、FPS、帧数，检查 mask 与 pose 的逐帧一致性，以及上下对比是否对齐。真实生成质量还需查看人物身份、服装、四肢、脚底及边缘。
