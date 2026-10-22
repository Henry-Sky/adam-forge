# 执行流程

优先使用项目 `interface/register_person.py`，参数与环境见 `docs/person-registration.md`。不要恢复 tests 内固定片段脚本。正式模块：`utils/person_registration.py`（任务编排）、`utils/person_video.py`（参考/mask/补帧/回贴/对比）、`utils/person_box.py`（3D 框预览）、`renders/inspect_fbx.py`（完整动作检查）、`renders/register_fbx_animation.py`（Blender 注册及骨骼投影）。Blender 配置来自本项目 `configs/render.yaml` 与 `configs/global.yaml`。

从项目根目录执行。变量根据用户输入和索引赋值：`source_data`、`origin_video`、`work`、`fbx`、`x y z length width height heading`、`reference`、`gpu_ids`。每个片段/动作/轮次使用独立工作目录。

## 1. 3D 框位置确认

Waymo 先查 metadata.csv 和 source-data/manifest.json，不扫描整个数据集。确认 images、intrinsics、extrinsics、ego2global、原视频逐帧对齐；内参对应视频尺寸，外参为 ego→CV camera。

```bash
python interface/register_person.py preview \
  --source-data "$source_data" --origin-video "$origin_video" \
  --xyz "$x" "$y" "$z" --lwh "$length" "$width" "$height" \
  --heading "$heading" --work-dir "$work"
```

输出 `box_preview.mp4/.png/.json` 和 registration.json。只画指定 3D 框，确认可见、朝向、尺寸、底面位置和全片投影。用户未确认位置时先交付预览再询问；说明这是 SKILL.md 的位置确认步骤。已确认同一参数则直接复用，不重复询问。XYZ/LWH/heading 改变后重新预览。底面 `Z-H/2` 不等于人物脚底。

## 2. 选择 FBX、检查全动作并注册

```bash
python interface/register_person.py prepare --work-dir "$work" --confirm-position \
  --fbx "$fbx" --fit-bounds clip --reference-image "$reference"
```

需要时传 `--blender`、`--animation-fps`、`--reference-scale` 或 `--skip-mesh`。默认 CPU；实际配置了渲染卡才选择 `--render-device`。

接口先生成 fbx_bounds.json：按原视频 FPS/帧数播放真实 FBX 动画，包括 root motion 和循环，记录首帧拟合、所有超框帧、所需 LWH、全动作并集和固定拟合结果。纯骨骼检查真实端点，带网格检查 evaluated mesh bounds。

- `--fit-bounds clip` 用全动作并集计算一次固定等比 scale/center；不逐帧重新居中。检查人物大小和脚底，必要时展示修订预览再进入 VACE。
- `--fit-bounds first-frame` 使用首帧范围；后续动作超框时停止。若用户要求保持既定人体大小，应调整框并保持人体变换；不能用自动缩小掩盖尺度变化。
- 首帧能装下不代表整段能装下；把所有 LWH 等比放大同时会改变归一化，不一定解决超框。纯骨骼范围还需为最终衣服、手脚留白。
- 有 mesh 时可生成 registered.mp4；纯骨骼明确只生成姿态。实体预览需要用户提供兼容蒙皮角色，不能宣称空画面是注册成功。

输出 `dwpose_box.mp4`、`dwpose_box_pose.json`、`box_mask.mp4`。使用实际骨骼的逐帧 3D 相机投影，不从渲染图重新检测姿态。Head 仅代理头部，不编造脸/手指。所有投影点和矩形一致，姿态有动画变化；mask 逐帧阈值 >=128 后严格等于世界 box 的 2D 投影矩形。不能用 mesh silhouette 冒充 box mask。

## 3. 参考图

用户提供外观图优先；需要去背景时使用可用图像编辑工具，检查人物身份、服装、边缘及尺寸。不要把新的生成图误称为像素不变的原人物抠图。保留原图，记录衍生图。

接口对已有推理画布大小的图保留构图，其他图等比放入人物框并补中性灰边；RGBA 先合成灰底再送 VACE。可用 --reference-scale 调整大小，但不得裁掉手脚。检查有效人物像素，T-pose 横向跨度可能导致人物过小。避免竖图直接拉伸成横图。

可在 prepare 时提供参考，也可在 run 时提供。无外观图且有 mesh 时可用 `renders/render_fbx_reference.py` 重新渲染高清首帧；其默认首帧归一化若与注册的 clip bounds 不同，需要同步固定变换。不能放大远处几十像素的人物充当高清参考。

## 4. 本地 VACE 与回贴

```bash
python interface/register_person.py run --work-dir "$work" \
  --model-gpus "${gpu_ids[@]}" --steps 50 --seed 1
```

默认提示词内置于 `utils/person_prompts.py`，中英文含义见 [prompts.md](prompts.md)。按用户风格调整 --prompt/--negative-prompt 或显式 --prompt-config JSON。写实正向避免 animated person，负向排除卡通/动漫/CGI；这些约束不保证身份完全复现。

先检查当前空闲显存。完整 `/tmp/vace-model-cache` 优先，验证七个 diffusion shards、UMT5、VAE 和 tokenizer；显式 --model-dir 优先于默认缓存，配置回退读取 setup/checkpoint_environment.sh。不要每轮复制权重或写入仓库。多卡分片至少四卡，不能把 CUDA_VISIBLE_DEVICES 当作自动聚合；--low-vram 与分片互斥。

接口自动同步补控制和 mask 至至少 5 帧的 `4n+1`，原始输出可能有补帧；回贴裁回原帧数/FPS/尺寸。宽高需为 16 的倍数。--dry-run 验证媒体、准备补帧并保存参数，不导入 Torch/加载权重，不能称为完成推理。

`run` 自动生成原始 VACE、仅 box 内回贴的最终视频，以及上方 DWPose/下方融合的逐帧对比。已有生成可直接回贴：

```bash
python interface/register_person.py compose --work-dir "$work" \
  --generated-video "$generated_video"
```

复用已有 DWPose 时，在新工作目录中通过 run 同时传 --origin-video、--control-video、--mask-video、--pose-json、--reference-image。该模式验证媒体和 mask，不补做历史 FBX 动作范围检查。已有结果必须显式 --overwrite 或使用新目录。

OOM 保留日志，只做一次有依据的资源调整，不能无边界重跑或干预其他任务。后台运行需求要独立启动并核验 PID/日志；会话退出后重跑未完成一组不等于从去噪中间步续跑。

## 5. 可选局部高分辨率输入

人物像素不足时保留 `tools/crop_pose_vace.py` 的固定局部裁剪功能：

```bash
python tools/crop_pose_vace.py prepare --source-data "$source_data" \
  --pose-json "$pose_json" --source-mode images --size 768 --padding 1.5 \
  --output-dir "$crop_work"
```

Waymo 高分辨率 FRONT 可用 --source-mode raw --data-root，先核对 metadata/manifest。统一变换背景、骨架、mask 和参考，保存 crop_config.json。局部 VACE 后逆变换，再只按原始 mask 回贴；裁剪窗口不是扩大编辑区。核查 compositor 的 silhouette/feather 行为符合用户需求。统一接口当前保持原画布，不隐式裁剪。

## 6. 检查与交付

确认编码前 mask 外 decoded origin 像素未变。最终 H.264 High/yuv420p/faststart 有损，不宣称逐像素无损；4:4:4 无损姿态可保留作控制并另给兼容预览。

检查完整解码帧数、FPS、尺寸与帧对齐，首中末帧及连续动作检查人物实体、身份服装、手脚、脚底、边缘和背景漂移。质量不佳如实说明。

交付 results/vace_raw.mp4、results/vace_result.mp4、results/dwpose_vs_vace_vertical.mp4，以及 registration.json、范围报告和日志。记录真实输入、固定变换、动画时序、参考图、完整提示词、模型/GPU和输出。源文件/标定变化要重新准备。更换动作分开命名，仅按用户要求清理旧输出。
