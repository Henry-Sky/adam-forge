# Waymo 指定行人姿态补全

`tools/complete_waymo_track_pose.py` 用 metadata 将 clip 映射回原始 TFRecord，并用 clip-local track ID 关联同一行人。DWPose 在每帧的对应人物区域推理 BODY_18；人物区域优先取带 `laser_object_id` 关联的原生相机框，其次取该人的 3D 框投影。框仅用于定位，不绘制到输出视频。

## 环境与权重

本次环境已有 NumPy 2.2.6、Pillow 12.3.0、imageio-ffmpeg 0.6.0。附加依赖隔离安装到 `/tmp`：

```bash
python -m pip install --target /tmp/waymo-pose-deps --no-deps \
  rtmlib==0.0.16 onnxruntime==1.23.2 opencv-python-headless==5.0.0.93 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

若国内镜像不可用，将索引替换为 `https://pypi.org/simple`。独立环境还需安装 `numpy Pillow imageio-ffmpeg` 及 ONNXRuntime 的依赖。

权重使用 [DWPose 官方发布](https://github.com/IDEA-Research/DWPose/tree/onnx) 的 `dw-ll_ucoco_384.onnx`，输入为 384×288。不要写入仓库或数据集目录：

```bash
mkdir -p /tmp/waymo-pose-models
curl -L --fail \
  https://hf-mirror.com/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx \
  -o /tmp/waymo-pose-models/dw-ll_ucoco_384.onnx
```

镜像失败时使用官方发布地址 `https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx`。

## 运行

```bash
PYTHONPATH=/tmp/waymo-pose-deps python tools/complete_waymo_track_pose.py \
  --clip-id clip-00023 --track-id 161 \
  --clips-root /mnt/data/waymo_v1_4_3/waymo_1_4_3_clips \
  --source-root /mnt/data/waymo_v1_4_3/individual_files \
  --model /tmp/waymo-pose-models/dw-ll_ucoco_384.onnx \
  --output-dir tests/waymo-id161-pose-completed \
  --score-threshold 0.35 --max-gap 2 --threads 4
```

所有输入保持只读；模型保存在外部路径，视频、JSON、预览仅写入 `--output-dir`，中间拼图使用 `/tmp` 临时目录。当前执行使用 CPU ONNXRuntime，线程数可调。

## 补全规则与输出

- 原生关键点作为锚点保留。模型对缺失关节补值，并检查躯干、有效关节数和定位框约束。
- 同一相机内仅对两端存在观测、全段有定位框、位移合理的最多 2 帧缺口插值；插值置信度降低。
- 对模型与插值点在人物框相对坐标上做自适应 EMA 平滑，保留人物平移和原标注锚点。
- 补全前增加独立遮挡拦截：用相机深度筛选更近的行人 3D 框，将投影凸包栅格化，估计目标全身与躯干区域被遮住的比例。全身覆盖达到 55% 或躯干达到 65% 时停止自动补全；只保留明确可见的原生标注关节。原生遮挡关节仍保存在原始字段中。
- 遮挡状态下禁止插值通过并重置平滑状态；全身覆盖低于 30%、躯干低于 35% 连续 3 帧后恢复补全。此规则按每个相机、每帧计算，不依赖 clip 编号或固定时间段。框投影覆盖是遮挡近似，不是像素分割。
- 每个关节在 JSON 中标记 `annotation`、`inference`、`interpolated` 或 `missing`。模型分数是模型响应分数，不作为经过校准的正确率。
- `*-completed.mp4` 为补全后的五视图；`*-comparison.mp4` 为上下对比（上：原标注，下：补全）。FRONT 位于上排中间。
- `poses-before-temporal.json` 保存平滑前估计，`poses-completed.json` 保存最终逐帧结果、定位来源、各关节来源、模型 SHA-256；`summary.json` 保存覆盖统计。

更新遮挡策略时，可复用相同 clip、track 的原始推理，避免重复运行模型。下例同时生成“旧补全/新策略”的上下对比：

```bash
PYTHONPATH=/tmp/waymo-pose-deps python tools/complete_waymo_track_pose.py \
  --clip-id clip-00023 --track-id 161 \
  --clips-root /mnt/data/waymo_v1_4_3/waymo_1_4_3_clips \
  --source-root /mnt/data/waymo_v1_4_3/individual_files \
  --model /tmp/waymo-pose-models/dw-ll_ucoco_384.onnx \
  --reuse-inference tests/waymo-id161-pose-completed/poses-before-temporal.json \
  --previous-poses tests/waymo-id161-pose-completed/poses-completed.json \
  --output-dir tests/waymo-id161-pose-occlusion-aware
```

`--max-body-occlusion`、`--max-torso-occlusion` 可调整停止补全阈值。复用输入要求 clip、人物 ID、帧号、帧数匹配；新的 JSON 中记录遮挡估计、阻断原因和每帧是否允许补全。

clip-00023 / ID 161 的遮挡拦截前基线：3D 框 199/199 帧；任一相机有关键点的帧从 79 增至 182；至少 4 条连线的帧从 64 增至 177。这个版本会在前景人物遮住目标时补出过多骨架。

加入遮挡拦截后，FRONT_RIGHT 最后 50 帧均停止自动补全，所有阻断帧都不含推理或插值点。FRONT、FRONT_LEFT、SIDE_LEFT 的最终关键点与上一版逐点相同。新版共 141 帧含关键点、130 帧有至少 4 条连线；覆盖下降来自主动拒绝遮挡位置的补全。原标注中的明确可见点仍可显示，完整原标注保存在 JSON 中。

本次检查了两段编码输出的帧数及持续时间、遮挡状态无自动补点、3 帧清晰后恢复、插值不跨遮挡、未受影响相机逐点一致，并用前后深度与投影覆盖的构造案例核对遮挡估计。查看了跨相机切换和遮挡附近样帧。覆盖统计不等于姿态精度的评测结果。
