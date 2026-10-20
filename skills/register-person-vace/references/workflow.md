# 执行流程与命令

以下命令从 apollo-forge 仓库根目录执行。`${...}` 是本次任务参数，先赋值；不要硬编码上次任务的 clip、坐标或输出路径。原始视频若未采用项目目录结构，可在输出目录建立临时 `asset/clip-NNNNN-NN/source-data` 和 `video/clip-NNNNN-NN/videos/video_origin.mp4` 链接，指向用户的 source-data 和视频；不要复制改写数据集。

## 预检

- Waymo 首先查看 `data/waymo-source-data/metadata.csv` 列名，按 clip ID 定位 TFRecord；不要遍历整个数据集。part 帧范围读取 source-data/manifest.json。
- 图片用 `source_image_frame_id` 排序；支持 `000000.jpg` 与 `location_phx-000000-1508010034528.jpg`。
- 确认 intrinsics 是视频尺寸下的 CV K，extrinsics 是 ego→CV camera；ego2global 是逐帧 ego→存储世界的刚体变换。参考世界→相机为 `extrinsic @ inv(pose_t) @ pose_0`。
- 依赖：NumPy、OpenCV、PyYAML、imageio-ffmpeg、google-crc32c、本地 DiffSynth/PyTorch 与 Blender。使用可用 Python 环境，不在 skill 中绑定临时 `/tmp` 依赖目录。
- 检查 CUDA 空闲显存；若 GPU 在沙箱内不可见，按运行环境的权限机制执行 GPU 命令。

## 注册和姿态

```bash
source setup/checkpoint_environment.sh
python tools/register_fbx_animation.py \
  --asset-root "$asset_root" --video-root "$video_root" --clip "$clip" --part "$part" \
  --fbx "$fbx" --xyz "$x" "$y" "$z" --lwh "$length" "$width" "$height" \
  --heading "$heading" --blender "$blender" --reference-root "$reference_root" \
  --output "$work/registered.mp4"

python tools/register_fbx_animation.py \
  --asset-root "$asset_root" --video-root "$video_root" --clip "$clip" --part "$part" \
  --fbx "$fbx" --xyz "$x" "$y" "$z" --lwh "$length" "$width" "$height" \
  --heading "$heading" --blender "$blender" --reference-root "$reference_root" \
  --mode dwpose-box --output "$work/pose.mp4"
```

`pose_pose.json` 保存每帧 `rectangle_xyxy` 和 `body_keypoints_xy`。矩形来自固定 3D box 的八角点/近裁面交点投影，不是逐帧 mesh AABB；已有 mesh mask 不能直接冒充用户指定的 box 编辑区。姿态视频自身可能用 4:4:4 无损作为模型输入，这与最终兼容交付格式是两回事。

## 高清参考与局部输入

```bash
python tools/render_fbx_reference.py \
  --source-data "$source_data" --fbx "$fbx" \
  --xyz "$x" "$y" "$z" --lwh "$length" "$width" "$height" --heading "$heading" \
  --reference-root "$reference_root" --size 768 --samples 32 --output "$work/crop/reference.png"

python tools/crop_pose_vace.py prepare \
  --source-data "$source_data" --pose-json "$work/pose_pose.json" \
  --data-root "$waymo_root" --source-mode raw --size 768 --padding 1.5 \
  --output-dir "$work/crop"
```

Waymo `raw` 模式通过 metadata 索引和 manifest 帧范围直接解码原始 FRONT 图像。普通已标定片段改用 `--source-mode images`，从 source-data/images 读取；此时不需要 Waymo 数据根目录/manifest。图片尺寸须与标定对应，不可只依据宽高比猜测额外裁剪。

`crop_config.json` 记录原视频大小、固定窗口 XYXY、推理大小、帧映射和原始/裁剪矩形。人物在控制图中宜有至少约 150 像素高度，兼顾上下文、手臂活动范围与显存；不能假定统一尺寸总是足够。查看 `control_preview.png`、`source_crop_preview.png` 和 `reference.png` 确认输入。

## 泛化提示词

按用户意图调整，避免把某次 FBX 的制服、身份或动作写死：

> Render a realistic animated person in the editable region. Preserve the character appearance, clothing and proportions from the reference image. Follow the body pose control in every frame, matching joint positions, gestures, timing and location. Render a solid, fully clothed human body with natural anatomy, coherent surface detail and smooth temporally consistent motion. Match scene scale, perspective, lighting and camera movement. Keep the person grounded. Replace the black rectangle and colored skeleton with the rendered character while preserving the unmasked environment.

负面提示词可用：`colored skeleton, pose lines, black rectangle, stick figure, wireframe, duplicate person, extra limbs, malformed anatomy, floating body, transparent body, flickering, changing identity, changing clothing, blurry character, background changes`。

## 推理与回贴

```bash
source setup/checkpoint_environment.sh
python wan_vace_inference.py \
  --control-video "$work/crop/control.mp4" --mask-video "$work/crop/mask.mp4" \
  --reference-image "$work/crop/reference.png" \
  --prompt "$prompt" --negative-prompt "$negative_prompt" \
  --width 768 --height 768 --num-frames "$num_frames" --fps "$fps" \
  --steps 50 --cfg-scale 5 --vace-scale 1 --seed 1 --device "$device" \
  --output "$work/crop/generated.mp4"

python tools/crop_pose_vace.py composite --work-dir "$work/crop" \
  --generated-video "$work/crop/generated.mp4" --origin-video "$origin_video" \
  --output "$work/person_vace.mp4"
```

有显存压力时加 `--low-vram`，预算是当前空闲显存减 4 GiB；它会使用 CPU/磁盘卸载，速度较慢。不要用显卡总容量代替空闲量。当前单段推理要求帧数 `4n+1`；29 帧可直接执行。其他长度需在控制和 mask 中同步重复末帧补齐，推理后截回原长度；长视频需在有重叠的窗口推理并明确处理接缝，不能无声改变 FPS 或丢帧。

当前 CLI 另提供 `--model-gpus 0 1 2 3`（不可与 `--low-vram` 同用）：第一个 GPU 放文本编码器/VAE 和主张量，最后一个放 VACE，中间 GPU 按连续 DiT blocks 分担权重。通过 Accelerate 的设备对齐 hook 让各块输出回主设备；该配置应以实际完整推理验证。

DiffSynth 另外有 `use_usp=True` 的序列并行实现，含 VACE forward，但本项目 CLI 未接入 USP。它主要分摊序列/激活与计算，不能宣称会自动拆分模型权重。若 OOM 发生在权重加载阶段，需卸载或另外验证模型分片实现；仅设置 `CUDA_VISIBLE_DEVICES` 或直接启动多个独立进程不能聚合显存。并行改造需要验证 rank 绑定、相同 seed/input、输出仅由 rank 0 写入，以及集体通信结束后的清理。

`composite` 先将整幅生成 crop 逆缩放回固定窗口，然后只复制原始逐帧 box 矩形中的像素。输出编码统一 libx264/High/YUV420P/CRF18/faststart。

## 验证与交付

- 解码控制、mask、生成、最终视频，检查帧数、FPS、尺寸；验证 mask 阈值后逐帧矩形一致。
- 检查首、中、末帧，必要时看连续播放。反馈人物可见度、动作和边缘融合，不以模型退出码代替质量检查。
- `composite` 在编码前验证框外 decoded origin 像素不变，输出 PNG 诊断帧；有损 MP4 允许编码误差。
- 用 FFmpeg 检查最终 `Video: h264 (High), yuv420p`。保留提示词、seed、参数、裁剪变换、日志及原始生成视频。
- 交付最终视频和必要的诊断链接。若质量不足，说明观察到的具体问题；额外大规模参数搜索需有任务依据。
