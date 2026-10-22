---
name: register-person-vace
description: Preview and confirm a calibrated world 3D box, select and fit an animated FBX person, project its moving 3D skeleton as DWPose, and blend the person into the original video with local Wan VACE and an appearance reference. Use for person registration, motion replacement, and character insertion in recorded scenes.
---

# 人物注册、DWPose 与 VACE 融合

遵循用户指定的顺序：**3D 框预览并确认位置 → 选择 FBX → 检查完整动作是否装得下 → 注册模型 → 从 3D 骨骼投影 DWPose → VACE 融合与原视频回贴**。读取 [执行流程](references/workflow.md)；提示词采用 [写实提示词](references/prompts.md)，根据用户的外观与风格要求调整。

项目提供 `interface/register_person.py` 时优先使用 `preview → prepare → run` 统一入口，已有生成用 `compose`；参数见项目 `docs/person-registration.md`。项目中的 `renders/inspect_fbx.py` 与 `utils/person_box.py` 是范围检查和画框的正式入口，skill 自带脚本仅作缺少这些入口时的后备。不要在 tests 输出目录再写固定片段的推理脚本。

## 位置与动作范围

- 输入是原视频、同帧标定、世界 XYZ/LWH、FBX、外观参考图。复用已知参数，缺失的从项目索引获取。先交付只画用户指定 3D 框的预览；不要把场景已有车辆框当作人物框。
- 若用户尚未确认位置，展示预览并等其确认后再选择/注册模型；这是本流程的位置确认步骤。已确认的参数或“沿用上一轮位置”的授权可直接复用，变更 XYZ/LWH/heading 后更新预览。不要在框还不可见时先跑 VACE。
- XYZ 是 box 中心，单位米；世界原点为片段首帧自车，X 右、Y 前、Z 上。LWH 沿 box 局部 XYZ，heading 绕 +Z。底面高度是 `Z-H/2`，不等于人物脚底高度。
- 选择 FBX 后，**必须按输出 FPS 和帧数检查整段实际播放动作**，包括根节点运动、抬手、跨步、转身和循环接缝；首帧能放下不代表整段能放下。使用 `scripts/inspect_fbx.py` 得到首帧拟合后的超框量、全动作并集及可容纳整段的固定拟合方案。
- 全动作拟合只计算一次固定等比缩放与中心，不逐帧重新归一化。超框时说明超出的轴向和尺寸：可以在原框内缩小人物，或调整框与位置；明显改变已确认的人物尺度/位置前展示修订预览。放大 LWH 会同时影响自动归一化，不能假定把框等比放大就能消除动作越界。
- 带网格 FBX 使用逐帧 evaluated mesh bounds；纯骨骼 FBX 使用真实骨骼端点范围。纯骨骼可以生成 DWPose，但不能声称已渲染出 FBX 网格；若需要实体注册预览，使用用户指定的兼容角色/蒙皮，或说明当前只能预览骨架。

## 几何与时间约束

- 需要逐帧 `intrinsics/`、`extrinsics/`、`ego2global/` 和与原视频对齐的 `images/`。参考世界到相机：`extrinsic_t @ inv(ego2global_t) @ ego2global_0`。不能凭像素位置虚构世界坐标。
- mesh、骨骼、参考位置、mask 和背景使用同一标定、相同固定归一化、帧号、裁剪与缩放。以真实视频 FPS 推进 FBX 动画；记录动画 FPS、起止和循环行为。
- DWPose 是 FBX 真实 3D 骨骼的逐帧 2D 相机投影，不从实体渲染视频重新检测姿态。Mixamo 映射按骨骼名去命名空间；Head 仅为头部代理点，不编造缺失的面部、手指关节。
- 控制视频：原场景背景上，将固定世界 box 每帧的完整 2D 投影矩形置黑，再绘制彩色姿态。mask 为同一矩形，白色可编辑、黑色固定。依据 pose JSON 生成，不能用 mesh silhouette 替代用户要求的 box mask。

## 外观与推理

- 用户提供的参考图优先。需要去背景时使用可用图像编辑工具，检查人物身份、细节与尺寸是否变化。保留原图；记录衍生参考图和编辑方式。
- 参考图必须等比缩放，按目标人物区域安排位置并补中性灰边，最终尺寸与推理画布一致。检查整个人物、手脚可见且有效人物像素足够。RGBA 先合成灰底再送 VACE，不能直接丢掉 alpha；已有合格参考图直接复用。无外观图且有 mesh 时可重新渲染高清首帧，不能放大远处几十像素的角色充当高清参考。
- 写实需求使用“photorealistic real human person”，并明确参考图身份/服装。避免正向使用有歧义的“animated person”；负向包含 animated character、cartoon、anime、illustration、CGI。提示词不能保证身份严格复现，需检查结果。用户要求时提供实际完整中英文提示词。
- 保留用户指定的分辨率；人物有效像素较少时可采用固定局部裁剪，控制、mask、参考图同步变换并记录可逆变换。裁剪不是扩大编辑权限。
- 先 source 项目 checkpoint 环境配置；完整 `/tmp/vace-model-cache` 存在时优先本地缓存，核验七个 DiT shards、UMT5、VAE 和 tokenizer。不要每轮重新复制权重，不把权重写入仓库。
- 按当前空闲显存选择 GPU。采用一次正常推理；OOM 时保留日志，只做一次有依据的显存调整。不要干预其他任务，不无边界重跑。用户要求退出会话继续时使用独立后台进程，记录 PID、日志和完成状态；未完成单组重新采样不等于从中间去噪步续跑。

## 交付

- VACE 原始输出只按逐帧 box mask 回贴到原视频；先验证编码前 mask 外像素保持原视频，再编码最终 MP4。有损视频不能宣称逐像素无损。
- 交付兼容 MP4：H.264 High、yuv420p、faststart，保持原视频帧数/FPS/尺寸。4:4:4 无损姿态可作控制中间件，另提供兼容预览。
- 默认提供上下对比，上方为本轮 DWPose、下方为本轮融合结果，按帧号对齐。首、中、末帧和连续动作检查人物实体、服装、手脚、脚底、边缘及背景漂移；质量不足如实说明。
- 参数与日志记录原视频、FBX、XYZ/LWH/heading、bounds/scale/center、动画时序、参考图、控制、mask、提示词、seed、模型路径/GPU、裁剪以及各输出。更换动作使用不同名字，避免覆盖其他动作；仅在用户要求时清理旧产物。
