---
name: register-person-vace
description: Register an animated FBX person into a calibrated source video at a world XYZ/LWH box, then blend the character with local Wan VACE using a cropped pose control, edit mask, and sharp first-frame reference. Use for person insertion and animation integration into recorded scenes.
---

# 人物动画注册与 VACE 环境融合

把用户提供的原始视频、人物动画 FBX、世界坐标 XYZ、包围框 LWH 转为人物注册与环境融合后的视频。使用项目中的本地权重与标定，不修改原始数据。

## 输入与坐标

- 获取视频、FBX、XYZ/LWH（米），以及对应的 `source-data`。可从当前上下文或数据索引定位标定，不要求用户重复提供已有信息。
- `source-data` 需要逐帧 `intrinsics/`、`extrinsics/`、`ego2global/`、对应 `images/`；视频尺寸、帧数和标定必须一致。缺少标定时先说明缺少哪些数据，不能从视频虚构相机或把像素位置当世界坐标。
- XYZ 是人物 box **中心**，世界原点为该片段第一帧自车，X 右、Y 前、Z 上。LWH 沿参考世界 XYZ，heading 默认为绕 +Z 的 0 弧度。用户指定其他约定时先转换。
- FBX 使用首帧网格 AABB 做一次固定等比缩放和中心对齐，保留之后的骨骼运动；不逐帧重新归一化，也不把人物固定在相机坐标。

## 执行

定位包含 `tools/register_fbx_animation.py`、`tools/crop_pose_vace.py`、`tools/render_fbx_reference.py` 和 `wan_vace_inference.py` 的 apollo-forge checkout。默认可查 `/workspace/apollo-forge`；找不到时说明依赖缺失。Blender 渲染助手和配置来自用户指定的 asset-forge checkout，通常是 `../asset-forge`。

按 [执行流程](references/workflow.md) 完成：

1. 校验输入与标定，渲染人物注册预览。
2. 从 FBX 的逐帧骨骼投影出 DWPose 风格彩色身体姿态；人物 box 的二维投影矩形置黑，框外保留背景。
3. 直接重新渲染高清 FBX 动画首帧作为外观参考。不能放大远处几十像素的人物渲染充当高清参考。
4. 用全片人物 box 并集确定一个固定、有上下文余量的裁剪窗口，优先读取索引定位的原始高分辨率图像。同步变换背景、姿态、mask；默认局部推理 768×768，按显存和人物有效像素调整。
5. 运行 VACE：控制视频 + 白色编辑/黑色固定 mask + 首帧参考图 + 泛化的人物渲染提示词。只读已有权重，先导入 `setup/checkpoint_environment.sh`。
6. 逆变换生成的局部视频，按原始逐帧人物 box mask 回填到原视频。裁剪窗口用于推理上下文，不代表整个窗口都允许修改。
7. 检查视频编码、帧对齐和视觉效果，交付最终 MP4、局部生成视频以及便于复现的参数。

## 必须保留的约束

- 相机/人物/背景使用同一帧、同一裁剪与缩放。使用实际捕获 FPS 推进 FBX 动画；保持输出时长。
- mask 白色 255 表示可编辑；黑色 0 表示固定。视频压缩后使用灰度 >=128 判定，并校验阈值后区域仍等于预期矩形。
- 使用同一个固定裁剪窗口覆盖该片段，避免逐帧追踪裁剪引入虚假相机运动；必要时明确记录可逆的逐帧变换。
- 最终交付 H.264 High、`yuv420p`、MP4、`+faststart`，保留 origin 分辨率/FPS。不要把 RGB/4:4:4 的中间文件当兼容交付视频。回填前可验证 mask 外像素完全一致；有损编码后不能宣称逐像素无损。
- 身体关键点直接取 FBX 骨骼，不能编造不存在的面部/手指关节。当前映射支持 Mixamo，Head 是头部代理点；其他 rig 需明确配置映射。
- GPU 显存按**当前空闲量**评估。遇 OOM 先检查占用；可换空闲卡或启用 `--low-vram`，必要时降低局部尺寸。保留失败日志；不干预其他任务。不要无边界重跑：一次正常运行失败后做有依据的资源调整；再次失败则保存可复现输入并说明阻塞。
- 逐帧检查 mask 外回填不变、骨骼有动画变化、人物是否实心/清楚、手脚姿态和脚底接地。若仍出现虚影或错误动作，如实交付并说明限制，不把完成推理等同于融合效果合格。
