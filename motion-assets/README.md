# 动作资产库

存放人物动作资产及完整动作预览。文件名采用小写英文与连字符，避免空格、括号和导出编号。FBX 放 `fbx/`，对应的同名 MP4 预览放本目录，预览参数与日志放 `metadata/`。场景注册、DWPose 和 VACE 结果放在各自任务的输出目录。

| FBX | 预览 | 内容 |
| --- | --- | --- |
| [fbx/waving.fbx](fbx/waving.fbx) | [waving.mp4](waving.mp4) | 挥手；带网格和动画骨架，预览显示实体 |
| [fbx/bboy-uprock-start.fbx](fbx/bboy-uprock-start.fbx) | [bboy-uprock-start.mp4](bboy-uprock-start.mp4) | Bboy Uprock 起始动作；仅动画骨架，预览显示彩色骨骼，实体注册需兼容蒙皮角色 |

预览使用固定相机、中性背景和整段动画范围归一化，以 FBX 导入帧率播放。输出为 640×640、H.264 High、yuv420p MP4。

重新生成预览（在项目根目录执行，设置可用的 Blender 路径）：

```bash
python tools/preview_motion_asset.py \
  --fbx motion-assets/fbx/waving.fbx --output motion-assets/waving.mp4 \
  --blender /path/to/blender
python tools/preview_motion_asset.py \
  --fbx motion-assets/fbx/bboy-uprock-start.fbx --output motion-assets/bboy-uprock-start.mp4 \
  --blender /path/to/blender
```

使用人物注册流程时引用这里的路径，例如：

```bash
python interface/register_person.py preview \
  --source-data data/waymo-asset-data/clip-00610-00/source-data \
  --origin-video data/waymo-video-data/clip-00610-00/videos/video_origin.mp4 \
  --xyz 0.11610073 26.53805042 1.80765938 --lwh 5 5 10 \
  --work-dir outputs/person-waving
# 查看框预览并确认位置后：
python interface/register_person.py prepare --work-dir outputs/person-waving \
  --confirm-position --fbx motion-assets/fbx/waving.fbx --fit-bounds clip
```

执行前检查 Blender、渲染依赖和标定路径；XYZ/LWH 是该片段的示例参数，其他片段应先确认 3D 框并检查整段 FBX 动作范围。
