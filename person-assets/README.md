# 人物注册资产库

集中保存人物外观参考、FBX 动作和动作预览。场景注册、DWPose 控制、mask 和 VACE 生成结果保存在各任务输出目录。

```text
person-assets/
├── references/  # 人物参考图，不登记 CSV，不生成预览视频
├── fbx/         # 动作模型
├── previews/    # 每个 FBX 对应的同名 MP4
├── index.csv    # 只索引动作资产和预览的对应关系
└── README.md
```

## 动作索引

每个 `fbx/<motion_id>.fbx` 必须对应且仅对应一个 `previews/<motion_id>.mp4`；文件名使用小写英文与连字符。CSV 中路径相对于本目录。

| motion_id | FBX | 预览 | 内容 |
| --- | --- | --- | --- |
| bboy-uprock-start | [bboy-uprock-start.fbx](fbx/bboy-uprock-start.fbx) | [bboy-uprock-start.mp4](previews/bboy-uprock-start.mp4) | 仅动画骨架，实体注册需要兼容蒙皮角色 |
| waving-1 | [waving-1.fbx](fbx/waving-1.fbx) | [waving-1.mp4](previews/waving-1.mp4) | 挥手动作 1，带网格和动画骨架，约 4.77 秒 |
| waving-2 | [waving-2.fbx](fbx/waving-2.fbx) | [waving-2.mp4](previews/waving-2.mp4) | 挥手动作 2，带网格和动画骨架，约 0.57 秒 |
| waving-3 | [waving-3.fbx](fbx/waving-3.fbx) | [waving-3.mp4](previews/waving-3.mp4) | 挥手动作 3，带网格和动画骨架，约 3.20 秒 |

`index.csv` 字段：`motion_id`、`name`、`fbx_path`、`preview_path`、`has_mesh`、`description`。增加动作时将 FBX 放进 `fbx/`，生成同名预览，并登记 CSV；参考图不写入这个索引。

## 人物参考图

- [traffic-officer-reference.png](references/traffic-officer-reference.png)：去蓝底、中性灰背景、832×480 的 VACE 参考图。
- [black-hoodie-reference.png](references/black-hoodie-reference.png)：黑色连帽外套、浅色长裤人物；由 `tests/1908389128.png` 提取人物后按标准排版，832×480 灰底，保留原始姿势与外观。

参考图独立管理，可以用于多个动作；不要求与 FBX 或 MP4 同名，也不绑定动作索引。

### 转换成标准参考图

标准格式为 832×480、RGB(127,127,127) 灰底、人物完整且等比缩放。默认读取 `traffic-officer-reference.png` 的人物位置与占比；输入人物的姿势和外观保持原样。转换不会修改 CSV。

```bash
python tools/convert_person_reference.py --input /path/to/person.png \
  --output person-assets/references/person-reference.png
# 蓝底图可自动识别，也可明确指定
python tools/convert_person_reference.py --input /path/to/blue-screen.jpg \
  --background-mode blue --output person-assets/references/person-reference.png
# 复杂背景需提供人物 mask，白色为人物
python tools/convert_person_reference.py --input /path/to/photo.jpg \
  --mask /path/to/person-mask.png --output person-assets/references/person-reference.png
```

支持透明 PNG、与边界连通的蓝底和纯色背景；背景颜色与衣服接近时建议使用透明图或外部 mask。`--template` 可指定其他格式样图，`--no-template` 使用默认居中留白，`--width/--height` 改画布，`--overwrite` 显式覆盖已有输出。不会把非 T-pose 输入合成为 T-pose。

## 根据 FBX 生成预览

在项目根目录执行；安装 `setup/requirements-tools.txt` 并准备 Blender。脚本通过 `--blender`、`BLENDER_BIN`、PATH 或本机安装目录发现 Blender。

```bash
# 自动输出到 person-assets/previews/bboy-uprock-start.mp4；覆盖旧视频需显式指定
python tools/preview_fbx.py --fbx person-assets/fbx/bboy-uprock-start.fbx --overwrite

# 也支持库外 FBX 和自定义输出
python tools/preview_fbx.py --fbx /path/to/person.fbx --output outputs/person-preview.mp4
```

默认 640×640、原始动画 FPS、H.264 High/yuv420p/+faststart，播放完整动作。固定相机按全动作范围取景；有网格时保留材质，纯骨骼显示彩色骨骼。`--size`、`--samples` 调整尺寸和采样。临时帧自动清理；运行日志和元数据默认保存在 `outputs/fbx-previews/`，可用 `--metadata-dir` 覆盖。

## 用于人物注册

```bash
python interface/register_person.py preview \
  --source-data data/waymo-asset-data/clip-00610-00/source-data \
  --origin-video data/waymo-video-data/clip-00610-00/videos/video_origin.mp4 \
  --xyz 0.11610073 26.53805042 1.80765938 --lwh 5 5 10 \
  --work-dir outputs/person-bboy
# 查看框预览并确认位置后
python interface/register_person.py prepare --work-dir outputs/person-bboy \
  --confirm-position --fbx person-assets/fbx/bboy-uprock-start.fbx --fit-bounds clip \
  --reference-image person-assets/references/traffic-officer-reference.png
```

XYZ/LWH 是示例片段的参数，其他片段应重新确认三维框，并检查完整动作是否超框。完整流程见 [人物注册接口](../docs/person-registration.md)。
