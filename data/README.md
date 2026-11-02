# 数据目录

此目录用于通过软链接访问外部数据，不存放或提交数据集本体。当前本地软链接：

| 数据 | 仓库内路径 | 外部目标目录 |
| --- | --- | --- |
| 人物对齐数据 | `data/adam-data` | `/mnt/data/adam-forge/adam-data` |

人物对齐数据通过 `data/adam-data` 软链接访问。`source-data/` 保存筛选出的 455 个 clip 源视频、PKL、caption 和元数据；`train-data/` 当前仅保留用户确认的824个审查通过对象训练对（原1,653个，按用户确认先后删除140个原1类、660个面积小于2000 px²、8个视觉0类及21个人工排除项），每个目录以 `metadata.csv` 的五位数字 clip ID 和五位 obj ID 命名（`<clip_id>_<obj_id>`），含 29 帧完整场景 RGB 图像及逐帧人物框。人物 crop 与场景视频的批处理入口及贴回方法见 [`tools/README.md`](../tools/README.md#人物投影-crop-与场景视频可脱离-codex-运行)。每个训练对新增 `scene_rgb.mp4`、`person_crop/person.mp4` 和逐帧位置记录 `person_crop/placement.json`，并在 `manifest.json` 记录路径；原始 `scene_rgb/*.jpg` 保留。crop 基于对应 ID 的 3D 框投影，合并同 ID 的 2D 框以防标注偏移导致截断，使用原尺寸像素和固定补黑画布；位置记录支持修改后贴回。退化图尚待补充。链接目标由外部存储提供，不纳入版本控制。

软链接及数据由 `.gitignore` 排除，此目录仅跟踪本 README。以上链接依赖本机外部存储；迁移环境时需按实际存储位置配置链接。Waymo 数据及 `data/3d-person-assets` 当前未在本目录配置，使用前需确认相应外部来源。

将数据软链接及其目标数据作为只读来源：未经明确授权，不修改、删除、移动、替换或重定向链接及源数据。筛选清单、截图和重建结果写入 `outputs/` 或临时目录，不写回数据目录。

本次824条保留名单及删除记录存放在本机忽略目录 `outputs/crop_review_20261102_final/`；原审查预览目录 `tests/crop_review_20261102/` 已按用户要求删除。
