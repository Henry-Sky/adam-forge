# 数据目录

此目录用于通过软链接访问外部数据，不存放或提交数据集本体。当前本地软链接：

| 数据 | 仓库内路径 | 外部目标目录 |
| --- | --- | --- |
| L3 源数据 | `data/l3-source-data` | `/mnt/data-dxh/dataconvert/L3-xirang-v5/L3_V5data_3` |
| 人物对齐数据 | `data/adam-data` | `/mnt/data/adam-forge/adam-data` |

L3 源数据目标目录包含 `clip_*` 场景目录及 `metadata.csv`，场景目录包含各相机图片和标注。访问时使用仓库相对路径 `data/l3-source-data`；`metadata.csv` 中的 `clip_id` 编号与 `source_file` 场景目录名对应。

人物对齐数据通过 `data/adam-data` 软链接访问。`source-data/` 保存筛选出的 455 个 clip 源视频、PKL、caption 和元数据；`train-data/` 当前保存 1,653 个对象训练对，每个目录以 `metadata.csv` 的五位数字 clip ID 和五位 obj ID 命名（`<clip_id>_<obj_id>`），含 29 帧完整场景 RGB 图像及逐帧人物框。人物 crop 与退化图尚待补充。链接目标由外部存储提供，不纳入版本控制。

软链接及数据由 `.gitignore` 排除，此目录仅跟踪本 README。以上链接依赖本机外部存储；迁移环境时需按实际存储位置配置链接。Waymo 数据及 `data/3d-person-assets` 当前未在本目录配置，使用前需确认相应外部来源。

将数据软链接及其目标数据作为只读来源：未经明确授权，不修改、删除、移动、替换或重定向链接及源数据。筛选清单、截图和重建结果写入 `outputs/` 或临时目录，不写回数据目录。
