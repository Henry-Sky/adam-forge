# 数据目录

此目录用于通过软链接访问外部数据，不存放或提交数据集本体。当前本地软链接：

| 数据 | 仓库内路径 | 外部目标目录 |
| --- | --- | --- |
| L3 源数据 | `data/l3-source-data` | `/mnt/data-dxh/dataconvert/L3-xirang-v5/L3_V5data_3` |

L3 源数据目标目录包含 `clip_*` 场景目录及 `metadata.csv`，场景目录包含各相机图片和标注。访问时使用仓库相对路径 `data/l3-source-data`；`metadata.csv` 中的 `clip_id` 编号与 `source_file` 场景目录名对应。

软链接及数据由 `.gitignore` 排除，此目录仅跟踪本 README。以上链接依赖本机外部存储；迁移环境时需按实际存储位置配置链接。Waymo 数据及 `data/3d-person-assets` 当前未在本目录配置，使用前需确认相应外部来源。

将数据软链接及其目标数据作为只读来源：未经明确授权，不修改、删除、移动、替换或重定向链接及源数据。筛选清单、截图和重建结果写入 `outputs/` 或临时目录，不写回数据目录。
