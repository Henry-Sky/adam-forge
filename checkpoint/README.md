# 模型权重目录

此目录用于通过软链接访问外部模型权重，不要将权重文件提交到 Git。当前本地权重来源根目录为 `/mnt/data/asset-forge/checkpoints`；当前分支未包含 `setup/checkpoint_environment.sh`，以下链接直接指向本机已有模型目录。

当前本地软链接：

| 模型 | 仓库内路径 | 外部目标目录 |
| --- | --- | --- |
| Qwen | `checkpoint/Qwen-Image-Edit-2511` | `/mnt/data/asset-forge/checkpoints/Qwen-Image-Edit-2511` |
| Harvester | `checkpoint/asset-harvester` | `/mnt/data/asset-forge/checkpoints/asset-harvester` |
| SAM3 | `checkpoint/sam3` | `/mnt/data/asset-forge/checkpoints/sam3` |

软链接及模型权重由 `.gitignore` 排除，此目录仅跟踪本 README。以上链接依赖本机外部存储；请将目标目录作为只读模型来源使用。
