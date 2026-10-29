# 项目 Skills

本目录保存可版本管理的项目 skill。当前可用：

- [`gaussian-character-motion`](gaussian-character-motion/SKILL.md)：场景与人物筛选、Qwen A-pose、Harvester Gaussian 重建、GEM-X/SOMA、网页人工配准、权重转移、LBS 人工确认和源动作恢复。

仓库中的版本用于共享与维护；本机可发现副本位于 `${CODEX_HOME:-$HOME/.codex}/skills/`。创建或修改时同步完整 skill 目录，包括 references、scripts、assets 和元数据，并校验两份目录内容一致。以下命令在仓库根目录执行，将仓库版本同步到本机（若目标已有其他改动，先比较并合并）：

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills/gaussian-character-motion"
cp -a skills/gaussian-character-motion/. "${CODEX_HOME:-$HOME/.codex}/skills/gaussian-character-motion/"
diff -qr skills/gaussian-character-motion "${CODEX_HOME:-$HOME/.codex}/skills/gaussian-character-motion"
```

使用 skill-creator 自带校验器；它位于 `.system/skill-creator/`，依赖 PyYAML。使用已按 [`setup/README.md`](../setup/README.md) 安装的 GEM-X 环境：

```bash
conda activate adam-gem-x
python "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py" skills/gaussian-character-motion
python "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py" "${CODEX_HOME:-$HOME/.codex}/skills/gaussian-character-motion"
```

校验器检查格式与元数据，不会执行重建推理。流程中的配准确认、LBS 确认和产物检查仍按 skill 执行。`outputs/` 下的历史产物仅是可选案例，迁移仓库后可能不存在。
