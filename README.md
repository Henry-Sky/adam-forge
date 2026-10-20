# AdamForge

用于将 FBX 人物动画注册到视频场景，并准备 Wan VACE 局部修复输入的工具集。当前仓库主要由独立脚本组成；整体设计与尚未实现的阶段见[工作流说明](docs/adam-forge-workflow.md)。

## 目录

| 路径 | 内容 |
| --- | --- |
| `tools/` | 视频、轨迹、FBX、Pose 与 VACE 的独立命令行工具 |
| `setup/` | 推理环境安装、检查点路径配置和工具依赖 |
| `configs/` | 项目配置及任务配置 |
| `interface/` | 面向使用者的独立任务入口（目前为空） |
| `renders/` | Blender 调用的渲染脚本（目前为空） |
| `utils/` | 可复用模块（目前为空） |
| `tests/` | 测试代码及相关小型样例、实验结果；目录内 README 说明其组成 |
| `data/` | 数据集挂载点，具体数据由外部路径提供 |
| `checkpoint/` | 模型权重挂载点，权重不得提交到仓库 |
| `third_party/` | 外部依赖源码及其许可证说明 |
| `logs/` | 本地安装、渲染和推理日志，不作为源码提交 |
| `docs/` | 工作流和项目文档 |

## 环境与使用

先查看安装脚本的参数和示例，再按需安装 Wan VACE 依赖：

```bash
bash setup/install_environment.sh --help
bash setup/install_environment.sh --dry-run
```

常用脚本都可用 `--help` 查看输入、输出和参数。例如：

```bash
python tools/convert_track_format.py --help
python tools/register_fbx_animation.py --help
python tools/prepare_pose_vace.py --help
```

Waymo 数据路径由 `data/` 下的符号链接提供。模型路径可通过环境变量覆盖，默认值见 `setup/checkpoint_environment.sh`。请勿把数据集、模型权重、机器专属配置或运行产物提交到 Git。

`tests/` 目录按项目配置在 Git 中忽略，仅用于本地测试与样例。测试说明保留在本地目录内。
