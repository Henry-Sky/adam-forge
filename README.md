# AdamForge

用于将 FBX 人物动画注册到视频场景，并准备 Wan VACE 局部修复输入的工具集。当前仓库主要由独立脚本组成；整体设计与尚未实现的阶段见[工作流说明](docs/adam-forge-workflow.md)。

## 目录

| 路径 | 内容 |
| --- | --- |
| `tools/` | 独立的三维框绘制、动作预览、局部裁剪和长视频分段推理工具 |
| `setup/` | 推理环境安装、检查点路径配置和工具依赖 |
| `configs/` | 项目配置及任务配置 |
| `interface/` | 人物注册分阶段接口和 Wan VACE 推理入口 |
| `renders/` | Blender 注册助手和 FBX 动作范围检查 |
| `utils/` | 几何、视频回贴和注册工作流公共模块 |
| `person-assets/` | 人物参考图、FBX 动作、同名预览视频与动作 CSV 索引 |
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
python tools/draw_3d_boxes.py --help
python tools/preview_fbx.py --help
python tools/convert_person_reference.py --help
python interface/register_person.py --help
```

Waymo 数据路径由 `data/` 下的符号链接提供。模型路径可通过环境变量覆盖，默认值见 `setup/checkpoint_environment.sh`。请勿把数据集、模型权重、机器专属配置或运行产物提交到 Git。

人物注册使用 `interface/register_person.py` 的 `preview → prepare → run` 三阶段，支持已有 DWPose 输入复用和单独 `compose`。完整参数、安装、运行及测试命令见[人物注册接口说明](docs/person-registration.md)。

`tests/` 存放本地输入素材与实验结果，默认由 Git 忽略。
