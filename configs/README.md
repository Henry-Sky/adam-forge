# 项目配置

## Wan2.1-VACE 人物擦除配置

`wan_vace_background.yaml` 用于 `tools/build_scene_background.py`，以选中 `adam-<clip_id>-<obj_id>` 目录的 `scene_input.mp4` 和 `scene_mask.mp4` 作为 VACE 视频和白色重绘区；由 `video` / `mask` 指定，可改为 `scene_liner.mp4` 对比首轮方案。默认 BF16、30 步、CFG 5、VACE scale 1、sigma shift 5、seed 0、VAE tiled；提示词描述连续背景，负面提示词抑制编辑区域的人形、残影和闪烁。

所有源帧保持原分辨率，右侧/底部仅补至 16 的倍数，RGB 复制边缘、Mask 补黑；不缩放、不截取帧段。VACE 生成后裁回原尺寸，仅将白色 Mask 内像素贴回解码的 `scene_rgb.mp4`，编码前 Mask 外像素严格保持一致。默认输出 `scene_background.mp4` 使用下述视频格式及 CRF 18。多 GPU 运行、日志和校验说明见 [`tools/README.md`](../tools/README.md#vace-指定人物擦除与完整场景背景视频)。

## Wan2.1-VACE 本地冒烟配置

`wan_vace_smoke.yaml` 用于 `tests/test_wan_vace.py`，所有模型和输入路径均相对仓库根目录。默认读取 `checkpoint/Wan2.1-VACE-14B` 及人物重建样本 `adam-00000-00057` 的 `scene_liner.mp4`、`scene_mask.mp4`。RGB 缩放使用 bicubic，Mask 先解码为全范围灰度、再使用最近邻缩放并检查严格二值；白色区域为 VACE 生成区。

默认 384×256、9 帧、4 步、BF16、64 CPU 线程，仅用于安装后的端到端检查；不代表正式生成参数。分辨率必须为 16 的倍数，帧数为 4n+1；输入至少含指定帧数，测试按起始帧连续读取。`prompt`、`negative_prompt`、`seed`、`cfg_scale` 和 `num_inference_steps` 均显式记录。环境、沙箱外 GPU 执行与输出说明见 [`setup/README.md`](../setup/README.md#wan21-vace-本地配置与测试)。

## 默认视频编码格式

项目视频默认采用以下格式；分辨率不作限制，按任务需求选择。

| 参数 | 默认要求 |
| --- | --- |
| 容器 | MP4（`.mp4`） |
| 视频编码 | H.264 / AVC，编码标记 `avc1` |
| Profile / Level | High / 4.0；较高分辨率或帧率需要时允许调整 Level |
| 像素格式 | `yuv420p`（8-bit、4:2:0） |
| 帧率 | 恒定 10 FPS；任务明确要求时可覆盖 |
| 扫描方式 | 逐行扫描 |
| 音频 | 默认不包含音轨 |
| 分辨率 | 不限制，按任务需求选择 |
| 码率 | 不固定，按画质和任务需求配置 |

默认格式描述输出约定，不表示当前所有工具已自动配置；新增或修改视频输出入口时应显式遵循，并在任务需要覆盖默认值时记录原因。

`scene_noise_input.yaml` 配置人物矩形扩边、7 帧平滑和固定场景坐标噪点。`scene_input.mp4` 使用与源相同的 YUV420 管线、默认 H.264 High/yuv420p 及 CRF 10，保留源颜色元数据，替代旧 v1 的 RGB/gbrp 编码。解码后噪点允许少量灰阶；同步输出的 `scene_mask.mp4` 为严格二值，继续采用无损编码。运行、颜色验收及旧文件备份说明见 [`tools/README.md`](../tools/README.md#人物矩形噪点输入与稳定遮罩)。

## 人物配对退化

`person_degradation.yaml` 配置 `tools/degrade_person.py`：随机种子、每片段变体数、分支组合权重、温和detail及独立legacy处理参数、ISP 采样范围及二维空间光照场。所有二元数组为均匀采样的 `[最小值, 最大值]`，上下限相等表示固定值。模式在每个片段变体开始时选定，全部帧固定参数；光照场绑定因果平滑的人物 Mask 外接框。`bbox_smoothing` 为上一帧框的权重。`wb_*_ev` 和 `light_*_ev` 是通道增益的 log2 值；ISP 白平衡增益按亮度权重归一化。`direction_deg=0` 对应向图像右侧增亮；`center_x/y`、`sigma` 使用人物框归一化坐标。gamma 在此表示线性 RGB 的幂指数；通道裁剪概率独立于其它 ISP 操作。

参数范围是初始试验配置，尚未做跨场景视觉调优。二维光照场叠加到原有光照，不进行物理重光照或地面投影阴影生成。详细输入输出、限制和验证命令见 `tools/README.md`。

版本2的组合顺序固定为二维光照 → ISP → detail。detail仅NLM去噪（h=6）与flat（mix=0.3）；旧完整链保存在legacy节，legacy模式权重0且只能独立执行。各模式均须声明legacy布尔字段。版本1配置不再由新入口加载，旧版预览与配置快照可继续用于历史比较。

`person_degradation_detail_medium.yaml` 是中等 detail 强度的独立试验配置：相较默认配置，仅将 `detail.denoise_h` 从6调到10，`detail.flat_mix` 从0.3调到0.55；随机种子、ISP、二维光照和其余参数保持一致。用于对同一人物逐帧比较增强去噪和平滑的效果，默认配置不变。示例：

```bash
OPENBLAS_NUM_THREADS=1 .conda/adam-gem-x/bin/python -B tools/build_scene_person_degradation.py \
    --pair-dir data/adam-data/train-data/00000_00057 \
    --config configs/person_degradation_detail_medium.yaml \
    --output-dir tests/scene_person_degradation_20261103_00000_00057_detail_medium --threads 64
```
