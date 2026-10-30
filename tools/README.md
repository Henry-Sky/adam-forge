# 独立工具

## 安装辅助工具

以下工具只依赖 Python 标准库，无命令行参数，通常由 `setup/` 的 Bash 安装脚本自动调用。
安装流程、依赖版本与下载源见 [`setup/README.md`](../setup/README.md)。

- `install_web_vendor.py`：安装并校验网页 Three.js / Spark 依赖，需要系统提供 curl；输出到忽略的 `web_server/src/static/.vendor/`，路径不受当前工作目录影响。
- `fix_decord_metadata.py`：使用目标环境的 Python 修复已安装 decord 0.6.0 的已知 wheel 标签及 RECORD 校验值，会修改该环境的包元数据。

在仓库根目录单独运行：

```bash
conda activate adam-web
python tools/install_web_vendor.py
conda activate adam-sam3
python tools/fix_decord_metadata.py
```

## 已配准模型的完整 LBS 绑定

`bind_soma_gaussians.py` 从网页保存/导出的共同 Z-up 空间 `soma.npz` 与 `gaussians.ply` 生成可驱动绑定，不再执行 ICP。使用 `adam-gem-x` 环境；依赖与安装入口见 [`setup/README.md`](../setup/README.md)。输入只读，输出目录必须不存在。

```bash
conda activate adam-gem-x
OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 CUDA_VISIBLE_DEVICES='' \
python tools/bind_soma_gaussians.py \
    --input-dir web_server/output/aligned-20261028T073920Z-165dcf96 \
    --assets checkpoint/GEM-X/soma-v0.3.3 \
    --output-dir outputs/aligned-gaussian-binding --top-k 8 --threads 64
```

`--top-k` 为每个 Gaussian 保留的骨骼数（1–32，默认 8）；`--threads` 为 SOMA/BLAS CPU 线程数（默认 64）；可用 `--progress /tmp/binding-progress.json` 输出进度。
工具通过匹配的 SOMA 0.3.3、v0.2.0 reference pose 和 legacy_pose021 procedural profile 重建网格并检查缓存误差，再捕获实际进入 LBS 的完整 110 关节世界变换。该兼容提取只在独立计算进程中临时包装方法，不修改第三方源码。

输出 `gaussians.ply`、`soma.npz`、`skinning.npz`、`skeleton.npz`、`binding.json`，以及主骨骼着色 PLY、左肘弯曲 PLY、示例目标骨架和网页检查数据。绑定采用当前配准姿态；`skeleton.npz` 中 `bind_world` 和 `bind_inverse` 均属于身份拟合后的当前完整骨架，不是模板 T-pose。

- `skinning.npz`：原 PLY 行序的 `gaussian_index`、`bone_indices[N,K]`、归一化 `weights[N,K]`、最近面/重心坐标/表面距离，以及距离超过 Z 轴身高 4% 的 `needs_review`。`discarded_weight` 记录截断前被舍弃的权重总和。
- `skeleton.npz`：`joint_names[J]`、`joint_parent_ids[J]`、`bind_world[J,4,4]`、`bind_inverse[J,4,4]`、`bind_local[J,4,4]`；程序扭转关节保留，不能用 77 个显示关节替代。
- `binding.json`：坐标、版本、输入/资产哈希、重建误差、绑定姿态回放误差、表面距离分位数及局限。

网页统一导出使用 `soma.npz`、`gaussians.ply`、`metadata.json`，有效绑定时附加 **`gaussian_binding.npz`**。此 NPZ 合并下述权重和骨架字段，并包含 `format=gaussian_lbs_binding_v1`、坐标及源文件哈希；根关节 parent 统一为 -1。网页导出不包含独立 `skinning.npz` / `skeleton.npz` 或预览文件。上面的多文件目录属于独立命令行工具的计算结果和检查数据。

`drive_gaussian_lbs.py` 在 `adam-web` 即可执行，只需 NumPy/SciPy。工具自动识别网页的 `gaussian_binding.npz` + `metadata.json`，也兼容命令行工具原有的双 NPZ 格式。目标变换 NPZ 必须包含与绑定顺序完全一致的 `joint_names` 和共同世界空间的 `joint_world[J,4,4]`。

```bash
conda activate adam-web
python tools/drive_gaussian_lbs.py \
    --binding outputs/aligned-gaussian-binding \
    --transforms outputs/aligned-gaussian-binding/preview_transforms.npz \
    --output-ply outputs/lbs-left-elbow.ply
```

驱动公式为 `D_j = target_world_j @ bind_inverse_j`。Gaussian 中心采用 LBS，协方差采用 `A C Aᵀ`（`A` 是加权线性变换），再分解成 PLY 的 wxyz 四元数与 log scales；DC 颜色、opacity、Gaussian 行序保持不变。近奇异混合的协方差特征值下限为 `1e-16`。目标骨架必须来自相同身份、局部尺度、global_scale、reference 和 procedural profile；不能直接将目标 public78 变换传入。

需要生成其他 SOMA 姿态的完整目标骨架时，在 `adam-gem-x` 中使用 `utils.soma_binding.SomaBindingRuntime(assets, threads=64)`，以更新了 `body_pose` / `global_orient` / `camera_translation` 的原始参数字典调用 `evaluate()`；返回的 `world` 即完整骨架，可连同 `runtime.names` 保存为上述 NPZ。导出时不要再次改变世界坐标基。

验证：

```bash
python -B -m unittest discover -s tests -p test_gaussian_lbs.py -v
```

最近表面插值可能在贴近的四肢、宽松衣物和附件处串权重；距离标记不能发现全部语义错误。此版本不含衣物物理、保体积 DQS 或姿态相关 Gaussian 外观。

## SOMA 与 Gaussian 配准、蒙皮权重转移

`align_soma_gaussians.py` 将 GEM-X 导出的 SOMA MID 人体网格对齐到
Asset Harvester 的 Gaussian PLY，并给每个 Gaussian 分配 SOMA 骨骼权重。
在仓库根目录使用 `adam-gem-x` 环境执行；环境安装见 [`setup/README.md`](../setup/README.md)。
依赖 NumPy、SciPy、trimesh、rtree、OpenCV 和 py-soma-x 0.3.3，均包含在该环境基础依赖中。

```bash
conda activate adam-gem-x
OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 \
python tools/align_soma_gaussians.py \
    --soma outputs/clip306-person32/gem-x-soma/soma_parameters.npz \
    --ply outputs/clip306-person32/harvester/reconstruction/person32/gaussians.ply \
    --assets checkpoint/GEM-X/soma-v0.3.3 \
    --output-dir outputs/clip306-person32/soma-gaussian-binding --threads 64
```

输入 NPZ 需要包含 `vertices_camera`、`faces`、`joints_camera`、
`joint_names`、`joint_parent_ids`，网格采用 SOMA MID 的原始顶点编号。
`--assets` 应与生成这些数据的 SOMA 版本一致。工具检查 NPZ 三角面与 USD 四边面
的顶点对应关系，避免错误地把权重赋给不同编号的网格。
`--output-dir` 必须是不存在的新目录；输入文件不会修改。
`--samples` 控制配准采样数，默认 12000；`--threads` 控制 KDTree 查询线程数，默认 64。
人工配准网页位于 [`web_server/`](../web_server/README.md)。
使用 `--transform <人工保存的 registration.json>` 可以跳过 ICP，按人工矩阵重新计算权重；
工具会验证源文件哈希和统一相似变换约定。
该流程执行 CPU 几何计算，不运行 GPU 推理。

配准首先补偿 Harvester 导出器的 X 轴 180° 旋转，按高度初始化尺度和中心，
随后通过截断距离的对称 ICP 求解统一尺度、旋转和平移。优化限制相对初始尺度
和旋转的变化，减少左右翻转和尺度坍缩。配准使用解码 opacity ≥0.1 的 Gaussian
作为候选并按 opacity 采样，权重转移仍覆盖原 PLY 的每一行。
这个轴向假设来自本项目 Harvester 导出器；其他工具生成的 PLY 需先确认坐标约定。

对每个 Gaussian，寻找配准网格上的最近三角面，按最近点的重心坐标插值三个顶点
的蒙皮权重，再归一化。输出保留全部 110 个变形关节，包括程序扭转关节。
77 个可控关节与这 110 个变形关节不是相同的索引空间。

| 输出 | 用途 |
| --- | --- |
| `soma_aligned.obj` | 配准到原 PLY 坐标系的 SOMA 网格 |
| `gaussian_binding.npz` | Gaussian 行号、完整权重、骨骼名称与父子关系、最近面、重心坐标、表面距离和检查标记 |
| `registration.json` | 输入哈希、4×4 配准矩阵、优化记录、距离统计和限制说明 |
| `alignment.png` | 正面和侧面 Gaussian 中心、网格采样点、骨架的几何检查图 |
| `bone_weights.png` | 以最大权重骨骼着色的检查图 |
| `gaussians_bone_colors.ply` | 保留原 Gaussian 位置、协方差参数、opacity 和行序，仅替换颜色的检查副本 |

NPZ 的主要约定：

- `weights[N,110]` 对应 `joint_names[110]`，每行非负且和为 1。
- `gaussian_index[N]` 与原始 PLY 行号一一对应。应用前应核对报告的 PLY SHA-256，避免绑定到不同点序。
- `soma_camera_to_ply[4,4]` 采用列向量约定，`p_ply = M @ [p_camera,1]`。
- `source_pose_vertices_ply` 和 `source_pose_joints_ply` 是变换后的源姿态几何，均在 PLY 归一化坐标系。
- `surface_distance` 的单位是 PLY 单位，除以报告的 `scale` 可换算到源 SOMA 米制尺度。
- `needs_review` 标记距离大于配准 Mesh 高度 4% 的 Gaussian；它只能发现部分几何异常。
- `public_weights_approx[N,78]` 把程序关节权重合并给最近的公开祖先（包含 Root），仅用于简化查看。精确变形应使用完整权重和匹配的程序关节变换。
- `template_bind_pose_world` 是 USD 原始模板绑定变换，保留原始约定；不能当作已拟合身份、源姿态在 PLY 空间的 inverse bind。

这是静态配准与权重初始化。衣物、安全帽及生成视图带来的形体差异不可能全部由
统一相似变换消除，距离统计也不是有真值的精度评估。启用动画前需要检查关节附近
权重，获得匹配的源/目标完整骨架变换，并正确处理 Gaussian 中心和协方差的变形。
本工具未执行姿态微调、规范姿态反解、动画或衣物非刚性拟合。

验证命令（测试不写入输入目录）：

```bash
python tools/align_soma_gaussians.py --help
python -B -m unittest discover -s tests -p test_soma_gaussian_binding.py -v
```

## Waymo 人物重建场景筛选与预览

`preview_waymo_people.py` 只读扫描 `metadata.csv` 指定编号范围内的 TFRecord，检查五路相机的原生行人框，选择每个 clip 最好的同一人物轨迹并生成 5 张场景与人物裁剪拼接预览。依赖 Python 3.10+、Pillow 10.1+（当前系统 Python 已有）；不依赖 TensorFlow、Waymo SDK 或 GPU，不下载模型。只支持本地未压缩 TFRecord，按 protobuf 字段范围读取并跳过激光雷达载荷；检查记录边界，但不校验 TFRecord CRC。

```bash
python tools/preview_waymo_people.py \
    --data-root data/waymo-data \
    --start-id 0 --end-id 999 --workers 64 \
    --output-dir outputs/waymo_person_previews_00000_00999
```

默认要求人物高至少 180px、宽至少 45px、高宽比 1.5–5、距画面边缘至少 12px，排除检测难度 LEVEL_2，以及被其他标注框覆盖超过 20% 的目标；同一相机同一人物的五张预览至少相隔 0.3 秒。可通过 `--min-height`、`--min-width`、`--max-overlap`、`--spacing-seconds` 调整。像素尺寸与框重叠仅用于候选排序，不能验证遮挡、人体完整性、运动模糊或重建质量。

输出目录包含每个入选 clip 的 5 张 JPG 与 `metadata.json`，以及浏览页 `index.html`、排序清单 `ranking.csv`、完整扫描清单 `scan.jsonl`、参数和错误统计 `summary.json`。帧号从 0 开始，元数据保留源 TFRecord、相机、人物 ID、时间戳和原图框坐标。扫描不完整会终止，不会伪报已检查全部 clip；完整扫描后的绘图中断可使用原参数加 `--render-only` 恢复。新扫描使用新目录，避免覆盖已有扫描结果。

验证：`python tools/preview_waymo_people.py --help`；实际处理后检查 `summary.json` 的错误列表、扫描范围及每个入选子目录恰好 5 张图片，并抽查人物裁剪。

## 人像退化对齐重建模型

`process_masked_person.py` 将原始人像的细碎纹理、局部明暗和颜色观感作受控调整，便于与重建人物外观对照。使用 `adam-gem-x` 环境现有 NumPy、SciPy、OpenCV；可选视频使用该环境的 FFmpeg。安装见 [`setup/README.md`](../setup/README.md)，无需新增依赖、模型下载或GPU。

流程始终为 **原始图像＋人物mask → 抠出前景 → 去噪 → flat → 提亮 → 饱和度 → 去灰 → 对比度 → 通道裁剪 → 同mask合成一次**。透明区域用最近前景颜色延拓，仅作滤波边界条件，不引入原背景或透明黑色。人物内部不做处理强度羽化，也不混回旧颜色。输入已有软alpha时仅最终合成一次，不将alpha预乘进处理颜色；它仍不能恢复源图边缘已混入的背景颜色。工具不生成mask，不做Gaussian取色或参数拟合。

单图（请替换实际输入路径，输出目录必须不存在）：

```bash
conda activate adam-gem-x
OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 \
python tools/process_masked_person.py \
    --image /path/to/original-person.png --mask /path/to/person-mask.png \
    --output-dir outputs/person-appearance --save-stages
```

连续帧按文件名排序，支持PNG/JPG/JPEG，mask必须为同名stem的灰度PNG，如 `frames/0001.jpg` 对应 `masks/0001.png`。图像要求8-bit三通道，mask要求同尺寸8-bit灰度且非空，不能直接把彩色类别标注当作mask。

```bash
python tools/process_masked_person.py \
    --images-dir /path/to/raw-frames --masks-dir /path/to/person-masks \
    --output-dir outputs/person-sequence --save-stages --video --fps 10
```

也可使用清单，路径相对于显式 `--source-root`（默认当前工作目录）：

```json
{
  "per_frame": [
    {"source": "frames/frame0.jpg", "mask": "masks/frame0.png", "crop_box": [100, 50, 300, 450]}
  ]
}
```

```bash
python tools/process_masked_person.py \
    --manifest /path/to/frames.json --source-root /path/to/source-root \
    --output-dir outputs/person-manifest --save-stages
```

`crop_box` 是可选整数xyxy，裁剪原图后与mask尺寸对应；不提供时处理整张图。清单未写 `mask` 时按 `--masks-dir/0000.png` 顺序查找，未指定该参数则查找清单旁 `input_masks/0000.png`，兼容已有人物帧清单的 `per_frame/source/crop_box` 字段。对大幅原图可先定义人物crop，以减少滤波工作量；flat空间尺度相对于处理图像高度。

输出：`foreground/` 最终RGBA、`composite/` 单次贴回后的RGB、`metadata.json` 参数及逐帧来源；`--save-stages` 另保存 `stages/00_cutout` 至 `07_channel_clip` 的RGBA。清单有crop时，composite是裁剪图，不是完整源场景。工具不内置Gaussian对照渲染；需要九宫格时，将Origin与七个累加阶段按行排列，Gaussian只放右下角作为显示参考。

`--video` 额外编码 `composite.mp4`，遵循 [`configs/README.md`](../configs/README.md)，默认10 FPS、H.264 High/4.0、yuv420p、无音轨，奇数宽高仅在右侧/底部补齐。要求全部帧尺寸一致，不会静默缩放变化的crop；此类输入先保留PNG或另作统一展示。FFmpeg按PATH、当前Python环境查找，可用 `--ffmpeg /path/to/ffmpeg` 指定。高分辨率任务需要的编码Level调整不属于当前CLI。

参数通过 `--config appearance.json` 覆盖，下表也是默认值：

| 配置键 | 默认值 | 含义 |
| --- | --- | --- |
| `denoise_h` | 14 | 彩色NLM，0关闭 |
| `template_window`, `search_window` | 7, 21 | 正奇数，搜索窗不小于模板窗 |
| `flat_sigma_color`, `flat_sigma_space_fraction`, `flat_mix` | 55, 0.025, 0.85 | 双边平滑，空间sigma至少3像素，mix=0关闭 |
| `exposure_ev` | 0.15 | 线性RGB中间调曝光，亮部保护 |
| `saturation` | 1.5 | HSV饱和度倍率 |
| `degray` | 0.015 | 线性RGB中性灰底校正，深阴影保护 |
| `contrast`, `contrast_pivot` | 1.15, 0.18 | 亮度S曲线斜率和支点 |
| `contrast_gain_min`, `contrast_gain_max` | 0.75, 1.2 | RGB亮度增益限制 |
| `black_point`, `white_point` | 24, 232 | 编码RGB通道黑白点，允许裁剪 |

例如更弱的处理：`{"denoise_h": 8, "flat_mix": 0.5, "saturation": 1.25, "black_point": 12, "white_point": 244}`。所有帧使用同一参数配置，不按Gaussian颜色或逐帧直方图自动变化；不含光流时序滤波。这里的flat是保边平滑，不是真实材质/光照分解。强处理可能抹掉脸部、反光条、褶皱及高光；需要保护反光条时先检查其细节在去噪/flat阶段是否已经丢失，而非只提高后续对比度。参数固定不等于已验证跨场景鲁棒性。

验证（全部样例只写临时目录）：

```bash
python tools/process_masked_person.py --help
python -B -m unittest discover -s tests -p test_masked_person.py -v
```

测试覆盖背景变化不影响前景滤波、软alpha只合成一次、无效参数/mask拒绝、清单裁剪、RGBA alpha一致、输出不覆盖，以及30帧/10 FPS视频编码。原实验数据若不在当前机器，不将临时样例声称为原视频像素回归。
