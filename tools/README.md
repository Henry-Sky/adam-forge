# 独立工具

## SAM3 按人物 ID 批量生成 mask

`build_person_masks.py` 使用现有 `adam-sam3` 环境与本地 `checkpoint/sam3/sam3.pt`，无新增依赖或下载。当前为 `sam3-jpeg-sequence-temporal-v2`：按帧序读取每对 `scene_rgb/*.jpg`，按已有 placement 放入固定尺寸人物 crop 画布，将全部 PIL 图像作为视频序列传入 `build_sam3_video_model`。不读取 MP4，不重新编码 JPEG，保留源 JPEG 解码后的像素；模型内部执行其标准缩放。**不是逐帧独立分割**，非锚点帧通过 SAM3 视频 tracker 的时序记忆传播。

程序先核对 manifest 与 placement 的人物 ID、帧号、逐帧框和图像路径。首、中、末帧的对应 ID 框转换为视频画布归一化的框角点（SAM 标签 2、3），为同一个 `obj_id` 添加三个条件锚点；随后从首帧向末帧传播全部帧。29 帧序列的锚点是 0、14、28。现有画布跟随人物 crop 移动，不等同于固定场景相机运动；身份由已有标注约束，遮挡与标注偏差仍可能影响质量。

```bash
# CPU 预检，不加载模型、不写入数据。
.conda/adam-sam3/bin/python -B tools/build_person_masks.py \
    --preflight-only --progress-dir outputs/sam3-mask-preflight

# 在沙箱外执行 GPU；用户已授权以新结果覆盖旧 mask。
bash tools/start_person_mask_task.sh --overwrite
```

默认 GPU 0–3、共 64 个计算线程，每卡一个持久模型进程、16 个 PyTorch/BLAS 线程；OpenCV 和 interop 各限制为 1。控制进程和 CUDA 辅助线程不计入 64 个计算线程。可通过 `--gpus`、`--threads`、`--train-root`、`--checkpoint` 覆盖默认参数；`--limit 1 --gpus 0 --threads 16` 用于单对验证。

启动器使用 `setsid -f nohup`，标准输入来自 `/dev/null`，stdout/stderr 写入任务日志；进程有独立会话且忽略 SIGHUP，退出终端或 Codex 不终止任务。主机重启或容器停止后需重新启动。

输出覆盖各对已有 `sam3_mask/`，不改动源 JPEG、manifest、既有 crop 或视频：

- `scene/000.png` 等：与原场景同尺寸，前景 255、背景 0。
- `crop/000.png` 等：与人物视频画布同尺寸、同偏移与补黑，补黑处强制为 0。
- `metadata.json`：算法版本、人物 ID、时序传播方式、锚点帧、提示框、来源和坐标。逐帧记录 `tracker_score_raw`，不将 tracker 分数称为预测 IoU。空 mask、目标 ID 缺失或超过一半前景位于对应 ID 框外会标记人工复查，不丢帧。

整对写入唯一临时目录，帧数及尺寸检查通过后发布。`--overwrite` 允许用完整新结果替换旧目录；支持时使用 Linux 原子目录交换，不支持时先将旧目录临时改名，再发布完整新目录，发布异常会恢复旧目录。此回退方式有很短的路径切换间隔，硬中断可能留下 `.sam3_mask-replaced-*`。替换成功后删除旧 mask，不保留两套结果。程序遇到单对异常会记录并继续，失败临时目录保留供诊断；源数据不删除。

签名包含算法版本、manifest/placement 内容、源 JPEG 大小和修改时间以及权重路径/大小/修改时间。重跑会跳过同签名且输出齐全的新版结果；旧版逐帧结果签名不同，会在 `--overwrite` 下重做。数据根目录 `.sam3-mask.lock` 防止两批并发写入。

本次视频后台任务位于 `outputs/sam3-video-mask-task-20261102T074606Z/`；旧逐帧任务已停止，旧日志仅保留历史。监控：

本次任务于 2026-11-02 08:26:38 UTC 完成：824 对、23,896 帧，0 失败，四个进程退出码均为 0，耗时约 40 分钟；其中首对已在视频模式验证中完成，批处理跳过该对。1,349 帧带复查标记，需人工判断，不等同于分割错误。

```bash
cat outputs/sam3-video-mask-task-20261102T074606Z/progress.json
tail -n 30 outputs/sam3-video-mask-task-20261102T074606Z/task.log
cat outputs/sam3-video-mask-task-20261102T074606Z/run.json
```

`run.json` 记录 PID、会话和最终状态，`gpu-*.json` 记录各卡当前训练对、成功/跳过/失败/复查计数，`gpu-*.jsonl` 记录逐对耗时和异常。停止时向 `task.pid` 中的主进程发送 SIGTERM，主进程会终止并回收四个子进程。恢复时重跑上述启动命令。

验证：

```bash
bash -n tools/start_person_mask_task.sh
OPENBLAS_NUM_THREADS=1 .conda/adam-sam3/bin/python -B -m unittest discover -s tests -p test_person_masks.py -v
# 在沙箱外进行实际视频推理和覆盖验证：
.conda/adam-sam3/bin/python -B tools/build_person_masks.py \
    --limit 1 --gpus 0 --threads 16 --overwrite --progress-dir outputs/sam3-video-mask-smoke-verified
```

测试仅写临时目录，覆盖原 JPEG crop 的像素与框映射、ID 错配拒绝、覆盖保护、完整替换及文件系统不支持交换时的回退与失败恢复。实际首对 29 帧已验证二值像素、原生尺寸、身份和 crop/scene 的逐像素对应，5 帧叠加预览见 `outputs/sam3-video-mask-smoke-verified/preview.jpg`。

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

## 人物投影 crop 与场景视频（可脱离 Codex 运行）

`build_person_videos.py` 在已有 `data/adam-data/train-data/<clip_id>_<obj_id>/` 内生成：

- `scene_rgb.mp4`：按 `scene_rgb/000.jpg` 等原始顺序编码的完整场景。
- `person_crop/person.mp4`：对应人物的固定画布视频，保留原生像素尺寸，逐帧居中补黑边，不缩放人物。
- `person_crop/placement.json`：逐帧源图路径、源视频帧号、纳秒时间戳、原图裁剪框 `paste_xyxy`、视频有效区域 `crop_in_video_xyxy`、正反向二维仿射矩阵、3D 框、3D 投影框及同 ID 的 2D 标注框。
- `person_crop/source_manifest.json`：更新前的训练对清单备份；原 `manifest.json` 的 `files` 增补视频和位置记录路径。

依赖使用现有 `adam-gem-x` 环境中的 NumPy、Pillow、FFmpeg；贴回工具另外使用该环境已有的 OpenCV。视频编码遵循 [`configs/README.md`](../configs/README.md)，场景 CRF 18、crop CRF 12。H.264 是有损编码；位置映射保持整数像素几何，不保证编码后的 RGB 数值与原图完全相同。

投影按对象 ID 和原始帧号匹配可信本地 PKL，使用相机外参、内参及有理径向/切向畸变，并采样 3D 框边缘、裁切相机近平面。通过同 ID、同相机的直接 2D 标注核对原始标注与场景图的分辨率比例，不能确认时记录失败而非猜测。由于部分 3D/2D 标注存在偏移，最终 crop 采用两者包围框并集，四边各留 5% 余量后裁至图像边界。框为左闭右开 `[x0,y0,x1,y1]`，数组索引为 `[y0:y1,x0:x1]`。

**此工具会向数据软链接指向的现有训练对写入派生结果，运行前应已获得修改该训练数据目录的授权。** 原始 JPG、源视频和 PKL 保留。每个视频编码后实际解码计数并检查尺寸、帧率及格式，再发布文件；已有完整输出经输入签名和视频检查后跳过，不同输入的已存在 crop 会报错。

在仓库根目录启动独立后台进程：

```bash
bash tools/start_person_video_task.sh
cat outputs/train-crop-task/task.pid
cat outputs/train-crop-task/progress.json
tail -f outputs/train-crop-task/task.log
```

启动器使用 `setsid -f nohup`，标准输入断开，日志落盘，运行不依赖 Codex 进程或交互终端。退出 Codex、关闭终端后继续；机器/容器关闭时不保证运行。默认 16 个处理进程，每个 FFmpeg 编码器 3 线程，加上单线程图像处理约使用 64 CPU 线程；使用软件 H.264 编码，不进行 GPU 推理。可通过 `ADAM_VIDEO_PYTHON` 指定其他已安装依赖的 Python。

`progress.json` 包含 `running` / `completed` / `completed_with_errors` / `failed`、总数、完成/复用/失败数、帧数、PID 和运行 ID；`results.jsonl` 按运行 ID 记录每个训练对的结果及错误。进度按源 clip 完成批次刷新。任务使用文件锁避免同一进度目录被多个批处理同时占用。中断后再次运行启动命令即可检查并复用完成结果、重做未完成项；机器重启后需重新运行启动命令。

直接前台运行或仅检查少量训练对：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.conda/adam-gem-x/bin/python tools/build_person_videos.py \
  --train-root data/adam-data/train-data --source-root data/adam-data/source-data \
  --workers 16 --encoder-threads 3 --progress-dir outputs/train-crop-task

# 重复 --pair-id 可选择多个样本；--padding 修改框边余量
.conda/adam-gem-x/bin/python tools/build_person_videos.py \
  --pair-id 00000_00005 --workers 1 --progress-dir outputs/train-crop-smoke
```

修改后的视频须保持画布尺寸、帧数、帧序及 10 FPS；贴回时忽略补黑区域，优先使用原 JPG 作背景，缺失时读取 `scene_rgb.mp4`。以下操作按矩形区域贴回，不做人物分割或边缘融合：

```bash
.conda/adam-gem-x/bin/python tools/paste_person_video.py \
  --pair-dir data/adam-data/train-data/00000_00005 \
  --edited-video /path/to/edited-person.mp4 \
  --output outputs/person-edited-scene.mp4

OPENBLAS_NUM_THREADS=1 .conda/adam-gem-x/bin/python -B -m unittest discover -s tests -p test_person_video.py -v
bash -n tools/start_person_video_task.sh
```

## SAM3 前的人物 crop 异常候选审查

`audit_person_crops.py` 使用 `adam-gem-x` 环境，对现有训练对执行只读质检，输出带 clip ID、obj ID 的独立审查目录。工具不调用 SAM3，不修改或删除数据。报告目录必须不存在，避免覆盖人工审查结果。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.conda/adam-gem-x/bin/python tools/audit_person_crops.py \
  --train-root data/adam-data/train-data --source-root data/adam-data/source-data \
  --output tests/crop_review_20261102 --workers 64 --controls 24

# 少量样本检查；输出改用新的空目录
.conda/adam-gem-x/bin/python tools/audit_person_crops.py \
  --pair-id 00000_00005 --pair-id 00017_00066 \
  --output outputs/crop-audit/smoke-new --workers 2 --controls 1

OPENBLAS_NUM_THREADS=1 .conda/adam-gem-x/bin/python -B -m unittest discover -s tests -p test_crop_audit.py -v
```

检查范围包括两个视频的实际解码帧数、尺寸、编码及 10 FPS，人物 ID、帧序与源 PKL 时间戳、正反向位置映射、3D/2D 框偏移、原生人物像素量、背景/补黑占比、跨帧位置和尺度突变、同相机同帧其他人物、低细节及首/中/末帧与源 JPG 区域的一致性。全部使用 CPU，OpenCV、FFprobe 和 BLAS 均限定单线程，默认并行 64 个源 clip。

建议分为三类，均不自动应用到数据集：

- `exclude`：建议剔除。完整性检查异常，或至少一半帧的可见目标高度小于 24 像素/宽度小于 8 像素。
- `review`：需人工复核。包括目标偏小（至少 25% 帧高度小于 48 像素/宽度小于 16 像素）、投影偏移、定位跳变、背景过多、多个竞争人物等；这些情况未必不能使用 SAM3。
- `keep`：暂未命中本次规则，不等同于人工确认合格。默认抽取 24 个确定性对照样本供复查。

小目标阈值是本次数据筛选建议，不是 SAM3 官方输入限制。`--thresholds /path/to/thresholds.json` 可覆盖 `utils/crop_audit.py` 中的已知阈值键，最终参数完整写入报告。原始遮挡/截断字段全为 0，真实遮挡和未标注人物仍需人工检查；颜色框仅为标注引导，不保证语义上始终正确。

输出：

- `index.html`：离线分页审查页，可按 clip/obj ID 或原因搜索，选择保留/剔除并导出人工决定 JSON；当前浏览器的选择保存在本地存储，提交审查时应导出 JSON。
- `exclude/<clip_id>_<obj_id>/`、`review/<clip_id>_<obj_id>/`：六个代表/风险帧的 `preview.jpg`、带 ID 的原 crop MP4 副本、指标与原因 `review.json`。预览分别展示干净 crop、叠加焦点框的 crop 和原场景上下文。
- `control/`：未命中规则的对照预览。
- `exclude_candidates.csv`、`review_candidates.csv`、`remaining_candidates.csv`、`all_candidates.csv`：分组和全量清单。
- `all_results.json`：完整指标、异常原因和逐帧 `focus_boxes_in_crop_xyxy`，可为后续 SAM3 提供按 ID 对齐的框提示，但应先通过本次人工审查。
- `summary.json`、`progress.json`、`README.md`：计数、可重叠的原因统计、阈值、处理进度和审查说明。

生成的 `tests/crop_review_*/` 按现有规则忽略，不纳入 Git；只有相应单元测试源文件显式跟踪。报告和媒体全部在独立目录内，完整目录可拷贝后离线打开 `index.html`。

## 人物配对退化：ISP 与二维光照

入口 `tools/degrade_person.py`，参数算法在 `utils/person_degradation.py`。使用 `adam-gem-x`，安装和依赖见 `setup/README.md`。默认参数统一维护在 [`configs/person_degradation.yaml`](../configs/person_degradation.yaml)，使用 PyYAML safe_load 读取并检查未知字段、范围和模式权重。CPU 默认64线程；脚本无需GPU或Gaussian。必须提供原始人物帧及已有Mask，脚本不执行分割。

```bash
conda activate adam-gem-x
python tools/degrade_person.py \
    --images-dir /path/to/original-frames --masks-dir /path/to/masks \
    --sample-id clip00001_obj00002 --config configs/person_degradation.yaml \
    --output-dir outputs/person-degradation --save-stages --video
```

单图使用 `--image /path/to/image.png --mask /path/to/mask.png`；JSON清单使用 `--manifest /path/to/frames.json --source-root /path/to/root`。清单格式为 `{"pair_id":"00001_00002","per_frame":[{"source":"frame.png","mask":"mask.png","frame_id":0,"crop_box":[0,0,100,200]}]}`；`crop_box` 可省略，Mask尺寸须对应裁剪后的图像。保留每条记录的时间戳、贴回矩阵、位置等额外字段。清单中缺省Mask时，采用 `--masks-dir/0000.png`，未设置该参数则采用清单旁 `input_masks/0000.png`。单次调用处理一个人物片段；目录模式按文件名排序，必须用连续且补零的帧名。`--sample-id` 应显式指定稳定的clip/person标识；缺省采用清单pair_id、图像stem或输入目录名，重复名称会使用相同随机流。

YAML `variants` 控制每片段输出变体数，`modes` 权重决定细节、ISP、二维光照的组合；`--mode isp` / `relighting` / `isp_relighting` / `detail` / `detail_isp_relighting` / `legacy` 可固定分支。每个变体从seed、sample-id、变体编号派生随机流，模式和各分支分别采样；同一人物同一变体不同模式共享对应分支参数，便于公平比较。detail仅运行NLM去噪和flat，默认h=6、flat_mix=0.3；独立ISP/光照不默认运行detail。完整历史处理由 `process_masked_person.py` 实现，新工具通过独立legacy模式复用，默认权重0，禁止与其它分支组合。

处理顺序：原图Mask前景提取 → 可选二维光照 → 可选ISP → 可选温和detail → 原Mask单次合成。detail不再执行曝光、饱和度、去灰、对比度或通道裁剪；在光照与调色后重新进行前景边界延拓再滤波。光照场绑定人物Mask外接框的归一化坐标，按YAML进行因果平滑；参数在片段内固定，不包含逐帧随机或动态光照轨迹。ISP支持线性曝光、亮度归一化的白平衡增益、线性亮度饱和度、支点幂函数对比度、gamma幂曲线和可选编码RGB通道裁剪；与旧脚本的亮部保护曝光和S曲线不同。

输出目录必须不存在。每个 `variant_000/` 保存：

- `input/`：退化合成PNG；`target/`：未处理的原图或同坐标crop PNG；`mask/`：原Mask PNG；`foreground/`：处理后RGBA。
- `pairs.json`：模式、实际采样参数、种子、逐帧原始文件哈希、输入输出路径、源帧信息、裁剪坐标、光照使用的人物框。目录上层 `config.yaml` 为配置快照，`metadata.json` 记录版本、身份和完成状态。
- `--save-stages` 保存实际运行的中间RGBA；`--video` 同时生成input.mp4和target.mp4，遵循 `configs/README.md`，`--fps`默认10。只接受等尺寸序列，不静默缩放；PNG是无损训练对，MP4只作预览。`--ffmpeg`可指定编码器。

GT始终来自未处理原图。透明区使用最近前景颜色延拓，二值Mask外合成PNG与GT一致，软alpha最终只合成一次。二维光照仅叠加空间亮度/光色，不移除原光照、不恢复材质、不生成地面阴影。不同模式可能出现近似无变化的随机结果，当前不自动拒绝或重采样；参数范围尚需真实样本视觉调优。固定参数不保证SAM Mask或逐帧滤波时序稳定。划分训练/验证集时先按源clip分组，再生成退化变体。

验证：

```bash
python tools/degrade_person.py --help
python -B -m unittest discover -s tests -p test_person_degradation.py -v
```

测试只写临时目录，覆盖全部模式真实CLI输出、原图GT与Mask保留、背景不变、软alpha单次合成、位置记录、输出不覆盖、可复现采样及非法配置拒绝。

配置版本2明确区分detail与legacy；版本1配置会拒绝加载，避免把旧调色链静默解释为温和detail。`metadata.json`输出格式为`person_degradation_pairs_v2`，`pairs.json.processing_order`记录实际运行的分支。历史JSON入口保持原行为。
