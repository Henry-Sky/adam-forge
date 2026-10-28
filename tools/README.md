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
