# 数据目录

## 人物任务共享日志与 Pod 交接

`data/adam-logs` 指向现有共享目录 `/mnt/data/adam-forge/logs`，用于人物任务的运行说明、环境和配置快照及测试证据；已核实目标存在。最新交接入口为 `data/adam-logs/README.md`，2026-11-05 的完整交接包为 `pod-handoff-20261105T134705Z/`，包含当前源码（含未提交工具）与 DiffSynth 快照、安装与三卡测试命令、79 组清单、三组数据哈希和关键日志。共享日志、数据、权重均不进入 Git；新 Pod 需挂载相应共享存储，不能只迁移软链接。

噪点输入版的三组原分辨率 VACE 重测在 7/30、7/30、9/30 步时中断，没有保存中间 latent，也没有替换原 `scene_background.mp4`。后台资源等待进程随后按用户要求停止，状态为 `cancelled_by_user`；目前无自动补跑安排。迁移后应从第 0 步重测这三组并审查质量，尚未扩跑其余 76 组。

`data/adam-data/train-data/` 为人物退化与人物重建混合数据集：无前缀的 `<clip_id>_<obj_id>` 目录用于人物退化，`adam-<clip_id>-<obj_id>` 目录用于人物重建。两类目录均保留五位 clip ID 和 obj ID；按任务选择对应目录。人物重建副本与原人物退化样本共享来源，不应作为独立样本重复计数。

79 个人物重建目录首次新增 `scene_mask.mp4`：根据选中人物的逐帧 SAM3 场景 Mask 取最小外接矩形，框内白色（255）、其余黑色（0），与同目录 `scene_rgb.mp4` 尺寸、帧数和帧率一致。采用 H.264 无损编码以保持严格二值；全部 79 个视频、2,291 帧已通过解码逐像素校验，0 失败。工具、编码例外与运行命令见 [`tools/README.md`](../tools/README.md#人物重建场景矩形遮罩视频)，逐帧框和校验报告见 `outputs/person_pixel_distribution_20261105/scene_mask_report.json`。

这 79 个目录另新增 `scene_liner.mp4`：以解码后的 `scene_rgb.mp4` 为源，在选中人物的 SAM3 轮廓内部，用轮廓外一像素背景环的 RGB 分片线性插值覆盖；凸包外采用最近边缘颜色。Mask 外像素在编码前保持原样，输出遵循默认有损视频编码，尺寸、帧数和帧率与源一致。全部 79 个视频、2,291 帧完成，0 失败；原视频及 Mask 保留。算法、运行与测试命令见 [`tools/README.md`](../tools/README.md#人物重建场景线性环境覆盖视频)，报告见 `outputs/person_pixel_distribution_20261105/scene_liner_report.json`。

2026-11-05 按用户授权，仅 `adam-00000-00057`、`adam-00000-00067`、`adam-00000-00184` 新增 `scene_background.mp4` 试验视频；以 `scene_liner` / `scene_mask` 做 VACE 条件，原分辨率、29 帧、30 步，生成后仅 Mask 内贴回原场景。三个视频均通过尺寸、帧数与格式检查，但首/中/末帧视觉审查未通过整体擦除质量要求：`00057` 为部分去除、仍有残留，`00067` 和 `00184` 明显保留人形。它们是待复查的试验结果，不能按合格背景数据使用；其余 76 组未扩跑。全部日志、参数、预览与质量记录位于 `outputs/vace-erase-clip00000-20261105/`（`quality_review.json`）。

同日按用户授权，仅上述三个 clip 0 目录新增 `scene_input.mp4`：以解码的 `scene_rgb.mp4` 为底图，将所选人物 SAM3 Mask 的逐帧最小外接矩形全部替换为等概率黑白噪点（每像素 RGB 同为 0 或 255），矩形外保持原样。固定 seed 0，并按 clip/obj/帧号生成独立噪点；保留原分辨率、29 帧和 10 FPS。初版采用无损 H.264 RGB 编码，全部 87 帧解码后与预期 RGB 完全一致（后续已按色彩兼容反馈替换，见下文）；输入视频生成后未重跑 VACE。命令及默认编码例外见 [`tools/README.md`](../tools/README.md#人物矩形噪点输入与稳定遮罩)，参数、逐帧框、哈希与预览见 `outputs/scene-noise-input-clip00000-20261105/`。

随后按用户反馈，仅这三个目录同步更新 `scene_input.mp4` 和 `scene_mask.mp4`：输入改为直接在源 YUV420 数据中替换噪点，H.264 High/yuv420p、CRF 10，保留源颜色元数据；不再采用 RGB/gbrp。遮罩基于原 SAM3 框，每边扩展 10%、至少 8 px，采用 7 帧双向三角平滑并确保覆盖当前帧扩展框；最终偶数坐标对齐色度块。噪点纹理固定在场景坐标，减少逐帧抽样闪烁。输入有损编码后允许少量灰阶，Mask 仍为严格 0/255。三个样本相较旧 SAM3 框的中心二阶差分抖动分别降低 44.0%、19.4%、57.9%；距 Mask 边缘 16 px 之外的背景单通道平均偏差最大 0.0841/255，RGB MAE 最大约 1.0711/255。参数、候选、预览及当前上一版备份见 `outputs/scene-noise-input-clip00000-yuv-stable-v3-20261105/`；初始 RGB 版备份在 `outputs/scene-noise-input-clip00000-yuv-stable-v2-20261105/<pair_id>/previous/`。原始 RGB、SAM3 PNG 和 placement 保持原样，旧 `scene_background.mp4` 未重跑；其余 76 组 Mask 未更改。

此目录用于通过软链接访问外部数据，不存放或提交数据集本体。当前本地软链接：

| 数据 | 仓库内路径 | 外部目标目录 |
| --- | --- | --- |
| 人物对齐数据 | `data/adam-data` | `/mnt/data/adam-forge/adam-data` |
| Waymo | `data/waymo-data` | `/mnt/data/waymo_v1_4_3/individual_files` |
| 3D 人物资源 | `data/3d-person-assets` | `/mnt/data/3d-person-assets` |

人物对齐数据通过 `data/adam-data` 软链接访问。`source-data/` 保存筛选出的 455 个 clip 源视频、PKL、caption 和元数据；`train-data/` 当前保留572个对象训练对（原1,653个，按用户确认先后删除140个原1类、660个面积小于2000 px²、8个视觉0类及21个人工排除项，余824个；2026-11-03按SAM3 Mask轮廓缺陷审计v2口径删除180对（日志 `deletion_log_20261103T025338Z.md`，删除前快照 `pre_deletion_state_20261103T025338Z.json`）；同日按持续欠分割覆盖重扫删除28对（日志 `deletion_log_20261103T030334Z_coverage28.md`，名单 `coverage_candidates.csv`，删除前快照 `pre_coverage_deletion_state_20261103T030334Z.json`）；两次删除依据与指标均见 `outputs/sam3_mask_defect_audit；同日全景退化批次前另行删除 `00045_00090`（616→615），此后又删除该批次中因空Mask帧无法构造29帧完整场景训练对的43对（日志 `deletion_log_20261103T083457Z_emptymask43.md`，快照 `pre_emptymask_deletion_state_20261103T083457Z.json`）），每个目录以 `metadata.csv` 的五位数字 clip ID 和五位 obj ID 命名（`<clip_id>_<obj_id>`），含 29 帧完整场景 RGB 图像及逐帧人物框。人物 crop 与场景视频的批处理入口及贴回方法见 [`tools/README.md`](../tools/README.md#人物投影-crop-与场景视频可脱离-codex-运行)。每个训练对新增 `scene_rgb.mp4`、`person_crop/person.mp4` 和逐帧位置记录 `person_crop/placement.json`，并在 `manifest.json` 记录路径；原始 `scene_rgb/*.jpg` 保留。crop 基于对应 ID 的 3D 框投影，合并同 ID 的 2D 框以防标注偏移导致截断，使用原尺寸像素和固定补黑画布；位置记录支持修改后贴回。退化图尚待补充。链接目标由外部存储提供，不纳入版本控制。

2026-11-05 按用户授权，在 `train-data/` 内为 29 帧 SAM3 人物 Mask 面积中位数 ≥8,000 像素的 79 组新增完整副本，命名为 `adam-<五位clip_id>-<五位obj_id>`。原 572 组保留，当前共 651 个训练对目录（其中 79 个为副本，并非新增独立样本）。复制时内部文件及原 ID 保持原样，仅目录改名；使用 64 线程复制，7,584 个文件全部通过源/副本 SHA-256 校验。随后按用户授权，仅从这 79 个人物重建副本中删除 `detail_strength_comparison.mp4`、`scene_person_degraded.mp4` 和 `scene_person_degradation.json`，共 237 个文件；原人物退化目录中的对应文件保留，删除记录见同一统计目录的 `reconstruction_cleanup.json`。选择明细、逐帧统计和复制报告见本机 `outputs/person_pixel_distribution_20261105/` 下的 `per_pair_median.csv`、`per_frame.csv`、`copy_report.json`。

软链接及数据由 `.gitignore` 排除，此目录仅跟踪本 README。以上链接依赖本机外部存储；迁移环境时需按实际存储位置配置链接。Waymo 数据链接依赖本机外部存储。

将数据软链接及其目标数据作为只读来源：未经明确授权，不修改、删除、移动、替换或重定向链接及源数据。筛选清单、截图和重建结果写入 `outputs/` 或临时目录，不写回数据目录。

本次824条保留名单及删除记录存放在本机忽略目录 `outputs/crop_review_20261102_final/`；原审查预览目录 `tests/crop_review_20261102/` 已按用户要求删除。

用户已授权对上述训练对运行 SAM3 并新增对应人物 ID 的掩膜。`tools/build_person_masks.py` 在每对的 `sam3_mask/scene/` 保存与原始 JPEG 同尺寸的逐帧二值 PNG，在 `sam3_mask/crop/` 保存与 `person_crop/person.mp4` 画布对齐的逐帧二值 PNG；文件名为 `000.png` 至 `028.png`，像素值为 0/255。`sam3_mask/metadata.json` 记录人物 ID、逐帧框提示、来源、tracker 原始评分、复查标记和坐标。当前使用 `sam3-jpeg-sequence-temporal-v2`：将原 JPEG 的原生 crop 解码为固定画布帧序列，输入 SAM3 视频 tracker；首、中、末帧以同一人物 ID 的框作条件锚点，其余帧利用时序记忆传播，不解码有损视频或重新编码 JPEG；不修改源图、原 manifest 或既有 crop。批次总计 824 对、23,896 帧（历史记录；此后该批次中180对已按上述mask缺陷审计删除）。运行、断点续跑和后台状态命令见 [`tools/README.md`](../tools/README.md#sam3-按人物-id-批量生成-mask)。

用户已授权使用上述视频时序结果直接覆盖已有逐帧 `sam3_mask/`。旧逐帧后台任务已停止；新版于 2026-11-02 08:26:38 UTC 完成全部 824 对、23,896 帧，0 失败；2026-11-03 据此审计删除180对后 train-data 余644对，四卡进程均正常退出。1,349 帧带人工复查标记，不表示已确认错误。任务记录见 `outputs/sam3-video-mask-task-20261102T074606Z/progress.json`。

2026-11-03 用户授权为当前全部保留人物新增三个与 `scene_rgb.mp4` 同级的产物：`scene_person_degraded.mp4`、`detail_strength_comparison.mp4`、`scene_person_degradation.json`。使用 `configs/person_degradation_detail_medium.yaml`，按二维光照 → ISP → 增强detail（h=10、flat_mix=0.55）退化人物后贴回全景；对比视频为原图 / 温和detail / 增强detail。JSON记录实际参数、配置、逐帧坐标和来源/输出哈希。原JPEG、RGB视频、Mask和placement不改写；具体后台运行与恢复方法见 `tools/README.md` 的“批量发布完整场景人物退化”。本次实际扫描615对，以任务 `inventory.json` 为准，运行完成情况以 `progress.json` 为准。 任务结果572对完成、43对因空Mask帧失败；该43对已于2026-11-03删除，现train-data的572对全部具备完整退化三产物。
