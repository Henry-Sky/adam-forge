# SOMA / Gaussian 临时配准与导出

在浏览器导入 SOMA 和 PLY，暂存到服务器临时目录。分别调整两个对象后，
导出包含配准 SOMA 和 PLY 的目录；导入源和临时副本都不会被修改。
不使用持久项目库，不覆盖导入文件。

## 启动

使用 [`setup/README.md`](../setup/README.md) 的 `<prefix>-web` 环境（默认 `adam-web`）：

```bash
bash setup/install_web.sh --prefix adam
conda activate adam-web
python web_server/api.py --port 8766
```

打开 **http://127.0.0.1:8766/adam-forge-web-server/**。默认只监听本机，远程访问使用 SSH 转发：

```bash
ssh -L 8766:127.0.0.1:8766 <服务器>
```

服务无账户认证，仅供可信环境使用。依赖 Three.js 0.180.0、Spark 2.4.0 固定版本，
安装时国内镜像优先并校验 SHA-512；浏览器运行时不访问外部 CDN，无需 npm 构建。

## 导入

左侧“导入”从服务器路径导入 SOMA 与 PLY，路径框右侧“浏览”可列出服务器目录、返回上级，并筛选选择对应 `.npz` / `.ply` 文件；“上传”从浏览器本地选择这两个文件。二者都复制到临时目录。
默认使用系统临时目录，服务正常退出或 SIGTERM 停止时自动清理；强制 SIGKILL 或系统崩溃
可能留下临时目录，需要手动清理。`--temp-dir` 可指定自管临时目录，此时不自动删除。
只有三维工作台页面，默认显示空白场景。左侧“当前资产”的“导入”按钮填写服务器 SOMA/PLY 路径；“上传”按钮选择本地 SOMA/PLY。临时任务通过 URL 中的 `project` 标识，服务重启后失效时页面回到空白状态，请重新导入。

- SOMA：NPZ，需含 `vertices_camera`、`faces`、`joints_camera`、`joint_names` 和 `joint_parent_ids`。
  单独的姿态系数文件、OBJ 或 PLY 不足以作为 SOMA 输入。
- Gaussian：本项目 Harvester 的 float32 binary little-endian PLY，包含 DC 颜色、opacity logits、
  log scales、wxyz 四元数。当前拒绝高阶球谐 PLY，避免旋转后外观方向出错。
- 本地上传单文件不超过 45 MB；更大文件先放到服务器，再通过服务器路径导入。
- Harvester/GEM-X 原始输出可开启按轴向与高度初始化；本工具导出的共同坐标系 NPZ 会自动识别并跳过再次初始化，避免重复换轴。其他已对齐文件请关闭初始化。

## 配准

右侧选择操作对象 SOMA / Gaussian，独立平移、旋转、统一缩放。
支持世界/局部操作轴，数值输入，骨架/网格/高斯显隐、线框和透明度，以及点云检查模式。
`W` 平移、`E` 旋转、`F` 聚焦；`Ctrl/Cmd+Z` 撤销，`Ctrl/Cmd+Shift+Z` 重做。
工作台采用右手 Z-up 坐标：正面从 +X 看，左侧面从 +Y 看，背面从 −X 看，顶视从 +Z 看；另有透视视图。地面网格为 XY 平面。右下角实时坐标轴预览与相机同步，标记世界 +X/+Y/+Z。
导入的旧 Y-up 资产显示时统一按 `(x,y,z) → (z,x,y)` 换基，两个模型保持原有相对关系。导出的相对矩阵仍映射原始输入坐标；模型目录导出则将显示变换烘焙到共同的 Z-up 空间。
只调整整体变换，不修改单关节角度、骨长或体型，不执行动画。

## 导出目录与记忆

右上角“保存”将当前变换写入项目临时 NPZ / PLY 的原始参数，重新加载后右侧平移、旋转归零、缩放恢复为 1，配准后的模型位置保持不变；两个对象的局部坐标轴同步与世界坐标轴对齐，切换“对象局部坐标”不会恢复保存前的轴向，SOMA 操作中心仍位于髋部；保存会清除当前变换的撤销历史和浏览器草稿。源文件不变，关闭临时服务后仍需通过导出或下载保留结果。

“导出”将模型写到服务器目录，“下载”将模型保存到本地；按钮与左侧“导入 / 上传”采用相同样式。点击时选择各自目录并保留记忆，右侧不再重复显示导出目录面板。

**服务器：** 点击“导出”弹出与导入面板同风格的窗口，填写服务器上的导出根目录；右侧“浏览…”可进入服务器目录，并通过“选择当前目录”回填路径。浏览器 localStorage 记住上次路径及选择。
每次在该根目录下自动创建 `aligned-<时间戳>-<标识>/`，可以反复导出，不覆盖已有结果。
目录在服务器上，不会自动出现在本机。

**本地：** 点击“下载”时，支持 File System Access API 的浏览器会打开本机目录选择器，在 IndexedDB 中记住
目录句柄，下次默认定位上次目录；浏览器需要时会重新请求写权限。每次创建独立子目录。
目录句柄只留在浏览器，服务器不会得到本机目录路径。此能力通常需 HTTPS 或 localhost；
SSH 转发到 localhost 可使用。其他浏览器自动下载 ZIP，解压后得到同样的目录。
ZIP 的下载位置由浏览器设置决定，网页不能读取或记住任意本机完整路径。
浏览器清理站点数据、隐私模式或禁止存储时，目录记忆可能失效。

导出内容：

```text
aligned-<时间戳>-<标识>/
├── soma.npz             # 已更新原始参数及网格、骨架的 SOMA 模型
├── gaussians.ply        # 已变换的 Gaussian
├── gaussian_binding.npz # 可选：合并的权重、完整骨架和逆绑定矩阵
└── metadata.json        # 文件清单、SHA-256、配准与绑定状态
```

右上角是唯一导出入口。有效绑定完成后共 4 个文件；未绑定、绑定计算中或已失效时仅输出 3 个文件，不创建空绑定文件。绑定 NPZ 只在绑定对应当前配准且没有未保存的整体变换时附带；`metadata.json.binding` 记录是否包含及原因。有绑定时，NPZ / PLY 保留当前绑定源的精确文件内容，避免权重与源哈希错配。

导出前完整写入暂存目录，全部成功后发布结果目录；失败不发布半成品。
本地原生目录写入失败会尝试移除本次新子目录；浏览器中断仍可能留下不完整子目录。
不对文件系统、浏览器和服务器之间作跨系统原子性保证。

保存和导出都将 SOMA 的 `global_orient`（旋转矩阵）、`camera_translation`（现为共同空间的根平移）和原生 `global_scale` 同步更新，网格与骨架缓存与参数一致。
局部 `body_pose`、`identity_coeffs` 和局部骨骼 `scale_params` 保持不变；GEM-X packed57 的 GS1 固定为 1，整体缩放使用 SOMA 原生 `global_scale`，不能直接交给忽略该字段的 GEM-X 固定 GS1 解码器。
使用 SOMA 解码时，将 `global_orient` 与 `body_pose` 拼成 77 个关节旋转，`camera_translation` 传给 `transl`、`global_scale` 传给同名参数；packed57 移除首位 GS1，并按 v0.3.3 约定补齐 4 个值为 1 的 Foot/ToeBase 局部尺度。保留原模型对应的 reference pose 和 procedural profile（GEM-X 示例为 v0.2.0 / legacy_pose021）。
旧导出的 `parameter_geometry_transform` 会吸收到这些原始参数中，随后删除，避免重复应用。缺少原始参数的 NPZ 会明确拒绝保存与导出。
原始 `intrinsics`、`keypoints_2d` 移入 `original_*`，避免误用于已变换几何。
Gaussian 的中心、方向四元数和 log scale 同步变换，颜色、opacity 和行序保持不变。
导出后的两个模型已经在共同空间，`metadata.json.soma_camera_to_ply` 为单位矩阵。
未保存的配准变更不会携带旧绑定；需先保存并重新生成。当前配准的有效绑定随右上角导出一起保存。

`Ctrl/Cmd+S` 触发保存临时副本，未导出修改仍保留在当前页面和浏览器草稿。
关闭服务前请完成模型导出，草稿不包含模型文件。

## Gaussian 蒙皮绑定与 LBS

导入已经配准的 `web_server/output/aligned-20261028T073920Z-165dcf96/soma.npz` 和同目录 `gaussians.ply` 即可使用；共同空间会自动识别，不再执行初始化对齐。也可在当前项目完成配准并点击“保存”。

右侧“Gaussian 蒙皮绑定”面板选择每点影响骨骼数（默认 8），点击“生成蒙皮绑定”。后台使用独立 GEM-X 环境重建匹配的 SOMA 0.3.3 完整骨架、检查参数与网格一致性，再按最近三角面重心插值转移权重。首次配置见 [`setup/README.md`](../setup/README.md)，计算使用 64 CPU 线程、每次一个后台任务，网页仍可浏览。

右上角显示“未绑定 / 绑定中 / 已绑定 / 绑定已失效 / 绑定失败”。完成后右侧检查菜单只保留“原始外观”“主骨骼权重”“表面距离”，权重与距离检查显示 Gaussian 中心，红色距离点需人工检查。

点击视窗顶部“聚焦”和“网格”之间的 **LBS** 进入鼠标驱动测试。默认选择左前臂，也可点击关节球或通过顶部关节下拉框切换，拖动彩色旋转环驱动关节及其子骨骼。点击“恢复绑定姿态”复位；再次点击 LBS 或按 Esc 退出并恢复原模型。右上角在测试时显示“LBS 测试中”。

LBS 测试严格遵循左侧“高斯人物”显隐、“真实 Gaussian splats / 点云检查模式”选择及点云 opacity 阈值。Splats 模式在 GPU 上按全部已保存的 K 个权重更新中心和协方差 `A C Aᵀ`；点云使用同一套权重和骨骼变换。未将 8/16/32 个权重截断成渲染库默认的 4 个。测试期间隐藏未驱动的 SOMA 网格，显示的测试骨架遵循“骨架”开关；退出后恢复左侧所选图层。

这是当前完整骨架的 FK/LBS 权重测试，不重新运行 SOMA 的程序扭转约束或 correctives，也不写入原始模型参数。测试期间锁定配准编辑、保存及导出操作，退出恢复；正式生成其他 SOMA 姿态仍应使用匹配的 SOMA 完整目标骨架。命令行绑定工具保留弯肘示例；网页导出只包含模型、可选合并绑定和 metadata，不包含预览副本。

蒙皮面板不再提供单独的导出 / 下载。右上角“导出 / 下载”自动附带当前有效的 `gaussian_binding.npz`，将原 `skinning.npz` 和 `skeleton.npz` 的字段合并在同一个文件中：每点索引/权重、质量标记、完整关节名称/层级、当前绑定世界矩阵、局部矩阵和逆绑定矩阵，以及源文件 SHA-256。当前姿态和目标姿态使用同一个坐标基。

当前配准姿态就是绑定姿态，驱动使用 `target_world @ bind_inverse`。完整骨架包括程序扭转关节，不能将网页 77 个显示关节直接用于该权重。改变配准后必须重新保存并生成绑定；旧任务会标记失效，右上角仍可导出配准模型，但不会附带失效绑定。后台使用源文件快照，任务失败不发布半成品。服务关闭会终止仍在计算的任务；完成结果只在项目临时目录中，需另行导出保留。

命令行绑定、LBS 驱动及数据字段详见 [`tools/README.md`](../tools/README.md)。网页基础配准不要求 GEM-X 环境，缺少计算环境或资产时绑定面板给出错误。

## 验证

```bash
python -B -m unittest discover -s web_server/src/tests -p 'test_*.py' -v
node --test web_server/src/tests/math.test.mjs
node --check web_server/src/static/app.js
node --check web_server/src/static/export-location.js
node --check web_server/src/static/binding-ui.js
node --check web_server/src/static/lbs-interaction.js
python -B -m unittest discover -s tests -p test_gaussian_lbs.py -v
```

测试只写临时目录，验证原始参数更新与旧映射迁移、连续保存不重复变换、版本冲突、保存失败回滚、两种导入、导出中心/四元数/尺度、源文件不变、失败不发布半成品、
本地 ZIP 内容和临时目录清理。浏览器实测覆盖服务器导出、本地 ZIP 导出和目录偏好恢复。
本地原生目录选择需要用户手势，权限行为由浏览器控制。

入口为 `api.py`；`src/server.py` 负责 HTTP 路由，`src/sessions.py` 负责临时导入和模型导出，`src/models.py` 负责场景与矩阵校验，`src/bindings.py` 管理后台蒙皮任务，`src/static/` 为前端。保存只更新当前项目临时副本，导出写入新目录。

## Python 接口与目录

```text
web_server/
├── api.py                 # 唯一运行入口，头部包含运行和调用示例
├── README.md
└── src/
    ├── server.py          # HTTP 路由及 create_server
    ├── sessions.py        # 临时资产和配准导出
    ├── models.py          # 场景数据及 Sim(3) 校验
    ├── static/            # HTML、CSS、JS 和忽略的 .vendor/
    └── tests/             # Python / JavaScript 验证
```

```python
from web_server.api import create_server
server, sessions = create_server(host="127.0.0.1", port=8766)
try:
    server.serve_forever()
finally:
    server.server_close()
    sessions.close()
```

运行 `python web_server/api.py --help` 查看参数。支持 `python -m web_server.api`。
`GET /api/browse` 浏览服务器文件，`POST /api/projects` 导入临时资产，
`GET /api/scene?project=<id>` 获取场景，`POST /api/commit?project=<id>` 保存临时模型并递增版本，`POST /api/save?project=<id>` 导出模型；
蒙皮任务使用 `POST /api/bindings?project=<id>`、`GET /api/bindings?project=<id>&job=<job>`；检查文件使用 `/bindings/<job>/<filename>?project=<id>`；鼠标测试按需读取 `interaction.json` 与 `interaction.bin`（后者为 little-endian float32 `[N,K,2]` 的骨骼索引/权重对），旧绑定也可直接使用，无需重算。这两份缓存不进入绑定导出包。
下载使用 `/exports/<id>/<filename>`。接口名称保留以兼容现有前端，未保留旧页面或旧启动脚本。

安装入口统一位于 `setup/install_web.sh`，前端包下载和 SHA-512 校验由 `tools/install_web_vendor.py` 辅助执行。支持 `--prefix`、`--reuse`、`--dry-run`；网页服务只需 NumPy/SciPy，不安装 Torch、CUDA 或模型权重。
