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
- Harvester/GEM-X 原始输出可开启按轴向与高度初始化；已在共同坐标系的文件请关闭。

## 配准

右侧选择操作对象 SOMA / Gaussian，独立平移、旋转、统一缩放。
支持世界/局部操作轴，数值输入，骨架/网格/高斯显隐、线框和透明度，以及点云检查模式。
`W` 平移、`E` 旋转、`F` 聚焦；`Ctrl/Cmd+Z` 撤销，`Ctrl/Cmd+Shift+Z` 重做。
工作台采用右手 Z-up 坐标：正面从 +X 看，左侧面从 +Y 看，背面从 −X 看，顶视从 +Z 看；另有透视视图。地面网格为 XY 平面。右下角实时坐标轴预览与相机同步，标记世界 +X/+Y/+Z。
导入的旧 Y-up 资产显示时统一按 `(x,y,z) → (z,x,y)` 换基，两个模型保持原有相对关系。导出的相对矩阵仍映射原始输入坐标；模型目录导出则将显示变换烘焙到共同的 Z-up 空间。
只调整整体变换，不修改单关节角度、骨长或体型，不执行动画。

## 导出目录与记忆

右上角“导出”将模型写到服务器目录，“下载”将模型保存到本地；按钮与左侧“导入 / 上传”采用相同样式。点击时选择各自目录并保留记忆，右侧不再重复显示导出目录面板。

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
├── soma.npz             # 已变换的 SOMA 网格、骨架及保留参数
├── gaussians.ply        # 已变换的 Gaussian
└── registration.json    # 应用的两个对象变换、来源哈希和坐标说明
```

导出前完整写入暂存目录，全部成功后发布结果目录；失败不发布半成品。
本地原生目录写入失败会尝试移除本次新子目录；浏览器中断仍可能留下不完整子目录。
不对文件系统、浏览器和服务器之间作跨系统原子性保证。

SOMA 网格与骨架写入共同导出空间；原始体型/姿态参数保留，并累计
`parameter_geometry_transform` 表明参数解码几何到导出几何的映射。
原始相机相关字段移入 `original_*`，避免误用于已变换几何。
Gaussian 的中心、方向四元数和 log scale 同步变换，颜色、opacity 和行序保持不变。
导出后的两个模型已经在共同空间，`registration.json.soma_camera_to_ply` 为单位矩阵。
旧蒙皮权重需要针对新文件重新计算。

`Ctrl/Cmd+S` 触发服务器导出，未导出修改仍保留在当前页面和浏览器草稿。
关闭服务前请完成模型导出，草稿不包含模型文件。

## 验证

```bash
python -B -m unittest discover -s web_server/src/tests -p 'test_*.py' -v
node --test web_server/src/tests/math.test.mjs
node --check web_server/src/static/app.js
node --check web_server/src/static/export-location.js
```

测试只写临时目录，验证两种导入、导出中心/四元数/尺度、输入不变、失败不发布半成品、
本地 ZIP 内容和临时目录清理。浏览器实测覆盖服务器导出、本地 ZIP 导出和目录偏好恢复。
本地原生目录选择需要用户手势，权限行为由浏览器控制。

入口为 `api.py`；`src/server.py` 负责 HTTP 路由，`src/sessions.py` 负责临时导入和模型导出，`src/models.py` 负责场景与矩阵校验，`src/static/` 为前端。已移除旧固定资产模式、项目列表页面及覆盖保存代码。

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
`GET /api/scene?project=<id>` 获取场景，`POST /api/save?project=<id>` 导出模型；
下载使用 `/exports/<id>/<filename>`。接口名称保留以兼容现有前端，未保留旧页面或旧启动脚本。

安装入口统一位于 `setup/install_web.sh`，前端包下载和 SHA-512 校验由 `setup/install_web_vendor.py` 辅助执行。支持 `--prefix`、`--reuse`、`--dry-run`；网页服务只需 NumPy/SciPy，不安装 Torch、CUDA 或模型权重。
