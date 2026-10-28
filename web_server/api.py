#!/usr/bin/env python3
"""AdamForge 网页服务的 Python 接口与唯一命令行入口。

运行示例（仓库根目录，环境依赖见 setup/README.md）：
    conda run --no-capture-output -n adam-web python web_server/api.py --port 8766
    conda run --no-capture-output -n adam-web python -m web_server.api --port 8766
    python web_server/api.py --host 127.0.0.1 --port 8766 --temp-dir /tmp/adam-web-imports

首次安装环境与网页依赖：bash setup/install_web.sh --prefix adam
页面：http://127.0.0.1:8766/adam-forge-web-server/
默认临时导入目录在服务退出时清理；--temp-dir 指定的目录由调用方管理。

Python 接口：
    from web_server.api import create_server
    server, sessions = create_server(host="127.0.0.1", port=8766)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        sessions.close()

HTTP 接口：
    GET  /api/browse?type=npz|ply&path=...   浏览服务器目录
    POST /api/projects                    临时导入 SOMA / PLY
    GET  /api/scene?project=<id>            读取场景
    POST /api/commit?project=<id>          更新项目临时模型参数
    POST /api/save?project=<id>            导出已配准文件，不覆盖导入源
    POST /api/bindings?project=<id>        后台生成完整 Gaussian 蒙皮绑定
    GET  /api/bindings?project=<id>&job=<id> 查询绑定任务
    GET  /exports/<id>/<file>              下载导出文件或 ZIP
"""
import argparse
from pathlib import Path
import signal
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web_server.src.server import create_server

__all__ = ["create_server", "main"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--temp-dir", type=Path, help="自管的临时导入目录；默认自动创建并清理")
    args = parser.parse_args()
    try:
        server, sessions = create_server(args.host, args.port, args.temp_dir)
    except (OSError, RuntimeError) as error:
        parser.error(str(error))
    print(f"AdamForge: http://{args.host}:{args.port}/adam-forge-web-server/", flush=True)
    print(f"Temporary import storage: {sessions.root}", flush=True)

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        sessions.close()


if __name__ == "__main__":
    main()
