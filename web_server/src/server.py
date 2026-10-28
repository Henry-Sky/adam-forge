"""HTTP routes for the registration workbench; launch through api.py."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlsplit, parse_qs

from .sessions import SessionStore

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent / 'static'


def handler_for(store):
    class Handler(BaseHTTPRequestHandler):
        def json_response(self, value, code=200):
            body = json.dumps(value).encode()
            self.send_response(code)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            with store.lock:
                try:
                    return self.get_request()
                except (ValueError,KeyError) as error:
                    return self.json_response({'error':str(error)},400)

        def get_request(self):
            ident=parse_qs(urlsplit(self.path).query).get('project',[''])[0]
            path = unquote(urlsplit(self.path).path)
            if path in ('/', '/adam-forge-web-server'):
                query=urlsplit(self.path).query
                self.send_response(302)
                self.send_header('Location','/adam-forge-web-server/'+('?' + query if query else ''))
                self.end_headers()
                return
            if path == '/adam-forge-web-server/':
                path='/'
            selected=store.get(ident) if ident and (path=='/api/scene' or path.startswith('/assets/')) else None
            if path == '/api/browse':
                query=parse_qs(urlsplit(self.path).query)
                suffix=query.get('type',['npz'])[0]
                if suffix not in ('npz','ply'):return self.json_response({'error':'文件类型不支持'},400)
                try:
                    folder=Path(query.get('path',[str(REPO)])[0]).expanduser().resolve()
                    if folder.is_file():folder=folder.parent
                    entries=[]
                    for child in folder.iterdir():
                        try:
                            directory=child.is_dir()
                            if directory or child.suffix.lower()=='.'+suffix:
                                entries.append({'name':child.name,'path':str(child),'directory':directory})
                        except OSError:continue
                    entries.sort(key=lambda x:(not x['directory'],x['name'].lower()))
                    return self.json_response({'path':str(folder),'parent':str(folder.parent),'entries':entries[:500],'truncated':len(entries)>500})
                except OSError as error:
                    return self.json_response({'error':f'目录无法读取：{error}'},400)
            if path == '/api/scene':
                if not selected:return self.json_response({'error':'请选择项目'},400)
                return self.json_response(selected.scene)
            paths = {'/':HERE/'index.html', '/app.js':HERE/'app.js', '/style.css':HERE/'style.css',
                '/math.js':HERE/'math.js','/export-location.js':HERE/'export-location.js', '/assets/gaussians.ply':selected.ply if selected else None,
                '/asset-import.js':HERE/'asset-import.js'}
            target = paths.get(path)
            download=False
            if path.startswith('/exports/'):
                parts=path.split('/')
                if len(parts)==4 and parts[2] in store.exports:
                    if parts[3] in ('soma.npz','gaussians.ply','registration.json'):
                        target=store.exports[parts[2]]/parts[3]
                    elif parts[3]=='download.zip':
                        target=store.root/'exports'/(parts[2]+'.zip');download=True
            if path.startswith('/vendor/'):
                root = (HERE/'.vendor').resolve()
                candidate = (root/path.removeprefix('/vendor/')).resolve()
                if candidate.is_relative_to(root):
                    target = candidate
            if not target or not target.is_file():
                return self.send_error(404)
            data = target.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type',mimetypes.guess_type(str(target))[0] or 'application/octet-stream')
            self.send_header('Content-Length',str(len(data)))
            if download:self.send_header('Content-Disposition','attachment; filename=aligned-soma-gaussians.zip')
            self.send_header('Cache-Control','no-cache')
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            path=urlsplit(self.path).path
            if path not in ('/api/save','/api/projects'):
                return self.send_error(404)
            origin = self.headers.get('Origin')
            if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                return self.json_response({'error':'跨域保存被拒绝'},403)
            try:
                size = int(self.headers.get('Content-Length','0'))
                if not 0 < size <= (128*1024*1024 if path=='/api/projects' else 65536):
                    raise ValueError('请求大小不正确')
                payload = json.loads(self.rfile.read(size))
                if path=='/api/projects':
                    result=store.create(payload)
                else:
                    ident=parse_qs(urlsplit(self.path).query).get('project',[''])[0]
                    result=store.save(ident,payload)
                self.json_response(result)
            except (ValueError,KeyError,TypeError) as error:
                self.json_response({'error':str(error)},400)
            except OSError as error:
                self.json_response({'error':f'保存失败：{error}'},500)
    return Handler



def create_server(host='127.0.0.1', port=8766, temp_dir=None):
    """Return (HTTP server, session store); caller must close both."""
    if not (HERE/'.vendor/three/build/three.module.js').is_file() or not (HERE/'.vendor/spark/dist/spark.module.js').is_file():
        raise RuntimeError('Run bash setup/install_web.sh first')
    store = SessionStore(temp_dir)
    try:
        return ThreadingHTTPServer((host, port), handler_for(store)), store
    except Exception:
        store.close()
        raise
