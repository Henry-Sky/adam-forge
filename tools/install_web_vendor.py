#!/usr/bin/env python3
"""Browser dependency helper, invoked by setup/install_web.sh (no environment creation)."""
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent.parent / 'web_server' / 'src' / 'static'
PACKAGES = {
    'three': ('0.180.0', 'sha512-o+qycAMZrh+TsE01GqWUxUIKR1AL0S8pq7zDkYOQw8GqfX8b8VoCKYUoHbhiX5j+7hr8XsuHDVU6+gkQJQKg9w=='),
    '@sparkjsdev/spark': ('2.4.0', 'sha512-aUjgBfhwDRcF3nUwiAD6Nh7IwvxEmW9liz2qrAXIm2BlPYBmn81QbXIpGFLBSr/o0TKGvEXbNIsKujgPGV/7QQ=='),
}


def main():
    for name, (version, integrity) in PACKAGES.items():
        short = name.split('/')[-1]
        target = ROOT / '.vendor' / short
        marker = target / 'installed.json'
        if marker.exists() and json.loads(marker.read_text()).get('integrity') == integrity:
            print(f'{name}@{version}: already installed')
            continue
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / 'package.tgz'
            for registry, direct in [('https://registry.npmmirror.com', True), ('https://registry.npmjs.org', False)]:
                print(f'Downloading {name}@{version} from {registry}', flush=True)
                command = ['curl', '--fail', '--silent', '--show-error', '--location', '--retry', '2', '--connect-timeout', '15', '--max-time', '180', '--output', str(archive)]
                if direct:
                    command += ['--noproxy', '*']
                command += [f'{registry}/{name}/-/{short}-{version}.tgz']
                result = subprocess.run(command)
                if result.returncode:
                    print('Source unavailable; falling back to official registry through default proxy')
                    continue
                actual = base64.b64encode(hashlib.sha512(archive.read_bytes()).digest()).decode()
                if 'sha512-' + actual != integrity:
                    print(f'WARNING: Package integrity mismatch: {name}; trying official registry', flush=True)
                    continue
                break
            else:
                raise RuntimeError(f'Could not download {name}')
            with tarfile.open(fileobj=io.BytesIO(archive.read_bytes())) as package:
                for member in package.getmembers():
                    parts = Path(member.name).parts
                    if len(parts) < 2 or parts[0] != 'package' or '..' in parts:
                        continue
                    if parts[1] not in {'build', 'examples', 'dist', 'LICENSE', 'LICENSE.md'}:
                        continue
                    if member.isfile():
                        dest = target.joinpath(*parts[1:])
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(package.extractfile(member).read())
            marker.write_text(json.dumps({'package': name, 'version': version, 'integrity': integrity}, indent=2)+'\n')
            print(f'Verified and installed {name}@{version}')


if __name__ == '__main__':
    main()
