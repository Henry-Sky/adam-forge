"""Installer helper regressions; all fixtures live in temporary directories."""
import base64
import csv
import hashlib
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fix_decord", ROOT / "tools/fix_decord_metadata.py")
FIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIX)


class SetupHelpersTest(unittest.TestCase):
    def test_torch_complete_local_bundle_does_not_contact_indexes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = '''source setup/common.sh
export ADAM_PIP_WHEELHOUSE="$1" CONDA_PREFIX="$1"
python() { printf 'python'; printf ' %s' "$@"; printf '\\n'; }
reachable() { printf 'unexpected network\\n'; return 99; }
pip_at() { printf 'unexpected pip index\\n'; return 99; }
run() { if [[ "$1" == env ]]; then shift; while [[ "$1" != python ]]; do shift; done; fi; "$@"; }
install_torch
'''
            result = subprocess.run(['bash', '-c', script, 'test', str(root)],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--no-index --find-links', result.stdout)
            self.assertNotIn('unexpected', result.stdout)
            self.assertEqual((root/'adam-torch-constraints.txt').read_text(),
                             'torch==2.10.0+cu128\ntorchvision==0.25.0+cu128\n')

    def test_local_wheels_preserve_selected_index_and_ignore_inherited_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheels = root / "wheel cache"
            wheels.mkdir()
            python = root / "python"
            python.write_text(
                '#!/bin/bash\n'
                '[[ -z "${PIP_FIND_LINKS+x}" && -z "${PIP_EXTRA_INDEX_URL+x}" ]] || exit 9\n'
                'printf "index=%s\\n" "$PIP_INDEX_URL"\n'
                'printf "arg=%s\\n" "$@"\n'
            )
            python.chmod(0o755)
            result = subprocess.run(
                ["bash", "-c", '''source setup/common.sh
export PATH="$1:$PATH" ADAM_PIP_WHEELHOUSE="$2"
export PIP_FIND_LINKS=https://unwanted.invalid PIP_EXTRA_INDEX_URL=https://unwanted.invalid
pip_at https://selected.invalid/simple example-package
''', "test", str(root), str(wheels)],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("index=https://selected.invalid/simple\n", result.stdout)
            self.assertIn(f"arg=--find-links\narg={wheels}\n", result.stdout)
            self.assertIn("arg=example-package\n", result.stdout)

    def test_decord_tag_and_record_are_consistent_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = Path("decord-0.6.0.dist-info/WHEEL")
            record = wheel.with_name("RECORD")
            (root / wheel).parent.mkdir()
            (root / wheel).write_text("Tag: cp36-cp36m-manylinux2010_x86_64\n")
            (root / record).write_text(f"{wheel},old,0\n{record},,\n")
            class Distribution:
                version = "0.6.0"
                files = [wheel, record]

                def locate_file(self, entry):
                    return root / entry

            with patch.object(FIX.importlib.metadata, "distribution", return_value=Distribution()):
                FIX.main()
                payload = (root / wheel).read_bytes()
                self.assertIn(b"Tag: py3-none-manylinux2010_x86_64", payload)
                rows = list(csv.reader((root / record).read_text().splitlines()))
                digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
                self.assertEqual(rows[0], [str(wheel), "sha256=" + digest, str(len(payload))])
                before = (root / record).read_bytes()
                FIX.main()
                self.assertEqual((root / record).read_bytes(), before)

    def test_conda_hooks_allow_unset_variables_and_preserve_failure(self):
        script = '''source setup/common.sh
conda() { printf '%s' "$UNDEFINED_VENDOR_VARIABLE"; return 7; }
code=0
conda_command install || code=$?
[[ "$code" == 7 && "$-" == *u* ]]
'''
        result = subprocess.run(["bash", "-c", script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
