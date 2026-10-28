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
SPEC = importlib.util.spec_from_file_location("fix_decord", ROOT / "setup/fix_decord_metadata.py")
FIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIX)


class SetupHelpersTest(unittest.TestCase):
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
