#!/usr/bin/env python3
"""Correct decord 0.6.0's internal wheel tag to match its published filename.

Run with the target environment's Python after installing decord. The upstream
Linux x86-64 wheel is named py3-none-manylinux2010_x86_64 but contains a stale
cp36-cp36m tag: https://github.com/dmlc/decord/issues/356.
Only that exact known mismatch is changed; native libraries are left intact.
"""
import base64
import csv
import hashlib
import importlib.metadata
import io


def main():
    distribution = importlib.metadata.distribution("decord")
    if distribution.version != "0.6.0":
        return
    files = distribution.files or []
    wheel_entry = next(p for p in files if str(p).endswith(".dist-info/WHEEL"))
    wheel_path = distribution.locate_file(wheel_entry)
    original = wheel_path.read_text()
    stale = "Tag: cp36-cp36m-manylinux2010_x86_64"
    if stale not in original:
        return
    corrected = original.replace(stale, "Tag: py3-none-manylinux2010_x86_64")
    payload = corrected.encode()
    record_entry = next(p for p in files if str(p).endswith(".dist-info/RECORD"))
    record_path = distribution.locate_file(record_entry)
    rows = list(csv.reader(io.StringIO(record_path.read_text())))
    row = next(r for r in rows if r[0] == str(wheel_entry))
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
    row[1:] = ["sha256=" + digest, str(len(payload))]
    wheel_path.write_bytes(payload)
    with record_path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    print("Corrected decord 0.6.0 wheel metadata and RECORD hash")


if __name__ == "__main__":
    main()
