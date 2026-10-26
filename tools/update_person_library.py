"""Stage or publish template-based FBX motions and complete mesh previews.

Parameters: --asset-root library directory; --work-dir backup/staging/check logs;
--workers parallel Blender processes; --threads CPU threads per process;
--size preview pixels; --samples Cycles samples; --only optional motion IDs.
--publish installs a fully verified batch after checking original file hashes.
The default stages results without writing to the library. Re-running resumes.
Waving 1/2/3 and the static template are always excluded from modification.

Example:
  python tools/update_person_library.py --asset-root data/3d-person-assets \
    --work-dir outputs/template-library-update --workers 12 --threads 8
  python tools/update_person_library.py --asset-root data/3d-person-assets \
    --work-dir outputs/template-library-update --publish
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.blender_runtime import find_blender, find_ffmpeg

EXCLUDED = {"waving-1", "waving-2", "waving-3", "humanoid-template"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def prepare(asset_root, work):
    manifest_path = work / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    targets = sorted(path.stem for path in (asset_root / "fbx").glob("*.fbx")
                     if path.stem not in EXCLUDED)
    assert targets
    backup = work / "backup"
    paths = ["README.md", "index.csv", "fbx/humanoid-template.fbx", "fbx/humanoid-template.retarget.json"]
    paths += [f"fbx/{name}.fbx" for name in targets]
    paths += [f"previews/{name}.mp4" for name in targets]
    protected = [f"fbx/{name}.fbx" for name in EXCLUDED]
    protected += [f"previews/waving-{i}.mp4" for i in (1, 2, 3)]
    protected += ["previews/humanoid-template.png", "fbx/humanoid-template.retarget.json"]
    hashes = {}
    for relative in sorted(set(paths + protected)):
        source = asset_root / relative
        assert source.is_file(), source
        hashes[relative] = digest(source)
        destination = backup / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    manifest = dict(targets=targets, original_hashes=hashes, protected=protected,
                    asset_root=str(asset_root), created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    save_json(manifest_path, manifest)
    return manifest


def stage_documentation(work, manifest):
    backup = work / "backup"
    stage = work / "staged"
    rows = list(csv.DictReader(io.StringIO((backup / "index.csv").read_text(encoding="utf-8-sig"))))
    for row in rows:
        if row["motion_id"] in manifest["targets"]:
            report = json.loads((work / "reports" / (row["motion_id"] + ".json")).read_text())
            row["has_mesh"] = "true"
            original_description = row["description"].split("。", 1)[0]
            row["description"] = (f"{original_description}。humanoid-template 人物网格及动画骨架，"
                                  f"{report['frames']}帧/{report['fps']:g}FPS；预览：人物工作室渲染")
    with (stage / "index.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    readme = (backup / "README.md").read_text()
    lines = readme.splitlines()
    for i, line in enumerate(lines):
        for name in manifest["targets"]:
            if line.startswith(f"| {name} |"):
                row = next(item for item in rows if item["motion_id"] == name)
                lines[i] = (f"| {name} | [{name}.fbx](fbx/{name}.fbx) | "
                            f"[{name}.mp4](previews/{name}.mp4) | {row['description']} |")
                break
    readme = "\n".join(lines) + "\n"
    readme = readme.replace(
        "本次从 `tests/Animations` 收录的 40 个动作预览为黑底彩色 BODY-18 DWPose 骨架，按 FBX 原始帧率和完整动画帧范围生成；FBX 中的网格信息仍由 `has_mesh` 标记。",
        "从 `tests/Animations` 收录的 40 个动作已更新为模板人物网格及其动画；预览按原始帧率和完整动作范围渲染人物模型，`has_mesh` 均为 `true`。",
    )
    readme = readme.replace("库中纯骨架动作的髋部位于动画骨架对象上", "原始纯骨架动作（更新前备份）的髋部位于动画骨架对象上")
    readme += ("\n## 模板人物动作更新\n\n除 waving-1/2/3 外的 40 个动作已使用 humanoid-template 重新绑定，"
               "均包含 2 个蒙皮网格和 65 根骨骼，预览为完整动作的人物工作室渲染。"
               "FBX 保留原始帧率、动作范围和完整根节点运动；为便于观看，预览取景抵消了线性位移。"
               "人物模板继续使用单张四视 PNG。更新前的 FBX、预览和索引备份保存在项目任务输出目录。\n")
    (stage / "README.md").write_text(readme)


def publish(asset_root, work, manifest):
    if (work / "published.json").exists():
        print("Batch has already been published", flush=True)
        return
    for relative, expected in manifest["original_hashes"].items():
        actual = digest(asset_root / relative)
        staged = work / "staged" / relative
        # Allow a verified interrupted publication to continue, while refusing
        # unrelated edits made to the library after its snapshot was taken.
        allowed_staged = relative not in manifest["protected"] and staged.exists() and actual == digest(staged)
        assert actual == expected or allowed_staged, f"Library changed during staging: {relative}"
    for name in manifest["targets"]:
        report_path = work / "reports" / f"{name}.json"
        assert report_path.is_file(), f"Incomplete motion: {name}"
        report = json.loads(report_path.read_text())
        assert report["preview_frames"] == report["frames"]
        assert report["minimum_limb_direction_cosine"] > .99
        for relative in (f"fbx/{name}.fbx", f"previews/{name}.mp4"):
            assert (work / "staged" / relative).is_file()
    stage_documentation(work, manifest)
    install = [f"fbx/{name}.fbx" for name in manifest["targets"]]
    install += [f"previews/{name}.mp4" for name in manifest["targets"]]
    install += ["index.csv", "README.md"]
    installed_hashes = {}
    for relative in install:
        source, destination = work / "staged" / relative, asset_root / relative
        temporary = destination.with_name(destination.name + ".template-update.tmp")
        shutil.copy2(source, temporary)
        assert digest(source) == digest(temporary)
        temporary.replace(destination)
        installed_hashes[relative] = digest(destination)
    for relative in manifest["protected"]:
        assert digest(asset_root / relative) == manifest["original_hashes"][relative]
    save_json(work / "published.json", dict(installed_hashes=installed_hashes,
                                           motion_count=len(manifest["targets"])))
    print(f"Published {len(manifest['targets'])} FBXs and previews; excluded assets unchanged", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    asset_root, work = args.asset_root.absolute(), args.work_dir.absolute()
    work.mkdir(parents=True, exist_ok=True)
    manifest = prepare(asset_root, work)
    if args.publish:
        publish(asset_root, work, manifest)
        return
    names = args.only or manifest["targets"]
    assert set(names).issubset(manifest["targets"])
    blender, ffmpeg = find_blender(), find_ffmpeg()
    for directory in ("jobs", "logs", "reports", "staged/fbx", "staged/previews", "frames"):
        (work / directory).mkdir(parents=True, exist_ok=True)
    def run(name):
        report_path = work / "reports" / f"{name}.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report["preview_frames"] == report["frames"]:
                return name, "resumed"
        job = dict(template=str(work / "backup/fbx/humanoid-template.fbx"),
                   profile=str(work / "backup/fbx/humanoid-template.retarget.json"),
                   motion=str(work / f"backup/fbx/{name}.fbx"),
                   output_fbx=str(work / f"staged/fbx/{name}.fbx"),
                   output_video=str(work / f"staged/previews/{name}.mp4"),
                   report=str(report_path), frames_dir=str(work / "frames" / name),
                   ffmpeg=ffmpeg, size=args.size, samples=args.samples, threads=args.threads,
                   denoise=True)
        job_path = work / "jobs" / f"{name}.json"
        save_json(job_path, job)
        environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads),
                           OIDN_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS="1")
        with (work / "logs" / f"{name}.log").open("w") as log:
            subprocess.run([blender, "-b", "--factory-startup", "-t", str(args.threads),
                            "--python-exit-code", "1", "--python",
                            str(ROOT / "renders/retarget_library_asset.py"), "--", "--job", str(job_path)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, env=environment)
        assert report_path.is_file()
        return name, "complete"
    errors = []
    print(f"Processing {len(names)} motions with {args.workers} workers; originals backed up", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, name): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                _, status = future.result()
                print(f"{name}: {status}", flush=True)
            except Exception as error:
                errors.append(name)
                print(f"{name}: FAILED {error}; see logs/{name}.log", flush=True)
    if errors:
        raise RuntimeError(f"Failed motions: {errors}")
    if all((work / "reports" / f"{name}.json").exists() for name in manifest["targets"]):
        stage_documentation(work, manifest)
    print("Staging complete", flush=True)


if __name__ == "__main__":
    main()
