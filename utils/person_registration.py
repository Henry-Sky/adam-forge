"""Staged calibrated person registration and VACE orchestration.

CLI: python interface/register_person.py --help
All task outputs and subprocess logs are confined to the chosen work directory.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from utils.person_video import (
    VideoInfo, compose, load_pose, make_mask, pad_video, prepare_reference,
    same_video_info, validate_mask, video_info,
)
from utils.person_prompts import DEFAULT_PROMPT, DEFAULT_NEGATIVE_PROMPT

ROOT = Path(__file__).resolve().parents[1]


def signature(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest(), size=path.stat().st_size)


def write_manifest(work, manifest):
    path = Path(work) / "registration.json"
    temporary = path.with_suffix(".pending.json")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read_manifest(work):
    path = Path(work) / "registration.json"
    if not path.is_file():
        raise ValueError(f"Run preview/prepare first, or supply all existing VACE inputs: {path}")
    return json.loads(path.read_text())


def check_signatures(manifest):
    for name, previous in manifest.get("input_signatures", {}).items():
        if signature(previous["path"]) != previous:
            raise ValueError(f"Input changed since preparation: {name}; use a new work directory")
    if manifest.get("calibration_signature"):
        if calibration_signature(manifest["source_data"]) != manifest["calibration_signature"]:
            raise ValueError("Calibration changed since box preview; create a new preview")


def calibration_signature(source):
    source = Path(source)
    records = []
    for folder in ("intrinsics", "extrinsics", "ego2global"):
        files = sorted((source / folder).glob("*.txt"))
        if not files:
            raise ValueError(f"Missing calibration files: {source / folder}")
        records.extend(signature(path) for path in files)
    return records


def prepare_work(work, protected=()):
    work = Path(work).resolve()
    for source in protected:
        source = Path(source).resolve()
        if work == source or (source.is_dir() and source in work.parents):
            raise ValueError("Work directory cannot replace or be inside source-data")
    work.mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(exist_ok=True)
    return work


def command(work, stage, arguments):
    args = [str(value) for value in arguments]
    log = Path(work) / "logs" / f"{stage}.log"
    print(f"{stage}: log={log}", flush=True)
    with log.open("w") as stream:
        stream.write(json.dumps(args) + "\n")
        stream.flush()
        try:
            subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"{stage} failed with exit code {error.returncode}; see {log}") from error
    return dict(stage=stage, arguments=args, log=str(log))


def preview(args):
    source = args.source_data.resolve()
    origin = args.origin_video.resolve()
    work = prepare_work(args.work_dir, (source, origin))
    if (work / "registration.json").exists():
        raise ValueError("Work directory already contains a registration; choose a new directory")
    info = video_info(origin)
    if not all(math.isfinite(v) for v in [*args.xyz, *args.lwh, args.heading]) or min(args.lwh) <= 0:
        raise ValueError("XYZ/heading must be finite; LWH must be positive")
    manifest = dict(schema_version=1, state="previewing", origin_video=str(origin),
                    source_data=str(source), video=info.to_dict(), xyz=args.xyz,
                    lwh=args.lwh, heading=args.heading,
                    calibration_signature=calibration_signature(source),
                    input_signatures={"origin": signature(origin)})
    write_manifest(work, manifest)
    try:
        step = command(work, "preview", [
            sys.executable, "-m", "utils.person_box", "--source-data", source,
            "--origin-video", origin, "--xyz", *args.xyz, "--lwh", *args.lwh,
            "--heading", args.heading, "--output", work / "box_preview.mp4",
        ])
        preview_record = json.loads((work / "box_preview.json").read_text())
        if preview_record["visible_frames"] == 0:
            raise ValueError("The box is not visible; adjust XYZ/LWH before registration")
        manifest.update(state="awaiting_position_confirmation", commands=[step])
        write_manifest(work, manifest)
    except Exception as error:
        manifest.update(state="failed", failed_stage="preview", error=str(error))
        write_manifest(work, manifest)
        raise
    print(f"Review {work / 'box_preview.mp4'}; then prepare --confirm-position", flush=True)


def attach_reference(manifest, work, source, scale):
    info = VideoInfo(**manifest["video"])
    pose = load_pose(manifest["pose_json"], info)
    rect = next(rec["rectangle_xyxy"] for rec in pose["frames"] if rec["rectangle_xyxy"])
    path = Path(work) / "reference.png"
    record = prepare_reference(source, path, info, rect, scale)
    manifest.update(reference_image=str(path), reference_transform=record)
    manifest.setdefault("input_signatures", {}).update(
        reference_source=signature(source), reference_prepared=signature(path),
    )


def prepare(args):
    work = Path(args.work_dir).resolve()
    manifest = read_manifest(work)
    retry_prepare = manifest["state"] == "failed" and manifest.get("failed_stage") == "prepare"
    if manifest["state"] != "awaiting_position_confirmation" and not retry_prepare:
        raise ValueError("Prepare requires a completed box preview in a fresh work directory")
    if not args.confirm_position:
        raise ValueError("Review box_preview.mp4, then pass --confirm-position")
    if args.animation_fps is not None and (not math.isfinite(args.animation_fps) or args.animation_fps <= 0):
        raise ValueError("Animation FPS must be positive and finite")
    check_signatures(manifest)
    from utils.blender_runtime import find_blender

    blender = shutil.which(args.blender) if args.blender else find_blender()
    if not blender:
        raise ValueError(f"Blender executable not found: {args.blender}")
    fbx = args.fbx.resolve()
    info = VideoInfo(**manifest["video"])
    manifest.update(state="preparing", position_confirmed=True, fbx=str(fbx),
                    fit_bounds=args.fit_bounds, animation_fps=args.animation_fps,
                    blender=blender)
    manifest["input_signatures"]["fbx"] = signature(fbx)
    write_manifest(work, manifest)
    try:
        timing = ["--animation-fps", args.animation_fps] if args.animation_fps else []
        step = command(work, "inspect_fbx", [
            blender, "--background", "--factory-startup", "--python-exit-code", "1",
            "--python", ROOT / "renders/inspect_fbx.py", "--", "--fbx", fbx,
            "--fps", info.fps, "--frames", info.frames, "--lwh", *manifest["lwh"],
            *timing, "--output", work / "fbx_bounds.json",
        ])
        manifest["commands"].append(step)
        bounds = json.loads((work / "fbx_bounds.json").read_text())
        manifest["bounds"] = bounds
        if not bounds["first_frame_fit_contains_all_motion"]:
            required = bounds["required_lwh_at_first_frame_scale_and_center"]
            print(f"First-frame fit exceeds the box; required LWH at that scale: {required}", flush=True)
            if args.fit_bounds == "first-frame":
                raise ValueError("FBX motion exceeds LWH; use --fit-bounds clip or revise the box")
        if bounds["armature_count"] != 1:
            raise ValueError("DWPose registration requires exactly one supported armature")
        # The low-level renderer consumes a clip layout; only links are temporary.
        with tempfile.TemporaryDirectory(prefix="layout-", dir=work) as temporary:
            layout = Path(temporary)
            asset = layout / "assets/clip-00000-00"
            video = layout / "videos/clip-00000-00/videos"
            asset.mkdir(parents=True)
            video.mkdir(parents=True)
            (asset / "source-data").symlink_to(manifest["source_data"], target_is_directory=True)
            (video / "video_origin.mp4").symlink_to(manifest["origin_video"])
            common = [
                sys.executable, ROOT / "renders/register_fbx_animation.py",
                "--asset-root", layout / "assets", "--video-root", layout / "videos",
                "--clip", "0", "--part", "0", "--fbx", fbx,
                "--xyz", *manifest["xyz"], "--lwh", *manifest["lwh"],
                "--heading", manifest["heading"], "--fit-bounds", args.fit_bounds,
                "--blender", blender, "--reference-root", ROOT,
                "--device", args.render_device, *timing,
            ]
            if bounds["has_mesh"] and not args.skip_mesh:
                manifest["commands"].append(command(work, "register_mesh", [
                    *common, "--output", work / "registered.mp4",
                ]))
                manifest["registered_video"] = str(work / "registered.mp4")
            elif not bounds["has_mesh"]:
                print("Skeleton-only FBX: generating pose control; no mesh render is available.", flush=True)
            manifest["commands"].append(command(work, "register_dwpose", [
                *common, "--mode", "dwpose-box", "--output", work / "dwpose_box.mp4",
            ]))
        pose_path = work / "dwpose_box_pose.json"
        pose = load_pose(pose_path, info)
        for rec in pose["frames"]:
            rect = rec["rectangle_xyxy"]
            if rect is None:
                if any(point is not None for point in rec["body_keypoints_xy"]):
                    raise ValueError("Visible pose has no edit rectangle")
                continue
            x0, y0, x1, y1 = rect
            for point in rec["body_keypoints_xy"]:
                if point is not None and not (x0 <= point[0] <= x1 and y0 <= point[1] <= y1):
                    raise ValueError(f"Pose leaves the edit box at frame {rec['frame_id']}; revise fit/LWH")
        mask = work / "box_mask.mp4"
        make_mask(mask, pose, info)
        control = work / "dwpose_box.mp4"
        if not same_video_info(video_info(control), info):
            raise ValueError("Registered pose video metadata differs from origin")
        manifest.update(control_video=str(control), pose_json=str(pose_path), mask_video=str(mask))
        manifest["input_signatures"].update(control=signature(control), pose=signature(pose_path), mask=signature(mask))
        if args.reference_image:
            attach_reference(manifest, work, args.reference_image, args.reference_scale)
        manifest.update(state="prepared")
        write_manifest(work, manifest)
    except Exception as error:
        manifest.update(state="failed", failed_stage="prepare", error=str(error))
        write_manifest(work, manifest)
        raise
    print(f"Prepared {work}; review full-motion fit and DWPose before run.", flush=True)


def import_existing(args):
    provided = [args.origin_video, args.control_video, args.mask_video, args.pose_json]
    if not any(provided):
        return read_manifest(args.work_dir)
    if not all(provided):
        raise ValueError("Existing inputs require --origin-video, --control-video, --mask-video and --pose-json together")
    work = prepare_work(args.work_dir, provided)
    if (work / "registration.json").exists():
        raise ValueError("Import existing inputs into a fresh work directory")
    info = video_info(args.origin_video)
    if not same_video_info(video_info(args.control_video), info):
        raise ValueError("Existing control video and origin metadata differ")
    pose = load_pose(args.pose_json, info)
    validate_mask(args.mask_video, pose, info)
    paths = dict(origin_video=args.origin_video, control_video=args.control_video,
                 mask_video=args.mask_video, pose_json=args.pose_json)
    manifest = dict(schema_version=1, state="prepared", imported_existing_inputs=True,
                    video=info.to_dict(), commands=[],
                    input_signatures={key: signature(value) for key, value in paths.items()},
                    **{key: str(value.resolve()) for key, value in paths.items()})
    write_manifest(work, manifest)
    return manifest


def choose_model_dir(explicit):
    from interface.wan_vace_inference import load_checkpoint_environment, resolve_model_files

    if explicit:
        resolve_model_files(explicit)
        return explicit.expanduser().resolve()
    cache = Path(os.environ.get("WAN_VACE_CACHE_DIR", "/tmp/vace-model-cache"))
    if cache.is_dir():
        try:
            files = resolve_model_files(cache)
            if len(files["diffusion"]) != 7:
                raise ValueError("Expected seven VACE diffusion shards")
            return cache.resolve()
        except (ValueError, KeyError, FileNotFoundError) as error:
            print(f"Incomplete local VACE cache: {error}; checking configured model path.", file=sys.stderr)
    load_checkpoint_environment()
    configured = os.environ.get("WAN_VACE_MODEL_DIR")
    if not configured:
        raise ValueError("No local model cache or configured WAN_VACE_MODEL_DIR")
    model = Path(configured)
    resolve_model_files(model)
    return model.resolve()


def run(args):
    from interface.wan_vace_inference import parse_args as inference_args

    work = Path(args.work_dir).resolve()
    manifest = import_existing(args)
    check_signatures(manifest)
    retry_run = manifest["state"] == "failed" and manifest.get("failed_stage") == "run"
    if manifest["state"] not in {"prepared", "complete"} and not retry_run:
        raise ValueError("Run requires prepared inputs")
    if args.reference_image:
        attach_reference(manifest, work, args.reference_image, args.reference_scale)
    if not manifest.get("reference_image"):
        raise ValueError("Supply --reference-image before VACE inference")
    info = VideoInfo(**manifest["video"])
    pose = load_pose(manifest["pose_json"], info)
    validate_mask(manifest["mask_video"], pose, info)
    if info.width % 16 or info.height % 16:
        raise ValueError("Native VACE inference needs width and height divisible by 16")
    model_dir = choose_model_dir(args.model_dir)
    config = (json.loads(args.prompt_config.read_text()) if args.prompt_config else
              dict(prompt=DEFAULT_PROMPT, negative_prompt=DEFAULT_NEGATIVE_PROMPT))
    prompt = args.prompt if args.prompt is not None else config["prompt"]
    negative = args.negative_prompt if args.negative_prompt is not None else config["negative_prompt"]
    padded_count = max(5, 4 * math.ceil((info.frames - 1) / 4) + 1)
    results = work / "results"
    results.mkdir(exist_ok=True)
    raw = results / "vace_raw.mp4"
    prior_outputs = (raw, results / "vace_result.mp4", results / "dwpose_vs_vace_vertical.mp4")
    if any(path.exists() for path in prior_outputs) and not args.overwrite:
        raise ValueError("VACE outputs already exist; use --overwrite or a new work directory")
    control, mask = Path(manifest["control_video"]), Path(manifest["mask_video"])
    if padded_count != info.frames:
        control, mask = work / "inference_control.mp4", work / "inference_mask.mp4"
        pad_video(manifest["control_video"], control, info, padded_count)
        pad_video(manifest["mask_video"], mask, info, padded_count)
    options = [
        "--control-video", str(control), "--mask-video", str(mask),
        "--reference-image", manifest["reference_image"], "--model-dir", str(model_dir),
        "--width", str(info.width), "--height", str(info.height), "--num-frames", str(padded_count),
        "--fps", str(info.fps), "--steps", str(args.steps), "--cfg-scale", str(args.cfg_scale),
        "--vace-scale", str(args.vace_scale), "--seed", str(args.seed),
        "--prompt", prompt, "--negative-prompt", negative, "--device", args.device,
        "--output", str(raw),
    ]
    if args.model_gpus:
        options += ["--model-gpus", *map(str, args.model_gpus)]
    if args.low_vram:
        options += ["--low-vram"]
    # Validate all CLI/model inputs before launching the expensive subprocess.
    inference_args(options + ["--dry-run"])
    manifest["inference"] = dict(
        model_dir=str(model_dir), model_gpus=args.model_gpus, device=args.device,
        prompt=prompt, negative_prompt=negative, steps=args.steps, seed=args.seed,
        cfg_scale=args.cfg_scale, vace_scale=args.vace_scale, frames=padded_count,
        control_video=str(control), mask_video=str(mask), output=str(raw), arguments=options,
    )
    write_manifest(work, manifest)
    if args.dry_run:
        print(json.dumps(manifest["inference"], indent=2))
        return
    manifest.update(state="running")
    write_manifest(work, manifest)
    try:
        step = command(work, "vace", [sys.executable, ROOT / "interface/wan_vace_inference.py", *options])
        manifest["commands"].append(step)
        manifest.update(state="generated")
        write_manifest(work, manifest)
        finalize(work, raw, args.overwrite)
    except Exception as error:
        manifest.update(state="failed", failed_stage="run", error=str(error))
        write_manifest(work, manifest)
        raise


def finalize(work, generated=None, overwrite=False):
    work = Path(work).resolve()
    manifest = read_manifest(work)
    check_signatures(manifest)
    info = VideoInfo(**manifest["video"])
    pose = load_pose(manifest["pose_json"], info)
    validate_mask(manifest["mask_video"], pose, info)
    generated = Path(generated).resolve() if generated else Path(manifest["inference"]["output"])
    raw_info = video_info(generated)
    allowed_counts = {info.frames, max(5, 4 * math.ceil((info.frames - 1) / 4) + 1)}
    if raw_info.frames not in allowed_counts:
        raise ValueError("Generated frame count is neither native nor its VACE padding count")
    results = work / "results"
    output, comparison = results / "vace_result.mp4", results / "dwpose_vs_vace_vertical.mp4"
    if not overwrite and any(path.exists() for path in (output, comparison)):
        raise ValueError("Composited outputs exist; use --overwrite")
    metrics = compose(
        manifest["origin_video"], generated, manifest["control_video"], pose,
        output, comparison, info, raw_info.frames,
    )
    manifest.update(state="complete", composite_validation=metrics,
                    outputs=dict(raw=str(generated), video=str(output), comparison=str(comparison)))
    write_manifest(work, manifest)
    print(json.dumps(manifest["outputs"], indent=2), flush=True)
