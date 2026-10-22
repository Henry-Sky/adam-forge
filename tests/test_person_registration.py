"""Run: python -m unittest discover -s tests -p test_person_registration.py -v

Synthetic media only; every artifact is under TemporaryDirectory. No GPU,
dataset, network or real model weights are needed.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from utils.person_video import (
    VideoInfo, compose, frames, load_pose, make_mask, pad_video,
    prepare_reference, validate_mask, video_info, writer,
)
from utils.person_registration import check_signatures, signature

ROOT = Path(__file__).resolve().parents[1]


class PersonRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="person-registration-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.info = VideoInfo(64, 64, 10, 6)
        self.origin = self.base / "origin.mp4"
        self.control = self.base / "control.mp4"
        self.generated = self.base / "generated.mp4"
        for path, offset in [(self.origin, 30), (self.control, 40), (self.generated, 150)]:
            with writer(path, self.info, lossless=True) as encoder:
                for index in range(self.info.frames):
                    frame = np.full((64, 64, 3), offset + index * 15, np.uint8)
                    encoder.send(frame.tobytes())
        self.pose = dict(width=64, height=64, fps=10, frames=[
            dict(frame_id=index, rectangle_xyxy=[16, 16, 48, 48], body_keypoints_xy=[])
            for index in range(6)
        ])
        self.pose_path = self.base / "pose.json"
        self.pose_path.write_text(json.dumps(self.pose))
        self.mask = self.base / "mask.mp4"
        make_mask(self.mask, self.pose, self.info)

    def test_mask_must_match_each_projected_rectangle(self):
        validate_mask(self.mask, self.pose, self.info)
        changed = json.loads(json.dumps(self.pose))
        changed["frames"][2]["rectangle_xyxy"][0] += 1
        with self.assertRaisesRegex(ValueError, "frame 2"):
            validate_mask(self.mask, changed, self.info)

    def test_frame_padding_repeats_last_frame(self):
        padded = self.base / "padded.mp4"
        pad_video(self.origin, padded, self.info, 9)
        self.assertEqual(video_info(padded).frames, 9)
        decoded = list(frames(padded))
        for frame in decoded[6:]:
            np.testing.assert_array_equal(frame, decoded[5])

    def test_reference_keeps_aspect_and_composites_alpha(self):
        source = self.base / "rgba.png"
        rgba = np.zeros((40, 20, 4), dtype=np.uint8)
        rgba[10:30, 6:14] = [255, 0, 0, 255]
        Image.fromarray(rgba).save(source)
        output = self.base / "reference.png"
        record = prepare_reference(source, output, self.info, [16, 16, 48, 48])
        x0, y0, x1, y1 = record["placement"]
        self.assertEqual(y1 - y0, 32)
        self.assertLessEqual(abs((x1 - x0) / (y1 - y0) - 8 / 20), .02)
        pixels = np.asarray(Image.open(output))
        np.testing.assert_array_equal(pixels[0, 0], [127, 127, 127])
        np.testing.assert_array_equal(pixels[32, 32], [255, 0, 0])

    def test_composite_and_comparison_preserve_alignment(self):
        padded = self.base / "padded_generation.mp4"
        pad_video(self.generated, padded, self.info, 9)
        output, comparison = self.base / "final.mp4", self.base / "comparison.mp4"
        metrics = compose(self.origin, padded, self.control, self.pose,
                          output, comparison, self.info, 9)
        self.assertTrue(metrics["mask_outside_equal_before_encoding"])
        self.assertEqual(metrics["trimmed_padding_frames"], 3)
        self.assertEqual(video_info(output), self.info)
        self.assertEqual(video_info(comparison), VideoInfo(64, 128, 10, 6))
        originals, predictions = list(frames(self.origin)), list(frames(self.generated))
        for index, frame in enumerate(frames(output)):
            # Compare decoded inputs; final delivery adds a lossy YUV420 encode.
            self.assertLess(abs(float(frame[:12].mean()) - float(originals[index][:12].mean())), 6)
            self.assertLess(abs(float(frame[24:40, 24:40].mean()) -
                                float(predictions[index][24:40, 24:40].mean())), 6)

    def test_pose_order_is_checked(self):
        self.pose["frames"][1]["frame_id"] = 3
        self.pose_path.write_text(json.dumps(self.pose))
        with self.assertRaisesRegex(ValueError, "consecutive"):
            load_pose(self.pose_path, self.info)

    def test_modified_input_is_rejected(self):
        record = {"input_signatures": {"origin": signature(self.origin)}}
        check_signatures(record)
        with self.origin.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Input changed"):
            check_signatures(record)

    def test_prepare_requires_review_before_loading_fbx(self):
        work = self.base / "needs-confirmation"
        work.mkdir()
        (work / "registration.json").write_text(json.dumps({"state": "awaiting_position_confirmation"}))
        result = subprocess.run([
            sys.executable, str(ROOT / "interface/register_person.py"), "prepare",
            "--work-dir", str(work), "--fbx", str(self.base / "missing.fbx"),
        ], cwd=self.base, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--confirm-position", result.stderr)

    def test_cli_import_dry_run_and_compose_without_torch(self):
        reference = self.base / "appearance.png"
        Image.new("RGB", (64, 64), (127, 127, 127)).save(reference)
        model = self.base / "model"
        tokenizer = model / "google/umt5-xxl"
        tokenizer.mkdir(parents=True)
        for index in range(1, 8):
            (model / f"diffusion_pytorch_model-{index:05d}-of-00007.safetensors").touch()
        for name in ("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"):
            (model / name).touch()
        for name in ("tokenizer_config.json", "tokenizer.json"):
            (tokenizer / name).write_text("{}")
        work = self.base / "task"
        command = [
            sys.executable, str(ROOT / "interface/register_person.py"), "run",
            "--work-dir", str(work), "--origin-video", str(self.origin),
            "--control-video", str(self.control), "--mask-video", str(self.mask),
            "--pose-json", str(self.pose_path), "--reference-image", str(reference),
            "--model-dir", str(model), "--dry-run",
        ]
        result = subprocess.run(command, cwd=self.base, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((work / "registration.json").read_text())
        self.assertEqual(manifest["state"], "prepared")
        self.assertEqual(manifest["inference"]["frames"], 9)
        # Empty fake weights work only because dry-run never loads any model.
        self.assertFalse((work / "results/vace_raw.mp4").exists())
        result = subprocess.run([
            sys.executable, str(ROOT / "interface/register_person.py"), "compose",
            "--work-dir", str(work), "--generated-video", str(self.generated),
        ], cwd=self.base, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((work / "registration.json").read_text())
        self.assertEqual(manifest["state"], "complete")
        self.assertTrue(Path(manifest["outputs"]["comparison"]).is_file())


if __name__ == "__main__":
    unittest.main()
