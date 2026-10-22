"""Media preparation and frame-aligned VACE compositing for register_person.py.

OpenCV/Pillow/imageio-ffmpeg are imported only by media operations, allowing
the interface's --help to run without GPU or optional inference dependencies.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int

    def to_dict(self):
        return asdict(self)


def video_info(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        info = VideoInfo(
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            capture.get(cv2.CAP_PROP_FPS),
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()
    if min(info.width, info.height, info.frames) <= 0 or not math.isfinite(info.fps) or info.fps <= 0:
        raise ValueError(f"Cannot read video metadata: {path}")
    return info


def same_video_info(actual, expected):
    return (actual.width, actual.height, actual.frames) == (
        expected.width, expected.height, expected.frames
    ) and math.isclose(actual.fps, expected.fps, rel_tol=1e-6, abs_tol=1e-6)


def frames(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
    finally:
        capture.release()


@contextmanager
def writer(path, info, lossless=False):
    import imageio_ffmpeg

    path = Path(path)
    if info.width % 2 or info.height % 2:
        raise ValueError("MP4 output requires even width and height")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".pending.mp4")
    params = ["-crf", "0" if lossless else "18", "-movflags", "+faststart"]
    if not lossless:
        params += ["-profile:v", "high"]
    encoder = imageio_ffmpeg.write_frames(
        str(temporary), (info.width, info.height), fps=info.fps,
        codec="libx264", pix_fmt_in="bgr24",
        pix_fmt_out="yuv444p" if lossless else "yuv420p",
        macro_block_size=1, quality=None, output_params=params,
    )
    encoder.send(None)
    try:
        yield encoder
        encoder.close()
        if not same_video_info(video_info(temporary), info):
            raise ValueError(f"Unexpected encoded metadata: {temporary}")
        temporary.replace(path)
    except BaseException:
        encoder.close()
        temporary.unlink(missing_ok=True)
        raise


def load_pose(path, info):
    pose = json.loads(Path(path).read_text())
    expected = VideoInfo(pose["width"], pose["height"], pose["fps"], len(pose["frames"]))
    if not same_video_info(expected, info):
        raise ValueError("Pose and origin video metadata differ")
    for index, rec in enumerate(pose["frames"]):
        if rec["frame_id"] != index:
            raise ValueError("Pose frames must be consecutive and ordered")
        rect = rec["rectangle_xyxy"]
        if rect is not None:
            if len(rect) != 4 or any(type(v) is not int for v in rect):
                raise ValueError("Pose rectangles must use integer pixel coordinates")
            x0, y0, x1, y1 = rect
            if not (0 <= x0 < x1 <= info.width and 0 <= y0 < y1 <= info.height):
                raise ValueError(f"Invalid edit rectangle at frame {index}")
    if not any(rec["rectangle_xyxy"] for rec in pose["frames"]):
        raise ValueError("No visible edit box in pose sequence")
    return pose


def rectangle_mask(rectangle, info):
    import numpy as np

    mask = np.zeros((info.height, info.width), dtype=bool)
    if rectangle is not None:
        x0, y0, x1, y1 = rectangle
        mask[y0:y1, x0:x1] = True
    return mask


def validate_mask(path, pose, info):
    import numpy as np

    if not same_video_info(video_info(path), info):
        raise ValueError("Mask and origin video metadata differ")
    count = 0
    for index, frame in enumerate(frames(path)):
        if index >= info.frames:
            raise ValueError("Mask contains extra frames")
        actual = frame.mean(axis=2) >= 128
        expected = rectangle_mask(pose["frames"][index]["rectangle_xyxy"], info)
        if not np.array_equal(actual, expected):
            raise ValueError(f"Mask differs from projected box at frame {index}")
        count += 1
    if count != info.frames:
        raise ValueError("Mask ended before the pose sequence")


def make_mask(path, pose, info):
    import numpy as np

    with writer(path, info, lossless=True) as encoder:
        for rec in pose["frames"]:
            mask = rectangle_mask(rec["rectangle_xyxy"], info).astype("uint8") * 255
            encoder.send(np.repeat(mask[:, :, None], 3, axis=2).tobytes())
    validate_mask(path, pose, info)


def prepare_reference(source, output, info, rectangle, scale=1.0):
    from PIL import Image

    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Reference scale must be positive and finite")
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Prepared reference must not replace its input")
    with Image.open(source) as image:
        image = image.convert("RGBA")
    canvas = Image.new("RGBA", (info.width, info.height), (127, 127, 127, 255))
    # Keep appearance detail independent from the scene's projected edit box.
    # The VACE entry point handles aspect-preserving canvas adaptation.
    if scale == 1:
        canvas = Image.new("RGBA", image.size, (127, 127, 127, 255))
        canvas.alpha_composite(image)
        placement = [0, 0, image.width, image.height]
    else:
        bounds = image.getchannel("A").getbbox()
        if bounds is None:
            raise ValueError("Reference is completely transparent")
        image = image.crop(bounds)
        x0, y0, x1, y1 = rectangle
        factor = min((x1 - x0) / image.width, (y1 - y0) / image.height) * scale
        size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
        left, top = round((x0 + x1 - size[0]) / 2), round((y0 + y1 - size[1]) / 2)
        if left < 0 or top < 0 or left + size[0] > info.width or top + size[1] > info.height:
            raise ValueError("Reference scale crops the character outside the canvas")
        canvas.alpha_composite(image.resize(size, Image.Resampling.LANCZOS), (left, top))
        placement = [left, top, left + size[0], top + size[1]]
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output)
    return dict(source=str(source.resolve()), output=str(output.resolve()),
                scale=scale, placement=placement, background=[127, 127, 127])


def pad_video(source, output, info, count):
    if count < info.frames:
        raise ValueError("Padding cannot remove frames")
    target = VideoInfo(info.width, info.height, info.fps, count)
    decoded = 0
    last = None
    with writer(output, target, lossless=True) as encoder:
        for frame in frames(source):
            if decoded >= info.frames:
                raise ValueError("Input contains extra frames")
            encoder.send(frame.tobytes())
            last = frame
            decoded += 1
        if decoded != info.frames:
            raise ValueError("Input ended before its advertised frame count")
        for _ in range(count - decoded):
            encoder.send(last.tobytes())


def compose(origin, generated, control, pose, output, comparison, info, generated_frames):
    import numpy as np

    raw = video_info(generated)
    if not same_video_info(raw, VideoInfo(info.width, info.height, info.fps, generated_frames)):
        raise ValueError("Generated video dimensions, FPS or frame count differ from the run")
    if not same_video_info(video_info(origin), info) or not same_video_info(video_info(control), info):
        raise ValueError("Origin/control changed since preparation")
    protected = {Path(path).resolve() for path in (origin, generated, control)}
    if Path(output).resolve() in protected or Path(comparison).resolve() in protected:
        raise ValueError("Composited outputs must not replace inputs")
    joined = VideoInfo(info.width, info.height * 2, info.fps, info.frames)
    source_iter, raw_iter, control_iter = frames(origin), frames(generated), frames(control)
    try:
        with writer(output, info) as final, writer(comparison, joined) as stacked:
            for index in range(info.frames):
                try:
                    background, prediction, top = next(source_iter), next(raw_iter), next(control_iter)
                except StopIteration as error:
                    raise ValueError(f"Missing input frame {index}") from error
                mask = rectangle_mask(pose["frames"][index]["rectangle_xyxy"], info)
                result = background.copy()
                result[mask] = prediction[mask]
                if not np.array_equal(result[~mask], background[~mask]):
                    raise ValueError("Compositing changed unmasked source pixels")
                final.send(result.tobytes())
                stacked.send(np.ascontiguousarray(np.vstack((top, result))).tobytes())
            for iterator in (source_iter, control_iter):
                if next(iterator, None) is not None:
                    raise ValueError("Origin/control has unexpected extra frames")
            for _ in range(generated_frames - info.frames):
                if next(raw_iter, None) is None:
                    raise ValueError("Padded generation ended early")
            if next(raw_iter, None) is not None:
                raise ValueError("Generated video has unexpected extra frames")
    finally:
        for iterator in (source_iter, raw_iter, control_iter):
            iterator.close()
    return dict(mask_outside_equal_before_encoding=True, frames=info.frames,
                trimmed_padding_frames=generated_frames - info.frames)
