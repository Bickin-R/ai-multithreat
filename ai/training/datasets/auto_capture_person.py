"""Automatically collect quality-screened webcam frames for person annotation.

Images remain unlabeled candidates. This tool never edits the person manifests.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

import cv2

from .capture_person_webcam import _session_name


def diagnose_camera(camera_index=0):
    """Print interpreter, OpenCV, device-node, and real camera-read diagnostics."""
    if camera_index < 0:
        raise ValueError("--camera must be a non-negative index")

    video0_exists = Path("/dev/video0").exists()
    requested_device = Path(f"/dev/video{camera_index}")
    requested_device_exists = requested_device.exists()
    cap = None
    opened = False
    frame_read = False
    resolution = None
    error = None
    try:
        cap = cv2.VideoCapture(camera_index)
        opened = bool(cap.isOpened())
        if opened:
            frame_read, frame = cap.read()
            frame_read = bool(frame_read and frame is not None)
            if frame_read:
                resolution = tuple(frame.shape)
    except cv2.error as exc:
        error = str(exc)
    finally:
        if cap is not None:
            cap.release()

    print(f"Python executable: {sys.executable}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Requested camera index: {camera_index}")
    print(f"/dev/video0 exists: {video0_exists}")
    if requested_device != Path("/dev/video0"):
        print(f"{requested_device} exists: {requested_device_exists}")
    print(f"OpenCV can open camera: {opened}")
    print(f"OpenCV can read a frame: {frame_read}")
    if resolution is not None:
        print(f"Frame shape (height, width, channels): {resolution}")
    if error:
        print(f"OpenCV error: {error}")
    if not opened:
        print("This Python runtime cannot access the requested camera device.", file=sys.stderr)
    elif not frame_read:
        print("The camera opened, but this Python runtime could not read a frame.", file=sys.stderr)
    return 0 if opened and frame_read else 2


def frame_quality(frame, previous=None):
    """Return (accepted, reason, measurements) using simple image heuristics."""
    if frame is None or getattr(frame, "ndim", 0) < 2:
        return False, "invalid_frame", {}
    height, width = frame.shape[:2]
    if width < 320 or height < 240:
        return False, "resolution", {"width": width, "height": height}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    metrics = {"width": width, "height": height, "laplacian_variance": blur,
               "mean_brightness": brightness}
    if blur < 35.0:
        return False, "blur", metrics
    if not 20.0 <= brightness <= 235.0:
        return False, "exposure", metrics
    if previous is not None:
        prior = cv2.resize(previous, (32, 24), interpolation=cv2.INTER_AREA)
        current = cv2.resize(frame, (32, 24), interpolation=cv2.INTER_AREA)
        difference = float(cv2.absdiff(prior, current).mean())
        metrics["previous_frame_difference"] = difference
        if difference < 2.0:
            return False, "duplicate", metrics
    return True, "accepted", metrics


def _save_without_overwrite(frame, output_dir, session):
    """Encode then create a unique path exclusively, so existing files survive."""
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return None
    prefix = f"{session}_"
    number = 1
    pattern = re.compile(re.escape(prefix) + r"(\d{6})\.jpg$")
    for path in output_dir.glob(f"{prefix}*.jpg"):
        match = pattern.fullmatch(path.name)
        if match:
            number = max(number, int(match.group(1)) + 1)
    while True:
        path = output_dir / f"{prefix}{number:06d}.jpg"
        try:
            with path.open("xb") as stream:
                stream.write(encoded.tobytes())
            return path
        except FileExistsError:
            number += 1


def _draw_status(frame, captured, target, session, interval, skipped):
    preview = frame.copy()
    lines = [f"Captured: {captured}/{target} | session: {session}",
             f"Interval: {interval:g}s | quality rejected: {skipped}",
             "Q / Esc: stop safely"]
    for index, line in enumerate(lines):
        y = 30 + index * 30
        cv2.putText(preview, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, .65,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(preview, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, .65,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return preview


def capture(camera=0, session=None, count=500, interval=1.0,
            output="data/person/images"):
    if camera < 0:
        raise ValueError("--camera must be a non-negative index")
    if count < 1:
        raise ValueError("--count must be at least 1")
    if interval <= 0:
        raise ValueError("--interval must be greater than 0")
    session = _session_name(session or datetime.now().strftime("person_%Y%m%d_%H%M%S"))
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "capture_metadata.jsonl"
    captured = skipped = frame_number = 0
    previous_saved = None
    cap = None
    try:
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            print(f"ERROR: camera index {camera} could not be opened by OpenCV.", file=sys.stderr)
            return 2
        window = "Automatic person dataset capture"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        next_capture = time.monotonic()
        while captured < count:
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"ERROR: camera index {camera} failed to return a frame; stopping safely.", file=sys.stderr)
                return 2
            frame_number += 1
            now = time.monotonic()
            if now >= next_capture:
                accepted, reason, metrics = frame_quality(frame, previous_saved)
                next_capture = now + interval
                if accepted:
                    path = _save_without_overwrite(frame, output_dir, session)
                    if path is None:
                        print("ERROR: OpenCV could not encode the captured frame.", file=sys.stderr)
                        return 2
                    row = {"image": path.name, "source": "webcam", "camera": camera,
                           "session": session, "captured_at": datetime.now(timezone.utc).isoformat(),
                           "frame_number": frame_number, "interval_seconds": interval, **metrics}
                    try:
                        with metadata_path.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                    except OSError as exc:
                        print(f"ERROR: image saved to {path}, but metadata could not be saved: {exc}", file=sys.stderr)
                        return 2
                    captured += 1
                    previous_saved = frame.copy()
                    print(f"Saved {path}")
                else:
                    skipped += 1
                    print(f"Skipped frame ({reason})")
            cv2.imshow(window, _draw_status(frame, captured, count, session, interval, skipped))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
        print(f"Capture ended. Saved {captured}/{count} image(s) in {output_dir}; metadata: {metadata_path}")
        return 0
    except cv2.error as exc:
        print(f"OpenCV error during capture: {exc}", file=sys.stderr)
        return 2
    finally:
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def _build_parser():
    parser = argparse.ArgumentParser(description="Automatically capture quality-screened, unlabeled person images from a webcam.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--check-camera", action="store_true",
                        help="Report Python/OpenCV/device access and read one frame without saving")
    parser.add_argument("--session", help="Session identifier used in unique filenames and metadata")
    parser.add_argument("--count", type=int, default=500, help="Target number of accepted images (default: 500)")
    parser.add_argument("--interval", type=float, default=1.0, help="Minimum seconds between capture attempts (default: 1)")
    parser.add_argument("--output", default="data/person/images", help="Image output directory")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.check_camera:
            return diagnose_camera(args.camera)
        return capture(args.camera, args.session, args.count, args.interval, args.output)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
