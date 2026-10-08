"""Live webcam inference for explicitly selected custom Person CNNs."""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np

from .legacy_person_inference import LegacyPersonInference
from .person_inference import PersonInference
from ai.tracking.person_tracker import PersonTracker
from ai.security.restricted_zone import RestrictedZone


def build_inference(architecture, checkpoint, device, confidence):
    """Construct a detector only for the explicitly requested architecture."""
    if architecture == "legacy20":
        return LegacyPersonInference(checkpoint, device=device, confidence=confidence)
    if architecture == "current40":
        return PersonInference(checkpoint, device=device, confidence=confidence)
    raise ValueError(f"Unsupported Person architecture: {architecture}")


def annotate_frame(frame, detections, fps, zone=None):
    """Draw detections and FPS on a copy, leaving the camera frame untouched."""
    display = frame.copy()
    if zone is not None:
        polygon = np.asarray(zone.points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(display, [polygon], isClosed=True, color=(255, 200, 0), thickness=2)
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        confidence = detection["confidence"]
        visible = detection.get("tracking_state", "tracked") == "tracked"
        if detection.get("inside_zone"):
            color = (0, 255, 0)
        else:
            color = (0, 0, 255) if visible else (0, 165, 255)
        cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        person_id = detection.get("person_id")
        label = f"ID: {person_id}" if person_id else f"person {confidence:.2f}"
        if person_id:
            label += f" {confidence:.2f}"
            if detection.get("inside_zone"):
                label += " IN ZONE"
            if not visible:
                label += " missing"
        cv2.putText(display, label, (int(x1), max(18, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(display, f"FPS: {fps:.1f}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return display


def run_webcam(detector, camera=0, window_name="Person CNN live inference",
               headless=False, output="debug/webcam_person", max_frames=None,
               tracker=None, zone=None):
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be a positive integer")
    if zone is not None and tracker is None:
        raise ValueError("restricted-zone display requires person tracking")
    output_dir = Path(output)
    if headless:
        output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open webcam camera {camera}")
    frames_processed = detections_observed = frames_saved = 0
    try:
        previous_time = time.perf_counter()
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to read a frame from webcam camera {camera}")
            detections = detector.detect(frame)
            displayed_detections = tracker.update(detections) if tracker else detections
            if zone is not None:
                displayed_detections = zone.evaluate(displayed_detections)
            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 1e-9)
            previous_time = now
            annotated = annotate_frame(frame, displayed_detections, fps, zone=zone)
            frames_processed += 1
            detections_observed += len(detections)
            if headless:
                # Save the first frame and then periodically so short smoke
                # runs also produce a visual artifact.
                if frames_processed == 1 or frames_processed % 30 == 0:
                    destination = output_dir / f"person_{frames_processed:06d}.jpg"
                    if not cv2.imwrite(str(destination), annotated):
                        raise RuntimeError(f"Could not save annotated frame: {destination}")
                    frames_saved += 1
            else:
                try:
                    cv2.imshow(window_name, annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error as exc:
                    raise RuntimeError(
                        "OpenCV GUI support is unavailable. Rerun with --headless "
                        "to save annotated frames without a display window."
                    ) from exc
            if max_frames is not None and frames_processed >= max_frames:
                break
    finally:
        capture.release()
        if not headless:
            # Some headless OpenCV builds throw from destroyAllWindows too.
            # Cleanup must not mask the actionable --headless error above.
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
    if headless:
        print(f"Headless capture complete: frames={frames_processed}, "
              f"detections={detections_observed}, annotated_frames_saved={frames_saved}, "
              f"output={output_dir}")
    return {"frames_processed": frames_processed,
            "detections_observed": detections_observed,
            "frames_saved": frames_saved}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/person_augmented.pt")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--architecture", choices=("legacy20", "current40"), required=True,
                        help="Required so checkpoint architecture is never guessed")
    parser.add_argument("--headless", action="store_true",
                        help="Run without OpenCV GUI and periodically save annotated frames")
    parser.add_argument("--output", default="debug/webcam_person",
                        help="Directory for headless annotated frames")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after this many processed frames (useful for smoke tests)")
    parser.add_argument("--track", action="store_true",
                        help="Track detected people across frames and draw persistent IDs")
    parser.add_argument("--zone", action="store_true",
                        help="Evaluate tracked foot points against a configured polygon")
    parser.add_argument("--zone-points", default=None,
                        help="Semicolon-separated frame points, e.g. '100,100;500,100;500,400;100,400'")
    return parser


def parse_zone_points(value):
    """Parse semicolon-separated x,y coordinates into a polygon."""
    if not value:
        raise ValueError("--zone requires --zone-points with at least three x,y pairs")
    try:
        points = [tuple(float(axis.strip()) for axis in item.split(","))
                  for item in value.split(";")]
    except ValueError as exc:
        raise ValueError("zone points must use 'x,y;x,y;x,y' format") from exc
    if any(len(point) != 2 for point in points):
        raise ValueError("zone points must use 'x,y;x,y;x,y' format")
    return RestrictedZone(points)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        zone = parse_zone_points(args.zone_points) if args.zone else None
        detector = build_inference(args.architecture, args.checkpoint, args.device, args.confidence)
        run_webcam(detector, args.camera, headless=args.headless,
                   output=args.output, max_frames=args.max_frames,
                   tracker=PersonTracker() if args.track or args.zone else None,
                   zone=zone)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    main()
