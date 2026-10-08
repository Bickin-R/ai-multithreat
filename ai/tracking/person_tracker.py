"""Lightweight IoU/centroid tracker for custom Person CNN detections."""
from dataclasses import dataclass
import math


def _iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


@dataclass
class _Track:
    numeric_id: int
    bbox: list
    confidence: float
    missed_frames: int = 0

    @property
    def person_id(self):
        return f"P{self.numeric_id:03d}"

    def as_dict(self):
        return {"person_id": self.person_id,
                "bbox": list(self.bbox),
                "confidence": self.confidence,
                "tracking_state": "missing" if self.missed_frames else "tracked",
                "missed_frames": self.missed_frames}


class PersonTracker:
    """Assign persistent IDs using greedy one-to-one IoU/centroid matching.

    Tracks survive ``max_missed_frames`` consecutive empty/unmatched frames.
    IDs are monotonic for the lifetime of a tracker and are never reused.
    """

    def __init__(self, iou_threshold=0.2, max_missed_frames=5,
                 centroid_distance_threshold=100.0):
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames cannot be negative")
        if centroid_distance_threshold <= 0:
            raise ValueError("centroid_distance_threshold must be positive")
        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = int(max_missed_frames)
        self.centroid_distance_threshold = float(centroid_distance_threshold)
        self._tracks = {}
        self._next_id = 1

    def reset(self):
        """Forget all tracks while keeping IDs monotonic and unreused."""
        self._tracks.clear()

    @staticmethod
    def _validated_detection(detection):
        if not isinstance(detection, dict):
            raise ValueError("each detection must be a dictionary")
        label = detection.get("class_name", detection.get("class", "person"))
        if label != "person":
            raise ValueError(f"PersonTracker only accepts person detections, got {label!r}")
        try:
            bbox = [float(value) for value in detection["bbox"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("each detection must have a numeric bbox [x1,y1,x2,y2]") from exc
        if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
            raise ValueError("bbox must contain four finite coordinates")
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("bbox must have positive width and height")
        try:
            confidence = float(detection["confidence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("each detection must have a numeric confidence") from exc
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return bbox, confidence

    def update(self, detections):
        """Match detections to tracks and return tracked and temporarily missing IDs."""
        current = [self._validated_detection(detection) for detection in detections]
        track_ids = list(self._tracks)
        candidates = []
        for track_id in track_ids:
            track = self._tracks[track_id]
            tc = _center(track.bbox)
            for index, (box, _) in enumerate(current):
                iou = _iou(track.bbox, box)
                dc = _center(box)
                distance = math.hypot(tc[0] - dc[0], tc[1] - dc[1])
                if iou >= self.iou_threshold or distance <= self.centroid_distance_threshold:
                    # Prefer overlap; centroid similarity breaks ties and handles
                    # brief box shifts with little or no overlap.
                    proximity = max(0.0, 1.0 - distance / self.centroid_distance_threshold)
                    candidates.append((max(iou, proximity), track_id, index))

        candidates.sort(reverse=True)
        matched_tracks = set()
        matched_detections = set()
        for _, track_id, index in candidates:
            if track_id in matched_tracks or index in matched_detections:
                continue
            box, confidence = current[index]
            track = self._tracks[track_id]
            track.bbox = box
            track.confidence = confidence
            track.missed_frames = 0
            matched_tracks.add(track_id)
            matched_detections.add(index)

        for track_id in track_ids:
            if track_id not in matched_tracks:
                self._tracks[track_id].missed_frames += 1
                if self._tracks[track_id].missed_frames > self.max_missed_frames:
                    del self._tracks[track_id]

        for index, (box, confidence) in enumerate(current):
            if index not in matched_detections:
                track = _Track(self._next_id, box, confidence)
                self._tracks[self._next_id] = track
                self._next_id += 1

        return [track.as_dict() for track in self._tracks.values()]
