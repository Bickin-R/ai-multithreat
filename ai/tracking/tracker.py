"""
Person Tracking Module.
Maintains consistent identities for detected persons across consecutive video frames.
Computes bounding box, center point, confidence, trajectory, velocity, and dwell time.
"""

import time
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return (intersection / union) if union > 0 else 0.0


class TrackedPerson:
    """Internal state representation for an actively tracked individual."""

    def __init__(self, person_id: int, detection: Dict[str, Any], timestamp: float):
        self.person_id = person_id
        self.bbox: List[int] = [int(v) for v in detection["bbox"]]
        self.center: List[int] = [
            int(detection.get("center", [(self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2])[0]),
            int(detection.get("center", [(self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2])[1]),
        ]
        self.confidence: float = float(detection.get("confidence", 0.90))
        self.first_seen_timestamp: float = timestamp
        self.last_seen_timestamp: float = timestamp
        self.disappeared_count: int = 0

        # Trajectory history: list of (x, y, timestamp)
        self.history: List[Tuple[int, int, float]] = [(self.center[0], self.center[1], timestamp)]

        # Velocity and acceleration metrics (pixels per second)
        self.velocity: float = 0.0
        self.acceleration: float = 0.0
        self.inside_zone: bool = bool(detection.get("inside_zone", False))

    def update(self, detection: Dict[str, Any], timestamp: float):
        """Updates track with a new matched detection."""
        old_center = self.center
        old_ts = self.last_seen_timestamp

        self.bbox = [int(v) for v in detection["bbox"]]
        self.center = [
            int((self.bbox[0] + self.bbox[2]) // 2),
            int((self.bbox[1] + self.bbox[3]) // 2),
        ]
        self.confidence = float(detection.get("confidence", self.confidence))
        self.last_seen_timestamp = timestamp
        self.disappeared_count = 0

        dt = timestamp - old_ts
        if dt > 0.001:
            dist = math.hypot(self.center[0] - old_center[0], self.center[1] - old_center[1])
            new_vel = dist / dt
            self.acceleration = (new_vel - self.velocity) / dt
            # Exponential moving average for velocity smoothing
            self.velocity = 0.7 * new_vel + 0.3 * self.velocity

        self.history.append((self.center[0], self.center[1], timestamp))
        if len(self.history) > 60:
            self.history.pop(0)

        if "inside_zone" in detection:
            self.inside_zone = bool(detection["inside_zone"])

    def mark_missed(self):
        """Increments missed counter when undetected in current frame."""
        self.disappeared_count += 1
        # Decaying velocity
        self.velocity *= 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Serializes to the required format for Team Member 2."""
        return {
            "person_id": self.person_id,
            "bbox": self.bbox,
            "center": self.center,
            "confidence": round(self.confidence, 3),
            "timestamp": round(self.last_seen_timestamp, 3),
            "inside_zone": self.inside_zone,
            "velocity": round(self.velocity, 2),
            "time_in_view": round(self.last_seen_timestamp - self.first_seen_timestamp, 2),
        }


class PersonTracker:
    """
    Robust multi-person tracker combining spatial distance and IoU overlap.
    Assigns continuous person_id values (Person #1, Person #2, etc.).
    Maintains identity through brief occlusions.
    """

    def __init__(
        self,
        max_disappeared: int = 25,
        max_distance: float = 120.0,
        min_iou: float = 0.15,
    ):
        """
        :param max_disappeared: Number of consecutive frames a track survives without detection.
        :param max_distance: Maximum centroid distance (pixels) to match a track to a detection.
        :param min_iou: Minimum IoU overlap to qualify as an identity match.
        """
        self.next_person_id = 1
        self.tracks: Dict[int, TrackedPerson] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_iou = min_iou

    def reset(self):
        """Resets tracker state."""
        self.next_person_id = 1
        self.tracks.clear()

    def update(
        self,
        detections: List[Dict[str, Any]],
        timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Updates tracked persons with current frame detections.

        :param detections: List of detection dicts with 'bbox' [x1, y1, x2, y2].
        :param timestamp: Frame timestamp (defaults to time.time()).
        :return: List of active tracked person dicts.
        """
        if timestamp is None:
            timestamp = time.time()

        # If no active tracks, register all detections as new persons
        if len(self.tracks) == 0:
            for det in detections:
                self._register(det, timestamp)
            return self.get_active_tracks()

        track_ids = list(self.tracks.keys())

        # If no detections in current frame, mark all active tracks as missed
        if len(detections) == 0:
            for tid in track_ids:
                self.tracks[tid].mark_missed()
                if self.tracks[tid].disappeared_count > self.max_disappeared:
                    del self.tracks[tid]
            return self.get_active_tracks()

        # Build cost matrix based on distance and IoU
        D = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
        for i, tid in enumerate(track_ids):
            track = self.tracks[tid]
            for j, det in enumerate(detections):
                det_bbox = det["bbox"]
                det_cx = (det_bbox[0] + det_bbox[2]) / 2.0
                det_cy = (det_bbox[1] + det_bbox[3]) / 2.0
                dist = math.hypot(track.center[0] - det_cx, track.center[1] - det_cy)
                iou = compute_iou(track.bbox, det_bbox)

                # Combined metric: distance penalized by IoU
                # If IoU is high, distance effectively shrinks
                effective_cost = dist * (1.0 - 0.5 * iou)
                D[i, j] = effective_cost

        # Greedy matching from lowest cost to highest
        matched_tracks = set()
        matched_detections = set()

        rows = D.min(axis=1).argsort()
        for row in rows:
            col = D[row].argmin()
            if row in matched_tracks or col in matched_detections:
                continue

            tid = track_ids[row]
            cost = D[row, col]
            iou = compute_iou(self.tracks[tid].bbox, detections[col]["bbox"])

            # Accept match if distance is within threshold OR IoU is significant
            if cost <= self.max_distance or iou >= self.min_iou:
                self.tracks[tid].update(detections[col], timestamp)
                matched_tracks.add(row)
                matched_detections.add(col)

        # Unmatched existing tracks
        for row, tid in enumerate(track_ids):
            if row not in matched_tracks:
                self.tracks[tid].mark_missed()
                if self.tracks[tid].disappeared_count > self.max_disappeared:
                    del self.tracks[tid]

        # Unmatched new detections -> register new tracks
        for col, det in enumerate(detections):
            if col not in matched_detections:
                self._register(det, timestamp)

        return self.get_active_tracks()

    def _register(self, detection: Dict[str, Any], timestamp: float):
        """Registers a new person and assigns a unique tracking ID."""
        track = TrackedPerson(self.next_person_id, detection, timestamp)
        self.tracks[self.next_person_id] = track
        self.next_person_id += 1

    def get_active_tracks(self) -> List[Dict[str, Any]]:
        """Returns structured data for all currently visible tracked persons."""
        return [
            track.to_dict()
            for track in self.tracks.values()
            if track.disappeared_count == 0  # Only return currently visible individuals
        ]


# Module-level default singleton tracker for direct functional usage
_global_tracker = PersonTracker()


def track_people(
    detections: List[Dict[str, Any]],
    timestamp: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Direct function interface for Team Member 2.

    Assigns stable person_id across consecutive frames.

    :param detections: List of person detections from detect_people(frame).
    :param timestamp: Optional frame timestamp.
    :return: List of tracked person dictionaries.
    """
    return _global_tracker.update(detections, timestamp)
