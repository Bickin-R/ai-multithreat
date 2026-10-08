"""Checkpoint-backed orchestration; no model is trained at application startup."""
from collections import deque
import time
from ai.inference import PersonInference, WeaponInference, ViolenceInference, FireInference
from ai.tracking.tracker import PersonTracker
from ai.zone_detection.zone import RestrictedZone, annotate_zone_status


class SurveillancePipeline:
    def __init__(self, zone=None, person_checkpoint=None, weapon_checkpoint=None,
                 violence_checkpoint=None, fire_checkpoint=None, device="cpu",
                 person_detector=None, weapon_detector=None, violence_detector=None, fire_detector=None):
        self.zone = zone if isinstance(zone, (RestrictedZone, type(None))) else RestrictedZone(zone)
        self.person_detector = person_detector or PersonInference(person_checkpoint, device)
        self.weapon_detector = weapon_detector or WeaponInference(weapon_checkpoint, device)
        self.violence_detector = violence_detector or ViolenceInference(violence_checkpoint, device)
        self.fire_detector = fire_detector or FireInference(fire_checkpoint, device)
        self.tracker = PersonTracker()
        self.frame_history = deque(maxlen=self.violence_detector.clip_length)

    def process(self, frame, timestamp=None):
        if frame is None: raise ValueError("frame must not be None")
        ts = time.time() if timestamp is None else float(timestamp)
        self.frame_history.append(frame.copy())
        raw_persons = self.person_detector.detect(frame)
        persons = self.tracker.update(raw_persons, ts) if self.person_detector.loaded else []
        annotate_zone_status(persons, self.zone, frame.shape)
        weapons = self.weapon_detector.detect(frame)
        required_frames = max(2, self.violence_detector.clip_length)
        if len(self.frame_history) >= required_frames:
            violence = self.violence_detector.classify(list(self.frame_history))
            violence_status = self.violence_detector.status
        else:
            violence = None
            violence_status = {**self.violence_detector.status,
                               "reason": f"Waiting for {required_frames} frames to form a clip"}
        fire = self.fire_detector.classify(frame)
        return {
            "timestamp": ts,
            "person_detection": {**self.person_detector.status, "detections": persons},
            "persons": persons,
            "weapon_detection": {**self.weapon_detector.status,
                                  "detections": weapons if self.weapon_detector.loaded else []},
            "weapons": weapons if self.weapon_detector.loaded else [],
            "violence_detection": {**violence_status, "result": violence},
            "violence": violence,
            "fire_detection": {**self.fire_detector.status, "result": fire},
            "fire": fire,
        }
