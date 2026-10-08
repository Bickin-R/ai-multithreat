"""
AI / Computer Vision Module for Intelligent Desktop Security Surveillance.
Team Member 1 - 24-Hour Hackathon Delivery.

This package exposes clean interfaces for:
- Camera / Video Input (ai.camera)
- Person Detection (ai.person_detection)
- Person Tracking (ai.tracking)
- Restricted Zone Detection (ai.zone_detection)
- Weapon Detection (ai.weapon_detection)
- Violence & Aggressive Behavior Detection (ai.violence_detection)
- Fire & Smoke Detection (ai.fire_detection)
- Unified Surveillance Pipeline (ai.pipeline)
"""

from ai.camera.camera_stream import CameraStream, list_available_cameras
from ai.person_detection.person_detector import PersonDetector, detect_people
from ai.tracking.tracker import PersonTracker, track_people
from ai.zone_detection.zone import RestrictedZone, check_zone, annotate_zone_status
from ai.weapon_detection.weapon_detector import WeaponDetector, detect_weapons
from ai.violence_detection.violence_detector import ViolenceDetector, detect_violence
from ai.fire_detection.fire_detector import FireDetector, detect_fire
from ai.pipeline.surveillance_pipeline import SurveillancePipeline

__all__ = [
    "CameraStream",
    "list_available_cameras",
    "PersonDetector",
    "detect_people",
    "PersonTracker",
    "track_people",
    "RestrictedZone",
    "check_zone",
    "annotate_zone_status",
    "WeaponDetector",
    "detect_weapons",
    "ViolenceDetector",
    "detect_violence",
    "FireDetector",
    "detect_fire",
    "SurveillancePipeline",
]
