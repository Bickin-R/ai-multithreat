"""Compatibility facade for checkpoint-backed custom weapon inference."""
from ai.inference.weapon_inference import WeaponInference


class WeaponDetector(WeaponInference):
    pass


_default_detector = None


def detect_weapons(frame, detector=None):
    global _default_detector
    if detector is not None: return detector.detect(frame)
    if _default_detector is None: _default_detector = WeaponDetector()
    return _default_detector.detect(frame)
