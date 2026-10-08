"""Compatibility facade for checkpoint-backed custom person inference."""
from ai.inference.person_inference import PersonInference


class PersonDetector(PersonInference):
    def __init__(self, checkpoint=None, confidence=0.5, device="cpu"):
        super().__init__(checkpoint, device, confidence)


_default_detector = None


def detect_people(frame, detector=None):
    global _default_detector
    if detector is not None: return detector.detect(frame)
    if _default_detector is None: _default_detector = PersonDetector()
    return _default_detector.detect(frame)
