"""Compatibility facade for checkpoint-backed custom fire inference."""
from ai.inference.fire_inference import FireInference


class FireDetector(FireInference):
    def detect(self, frame): return self.classify(frame)


_default_detector = None


def detect_fire(frame, detector=None):
    global _default_detector
    if detector is not None: return detector.detect(frame)
    if _default_detector is None: _default_detector = FireDetector()
    return _default_detector.detect(frame)
