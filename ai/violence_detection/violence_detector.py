"""Compatibility facade for clip-based custom violence inference."""
from ai.inference.violence_inference import ViolenceInference


class ViolenceDetector(ViolenceInference):
    def detect(self, frames): return self.classify(frames if isinstance(frames, (list, tuple)) else [frames])


_default_detector = None


def detect_violence(frames, detector=None):
    global _default_detector
    if detector is not None: return detector.detect(frames)
    if _default_detector is None: _default_detector = ViolenceDetector()
    return _default_detector.detect(frames)
