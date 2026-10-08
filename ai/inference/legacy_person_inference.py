"""Explicit, inference-only loader for legacy 20x20 Person checkpoints."""
from pathlib import Path

import torch

from ai.device import resolve_device
from ai.models import LegacyPersonDetector20
from .common import INPUT_SIZE, decode_grid, load_frame, validate_frame


def load_legacy_person_checkpoint(model, checkpoint, device="cpu"):
    """Load only the known unversioned-grid legacy Person checkpoint format.

    This function is intentionally separate from the general checkpoint
    loader: callers must opt into the legacy20 architecture explicitly.
    """
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    resolved = resolve_device(device)
    payload = torch.load(path, map_location=resolved, weights_only=True)
    if payload.get("format_version") != 1 or payload.get("model_name") != "PersonDetectorModel":
        raise ValueError("Checkpoint is not in the supported legacy Person format")
    if payload.get("class_names") != ["person"] or payload.get("input_size") != list(INPUT_SIZE):
        raise ValueError("Legacy Person checkpoint class order or input size is incompatible")
    if payload.get("architecture_id") is not None or payload.get("output_grid") is not None:
        raise ValueError("legacy20 accepts only checkpoints without architecture/grid metadata")
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(resolved).eval()


class LegacyPersonInference:
    """Person inference adapter for explicitly selected 20x20 checkpoints."""

    def __init__(self, checkpoint, device="cpu", confidence=0.5):
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.device = resolve_device(device)
        self.confidence = confidence
        self.model = load_legacy_person_checkpoint(
            LegacyPersonDetector20(), checkpoint, self.device)
        self.loaded = True

    def detect(self, frame):
        validate_frame(frame)
        with torch.no_grad():
            output = self.model(load_frame(frame).to(self.device))
            return decode_grid(output, frame.shape, ["person"], self.confidence)
