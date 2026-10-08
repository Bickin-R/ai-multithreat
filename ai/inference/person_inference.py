import torch
from ai.models import PersonDetectorModel
from .common import load_checkpoint, load_frame, decode_grid, validate_frame
from ai.device import resolve_device


class PersonInference:
    def __init__(self, checkpoint=None, device="cpu", confidence=0.5):
        if not 0.0 <= confidence <= 1.0: raise ValueError("confidence must be between 0 and 1")
        self.model=PersonDetectorModel(); self.device=resolve_device(device); self.confidence=confidence
        self.loaded=load_checkpoint(self.model,checkpoint,device)
        if self.loaded: self.model.to(device).eval()

    @property
    def status(self): return {"available":self.loaded,"loaded":self.loaded,
                              "reason":None if self.loaded else "No trained checkpoint supplied"}

    def detect(self, frame):
        validate_frame(frame)
        if not self.loaded: return []
        with torch.no_grad():
            return decode_grid(self.model(load_frame(frame).to(self.device)),frame.shape,["person"],self.confidence)
