import torch
from ai.models import FireClassifier
from .common import load_checkpoint, load_frame, validate_frame
from ai.device import resolve_device


class FireInference:
    def __init__(self, checkpoint=None, device="cpu"):
        self.model=FireClassifier(); self.device=resolve_device(device)
        self.loaded=load_checkpoint(self.model,checkpoint,device)
        if self.loaded: self.model.to(device).eval()

    @property
    def status(self): return {"available":self.loaded,"loaded":self.loaded,
                              "reason":None if self.loaded else "No trained checkpoint supplied"}

    def classify(self, frame):
        validate_frame(frame)
        if not self.loaded: return None
        with torch.no_grad():
            prob=self.model(load_frame(frame).to(self.device)).softmax(1)[0]; idx=int(prob.argmax())
        return {"detected":idx==1,"confidence":float(prob[idx])}
