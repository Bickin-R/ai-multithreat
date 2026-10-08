import torch
from ai.models import ViolenceClassifier
from .common import load_checkpoint, load_frame, validate_frame
from ai.device import resolve_device


class ViolenceInference:
    def __init__(self, checkpoint=None, device="cpu", clip_length=8):
        if clip_length < 2: raise ValueError("clip_length must be at least two frames")
        self.model=ViolenceClassifier(); self.device=resolve_device(device); self.clip_length=clip_length
        self.loaded=load_checkpoint(self.model,checkpoint,device)
        if self.loaded: self.model.to(device).eval()

    @property
    def status(self): return {"available":self.loaded,"loaded":self.loaded,
                              "reason":None if self.loaded else "No trained checkpoint supplied"}

    def classify(self, frames):
        if not isinstance(frames,(list,tuple)) or len(frames)<2:
            raise ValueError("frames must be a list/tuple containing at least two frames")
        for frame in frames: validate_frame(frame)
        if not self.loaded: return None
        indices=[min(int(i*len(frames)/self.clip_length),len(frames)-1) for i in range(self.clip_length)]
        clip=torch.cat([load_frame(frames[i]) for i in indices],dim=0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob=self.model(clip).softmax(1)[0]; idx=int(prob.argmax())
        return {"detected":idx==1,"confidence":float(prob[idx]),"type":"violence" if idx==1 else "normal"}
