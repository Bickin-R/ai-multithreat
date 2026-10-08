"""Explicit CPU/CUDA/MPS device validation shared by training and inference."""
import torch


def resolve_device(device="cpu"):
    result=torch.device(device)
    if result.type=="cuda":
        if not torch.cuda.is_available(): raise RuntimeError(f"CUDA was requested but is unavailable: {result}")
        if result.index is not None and result.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {result.index} is unavailable")
    elif result.type=="mps":
        if not getattr(torch.backends,"mps",None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
    elif result.type!="cpu":
        raise ValueError(f"Unsupported device type: {result.type}")
    return result
