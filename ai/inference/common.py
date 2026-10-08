"""Shared checkpoint loading, image preprocessing and simple box postprocessing."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ai.device import resolve_device

INPUT_SIZE = (320, 320)


def validate_frame(frame):
    if frame is None: raise ValueError("frame must not be None")
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be an HxWx3 OpenCV BGR numpy array")
    if frame.shape[0] < 1 or frame.shape[1] < 1:
        raise ValueError("frame dimensions must be non-zero")


def load_frame(frame):
    validate_frame(frame)
    rgb = frame[..., ::-1].copy()  # OpenCV BGR to RGB
    image = Image.fromarray(rgb).resize(INPUT_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2,0,1).unsqueeze(0)


def load_checkpoint(model, path, device):
    if not path or not Path(path).is_file(): return False
    device=resolve_device(device)
    payload = torch.load(path, map_location=device, weights_only=True)
    expected_classes = list(getattr(model, "class_names", []))
    if payload.get("format_version") != 1 or payload.get("model_name") != model.__class__.__name__:
        raise ValueError(f"Checkpoint does not match architecture {model.__class__.__name__}")
    if payload.get("class_names") != expected_classes:
        if model.__class__.__name__ == "WeaponDetectorModel" and payload.get("class_names") == ["knife", "gun", "other_weapon"]:
            raise ValueError("Incompatible legacy 3-class weapon checkpoint; IDs cannot be remapped automatically to the new 5-class model")
        raise ValueError("Checkpoint class order does not match inference configuration")
    if payload.get("input_size") != [320, 320]:
        raise ValueError("Checkpoint class order or input size does not match inference configuration")
    expected_architecture = getattr(model, "architecture_id", None)
    if expected_architecture is not None:
        if payload.get("architecture_id") != expected_architecture:
            raise ValueError("Checkpoint architecture_id is missing or incompatible with this Person grid architecture")
        if payload.get("output_grid") != list(model.output_grid):
            raise ValueError(f"Checkpoint output_grid must be {list(model.output_grid)}")
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return True


def decode_grid(output, frame_shape, class_names, threshold=0.5):
    output = output.detach().cpu()[0]
    _, gh, gw = output.shape; h, w = frame_shape[:2]; detections=[]
    person_slots = len(class_names) == 1 and output.shape[0] == 10
    slots = output.reshape(2,5,gh,gw) if person_slots else output.unsqueeze(0)
    for slot in slots:
        probs = slot[0].sigmoid()
        for iy, ix in (probs >= threshold).nonzero().tolist():
            cx,cy,bw,bh = slot[1:5,iy,ix].sigmoid().tolist()
            x1=max(0,round((cx-bw/2)*w)); y1=max(0,round((cy-bh/2)*h))
            x2=min(w,round((cx+bw/2)*w)); y2=min(h,round((cy+bh/2)*h))
            score=float(probs[iy,ix]); class_id=0
            if not person_slots and output.shape[0]>5:
                cls_scores=slot[5:,iy,ix].softmax(0); cls=int(cls_scores.argmax()); score*=float(cls_scores[cls])
                class_id=cls; label=class_names[cls]
            else: label=class_names[0]
            if score < threshold or x2 <= x1 or y2 <= y1: continue
            detections.append({"class":label,"class_id":class_id,"class_name":label,
                               "confidence":score,"bbox":[x1,y1,x2,y2],
                               "center":[(x1+x2)//2,(y1+y2)//2]})
    return non_max_suppression(detections, iou_threshold=0.5)


def non_max_suppression(detections, iou_threshold=0.5):
    """Greedy class-aware NMS in original-frame pixel coordinates."""
    kept=[]
    for label in {d["class"] for d in detections}:
        pending=sorted((d for d in detections if d["class"]==label),key=lambda d:d["confidence"],reverse=True)
        while pending:
            best=pending.pop(0); kept.append(best); survivors=[]
            for item in pending:
                a,b=best["bbox"],item["bbox"]
                iw=max(0,min(a[2],b[2])-max(a[0],b[0])); ih=max(0,min(a[3],b[3])-max(a[1],b[1]))
                inter=iw*ih; area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1]); area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
                union=area_a+area_b-inter; iou=inter/union if union else 0
                if iou <= iou_threshold: survivors.append(item)
            pending=survivors
    return sorted(kept,key=lambda d:d["confidence"],reverse=True)
