"""Training, validation, and versioned state-dict checkpoint helpers."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ai.device import resolve_device

CHECKPOINT_VERSION = 1
INPUT_SIZE = [320, 320]


def checkpoint_payload(model):
    return {"format_version": CHECKPOINT_VERSION, "model_name": model.__class__.__name__,
            "state_dict": model.state_dict(), "class_names": list(getattr(model, "class_names", [])),
            "input_size": INPUT_SIZE}


def load_model_checkpoint(model, checkpoint, device="cpu"):
    """Strictly load a checkpoint made for this exact model class and class order."""
    path = Path(checkpoint)
    if not path.is_file(): raise FileNotFoundError(f"Checkpoint not found: {path}")
    device = resolve_device(device)
    payload = torch.load(path, map_location=device, weights_only=True)
    expected = list(getattr(model, "class_names", []))
    if payload.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported or missing checkpoint format_version")
    if payload.get("model_name") != model.__class__.__name__:
        raise ValueError(f"Checkpoint model {payload.get('model_name')!r} does not match {model.__class__.__name__!r}")
    if payload.get("class_names") != expected:
        if model.__class__.__name__ == "WeaponDetectorModel" and payload.get("class_names") == ["knife", "gun", "other_weapon"]:
            raise ValueError("Incompatible legacy 3-class weapon checkpoint; IDs cannot be remapped automatically to the new 5-class model")
        raise ValueError(f"Checkpoint classes {payload.get('class_names')!r} do not match {expected!r}")
    if payload.get("input_size") != INPUT_SIZE:
        raise ValueError(f"Checkpoint input_size must be {INPUT_SIZE}")
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device)


def _save_best(model, checkpoint):
    path = Path(checkpoint); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model), path)


def _check_training_args(dataset, epochs, batch_size, lr):
    if len(dataset) == 0: raise ValueError("training/validation dataset cannot be empty")
    if epochs < 1 or batch_size < 1 or lr <= 0: raise ValueError("epochs, batch_size, and lr must be positive")


def train_classifier(model, train_ds, val_ds, checkpoint, epochs=10, batch_size=16, lr=1e-3, device="cpu"):
    _check_training_args(train_ds, epochs, batch_size, lr); _check_training_args(val_ds, epochs, batch_size, lr)
    device = resolve_device(device); model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr); best = float("inf")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    for epoch in range(epochs):
        model.train(); train_loss_sum = 0.0; train_samples = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True); logits = model(x)
            if logits.ndim != 2 or logits.shape[0] != y.shape[0]: raise ValueError("classifier must return [batch, classes]")
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward(); optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y); train_samples += len(y)
        model.eval(); val_loss_sum = 0.0; correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device, dtype=torch.long); logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                val_loss_sum += float(loss) * len(y); correct += int((logits.argmax(1) == y).sum()); total += len(y)
        val_loss = val_loss_sum / total
        print(f"epoch={epoch+1} train_loss={train_loss_sum/train_samples:.5f} val_loss={val_loss:.5f} val_accuracy={correct/total:.4f}")
        if val_loss < best:
            best = val_loss; _save_best(model, checkpoint)


def validate_checkpoint(model, val_ds, checkpoint, batch_size=16, device="cpu"):
    if batch_size < 1 or len(val_ds) == 0: raise ValueError("validation dataset and batch_size must be non-empty/positive")
    load_model_checkpoint(model, checkpoint, device); model.eval()
    loader = DataLoader(val_ds, batch_size=batch_size); loss_sum = 0.0; correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device, dtype=torch.long); logits=model(x)
            loss_sum += float(torch.nn.functional.cross_entropy(logits,y))*len(y)
            correct += int((logits.argmax(1)==y).sum()); total += len(y)
    result={"mean_loss":loss_sum/total,"accuracy":correct/total,"samples":total}
    print(result); return result


def _make_targets(pred, objects, class_count):
    """Encode normalized [class,cx,cy,w,h] boxes into one object per output cell."""
    batch, channels, gh, gw = pred.shape
    target = torch.zeros_like(pred); positive = torch.zeros((batch, gh, gw), dtype=torch.bool, device=pred.device)
    target_classes = torch.zeros((batch, gh, gw), dtype=torch.long, device=pred.device)
    for b, rows in enumerate(objects):
        occupied = set()
        for row in rows:
            cls, cx, cy, bw, bh = [float(v) for v in row.tolist()]
            if not cls.is_integer() or not 0 <= int(cls) < class_count:
                raise ValueError(f"invalid class ID {cls} for {class_count} classes")
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                raise ValueError(f"invalid normalized center/size box: {(cx,cy,bw,bh)}")
            ix = min(gw-1, int(cx*gw)); iy = min(gh-1, int(cy*gh))
            if (iy,ix) in occupied:
                raise ValueError(f"multiple objects map to grid cell {(iy,ix)}; this model supports one object per cell")
            occupied.add((iy,ix)); positive[b,iy,ix]=True
            target[b,0,iy,ix]=1; target[b,1:5,iy,ix]=torch.tensor([cx,cy,bw,bh],device=pred.device)
            target_classes[b,iy,ix]=int(cls)
    return target, positive, target_classes


def _detection_loss(pred, objects, class_count):
    target, positive, target_classes = _make_targets(pred, objects, class_count)
    object_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred[:,0], target[:,0])
    box_loss = pred.sum()*0
    class_loss = pred.sum()*0
    if positive.any():
        box_pred = pred[:,1:5].sigmoid().permute(0,2,3,1)[positive]
        box_true = target[:,1:5].permute(0,2,3,1)[positive]
        box_loss = torch.nn.functional.smooth_l1_loss(box_pred, box_true)
        if pred.shape[1] > 5:
            class_logits = pred[:,5:].permute(0,2,3,1)[positive]
            class_loss = torch.nn.functional.cross_entropy(class_logits, target_classes[positive])
    return object_loss + box_loss + class_loss


def train_detector(model, train_ds, val_ds, checkpoint, epochs=10, batch_size=8, lr=1e-3, device="cpu"):
    _check_training_args(train_ds,epochs,batch_size,lr); _check_training_args(val_ds,epochs,batch_size,lr)
    device=resolve_device(device); model.to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,collate_fn=_detection_collate)
    val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False,collate_fn=_detection_collate)
    class_count=len(model.class_names); best=float("inf")
    for epoch in range(epochs):
        model.train(); loss_sum=0.0; samples=0
        for images,objects in train_loader:
            images=images.to(device); pred=model(images)
            loss=_detection_loss(pred,objects,class_count)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            loss_sum+=float(loss.detach())*len(images); samples+=len(images)
        val_loss=_detector_validation_loss(model,val_loader,class_count,device)
        print(f"epoch={epoch+1} train_loss={loss_sum/samples:.5f} val_loss={val_loss:.5f}")
        if val_loss < best: best=val_loss; _save_best(model,checkpoint)


def _detection_collate(batch):
    return torch.stack([x for x,_ in batch]), [y for _,y in batch]


def _detector_validation_loss(model, loader, class_count, device):
    model.eval(); loss_sum=0.0; samples=0
    with torch.no_grad():
        for images,objects in loader:
            images=images.to(device); loss=_detection_loss(model(images),objects,class_count)
            loss_sum+=float(loss)*len(images); samples+=len(images)
    return loss_sum/samples


def _detector_loss(model, dataset, batch_size, device):
    if len(dataset)==0 or batch_size<1: raise ValueError("validation dataset and batch_size must be non-empty/positive")
    loader=DataLoader(dataset,batch_size=batch_size,collate_fn=_detection_collate)
    return _detector_validation_loss(model,loader,len(model.class_names),resolve_device(device))
