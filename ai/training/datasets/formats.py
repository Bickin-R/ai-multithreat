"""Strict JSONL dataset readers and validation for custom models."""
import json
import hashlib
import math
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError
from torch.utils.data import Dataset
from .source_provenance import read_source_manifest, record_provenance_errors

SIZE = (320, 320)
PERSON_CLASSES = ["person"]
WEAPON_CLASSES = ["knife", "gun", "arivaal", "baseball_bat", "other_weapon"]
FIRE_CLASSES = ["normal", "fire"]
VIOLENCE_CLASSES = ["normal", "violence"]


def load_rgb(path):
    try:
        with Image.open(path) as image:
            return _image_to_tensor(image.convert("RGB"))
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Cannot read RGB image {path}: {exc}") from exc


def _image_to_tensor(image):
    array = np.asarray(image.resize(SIZE), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def _file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def _records(root, manifest, kind, class_names):
    root, path = Path(root).resolve(), Path(manifest)
    source_manifest, source_manifest_error = read_source_manifest(root)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: malformed JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: each record must be a JSON object")
        refs = row.get("frames") if kind == "clip" else [row.get("image")]
        if kind == "clip":
            if not isinstance(refs, list) or len(refs) < 2:
                raise ValueError(f"{path}:{line_no}: clip needs at least two frame paths")
        elif not isinstance(refs[0], str) or not refs[0]:
            raise ValueError(f"{path}:{line_no}: missing non-empty image path")
        for ref in refs:
            if not isinstance(ref, str) or not ref:
                raise ValueError(f"{path}:{line_no}: frame/image paths must be non-empty strings")
            image_path = (root / ref).resolve()
            if not image_path.is_relative_to(root):
                raise ValueError(f"{path}:{line_no}: image path escapes dataset root: {ref}")
            if not image_path.is_file():
                raise FileNotFoundError(f"{path}:{line_no}: missing image/frame: {image_path}")
            try:
                with Image.open(image_path) as image:
                    if image.width <= 0 or image.height <= 0:
                        raise ValueError("zero-sized image")
                    image.verify()
            except (OSError, UnidentifiedImageError) as exc:
                raise ValueError(f"{path}:{line_no}: invalid image/frame {image_path}: {exc}") from exc

        if kind in ("classification", "clip"):
            if "label" not in row or isinstance(row["label"], bool) or not isinstance(row["label"], int):
                raise ValueError(f"{path}:{line_no}: label must be an integer class ID")
            if not 0 <= row["label"] < len(class_names):
                raise ValueError(f"{path}:{line_no}: class ID {row['label']} outside 0..{len(class_names)-1}")
        if kind == "detection":
            if class_names == PERSON_CLASSES:
                provenance_errors = record_provenance_errors(
                    row, source_manifest, source_manifest_error)
                if provenance_errors:
                    raise ValueError(f"{path}:{line_no}: " + "; ".join(provenance_errors))
            if class_names == WEAPON_CLASSES and row.get("reviewed") is not True:
                raise ValueError(f"{path}:{line_no}: weapon records require reviewed=true; pending annotations cannot train")
            if "objects" not in row or not isinstance(row["objects"], list):
                raise ValueError(f"{path}:{line_no}: objects must be a list (empty list means verified negative)")
            with Image.open(root / row["image"]) as image:
                width, height = image.size
            for obj_no, obj in enumerate(row["objects"]):
                prefix = f"{path}:{line_no}: object {obj_no}"
                if not isinstance(obj, dict) or "class" not in obj or "bbox" not in obj:
                    raise ValueError(f"{prefix}: requires integer class and bbox [x1,y1,x2,y2]")
                cls, box = obj["class"], obj["bbox"]
                if isinstance(cls, bool) or not isinstance(cls, int) or not 0 <= cls < len(class_names):
                    raise ValueError(f"{prefix}: class ID must be in 0..{len(class_names)-1}")
                if not isinstance(box, list) or len(box) != 4:
                    raise ValueError(f"{prefix}: bbox must be [x1,y1,x2,y2]")
                try:
                    if any(isinstance(v, bool) for v in box): raise ValueError("boolean coordinate")
                    x1, y1, x2, y2 = [float(v) for v in box]
                except (TypeError, ValueError) as exc: raise ValueError(f"{prefix}: bbox coordinates must be numeric") from exc
                if not all(math.isfinite(v) for v in (x1,y1,x2,y2)) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"{prefix}: invalid/out-of-image XYXY bbox {box} for {width}x{height}")
        rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def validate_dataset_splits(root, train_manifest, val_manifest, test_manifest, kind, class_names):
    """Validate all splits and reject source image/frame overlap across splits."""
    split_rows = {name: _records(root, path, kind, class_names) for name, path in
                  (("train", train_manifest), ("validation", val_manifest), ("test", test_manifest)) if path}
    sources = {}
    hashes = {}
    source_groups = {}
    root = Path(root).resolve()
    for split, rows in split_rows.items():
        within_split = set()
        for row in rows:
            refs = row["frames"] if kind == "clip" else [row["image"]]
            sample_key = tuple(str((root / ref).resolve()) for ref in refs)
            if sample_key in within_split:
                raise ValueError(f"Duplicate sample in {split} split: {refs}")
            within_split.add(sample_key)
            if kind == "detection" and class_names == WEAPON_CLASSES and row.get("reviewed") is not True:
                raise ValueError(f"Unreviewed weapon record cannot be used: {row['image']}")
            if kind == "detection" and row.get("source_group"):
                group = row["source_group"]
                previous_group_split = source_groups.get(group)
                if previous_group_split and previous_group_split != split:
                    raise ValueError(f"Data leakage: source group {group!r} appears in {previous_group_split} and {split}")
                source_groups[group] = split
            for ref in refs:
                resolved = str((root / ref).resolve())
                prior = sources.get(resolved)
                if prior and prior != split:
                    raise ValueError(f"Data leakage: {ref} appears in {prior} and {split} splits")
                sources[resolved] = split
                if kind == "detection":
                    digest = _file_hash(resolved)
                    prior_hash = hashes.get(digest)
                    if prior_hash and prior_hash[1] != split:
                        raise ValueError(f"Duplicate image content leaks between {prior_hash[1]} and {split}: {ref}")
                    hashes[digest] = (resolved, split)
    return {name: len(rows) for name, rows in split_rows.items()}


class ImageClassificationDataset(Dataset):
    def __init__(self, root, manifest, class_names=FIRE_CLASSES):
        self.root = Path(root); self.rows = _records(root, manifest, "classification", class_names)
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        return load_rgb(self.root / row["image"]), int(row["label"])


class DetectionDataset(Dataset):
    """XYXY pixel boxes become normalized boxes; augmentation is opt-in."""
    def __init__(self, root, manifest, class_names=PERSON_CLASSES, augment=False):
        self.root = Path(root); self.rows = _records(root, manifest, "detection", class_names)
        self.augment = bool(augment)
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image"]) as source:
            width, height = source.size
            image = source.convert("RGB").resize(SIZE)
        objects = []
        for obj in row["objects"]:
            x1,y1,x2,y2 = map(float,obj["bbox"])
            objects.append([obj["class"],(x1+x2)/(2*width),(y1+y2)/(2*height),
                            (x2-x1)/width,(y2-y1)/height])
        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                for obj in objects:
                    obj[1] = 1.0 - obj[1]
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        return _image_to_tensor(image), torch.tensor(objects,dtype=torch.float32).reshape(-1,5)


class ClipClassificationDataset(Dataset):
    def __init__(self, root, manifest, class_names=VIOLENCE_CLASSES, clip_length=8):
        self.root,self.clip_length=Path(root),clip_length
        if clip_length < 2: raise ValueError("clip_length must be at least two")
        self.rows=_records(root,manifest,"clip",class_names)
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]; paths=row["frames"]
        selected=[paths[min(int(i*len(paths)/self.clip_length),len(paths)-1)] for i in range(self.clip_length)]
        return torch.stack([load_rgb(self.root/path) for path in selected]),int(row["label"])
