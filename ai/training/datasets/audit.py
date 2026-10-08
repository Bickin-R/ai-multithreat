"""Non-throwing dataset audit with per-sample diagnostics and statistics."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from PIL import Image, UnidentifiedImageError

TASKS = {
    "person": ("detection", ["person"]),
    "weapon": ("detection", ["knife", "gun", "arivaal", "baseball_bat", "other_weapon"]),
    "fire": ("classification", ["normal", "fire"]),
    "violence": ("clip", ["normal", "violence"]),
}
SPLITS = (("train", "train.jsonl"), ("validation", "val.jsonl"), ("test", "test.jsonl"))
FRAME_NUMBER = re.compile(r"^(.*?)(\d+)(\D*)$")


def _content_hash(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def _new_split():
    return {"samples": 0, "valid_samples": 0, "invalid_samples": 0,
            "class_distribution": Counter(), "errors": [], "missing_files": 0,
            "class_images": Counter(), "positive_images": 0, "negative_images": 0, "reviewed_images": 0,
            "invalid_annotations": 0, "invalid_bounding_boxes": 0,
            "empty_annotations": 0, "duplicate_samples": 0,
            "objects": 0, "objects_per_sample": [], "box_widths": [], "box_heights": [],
            "clip_frame_counts": [], "ordering_problems": 0}


def _error(report, message, category="invalid_annotations"):
    report["errors"].append(message)
    if category in report:
        report[category] += 1


def _read_ref(root, ref, report, location):
    if not isinstance(ref, str) or not ref.strip():
        _error(report, f"{location}: missing/invalid path", "missing_files")
        return None
    path = (root / ref).resolve()
    if not path.is_relative_to(root):
        _error(report, f"{location}: path escapes dataset root: {ref}")
        return None
    if not path.is_file():
        _error(report, f"{location}: missing file: {ref}", "missing_files")
        return None
    try:
        with Image.open(path) as im:
            width, height = im.size
            im.verify()
        if width <= 0 or height <= 0:
            raise ValueError("image has zero dimensions")
        return path, width, height
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        _error(report, f"{location}: unreadable image {ref}: {exc}", "missing_files")
        return None


def _line_records(path, report):
    if not path.is_file():
        _error(report, f"Manifest missing: {path}")
        return []
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        report["samples"] += 1
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("record must be a JSON object")
            records.append((line_no, row))
        except (json.JSONDecodeError, ValueError) as exc:
            _error(report, f"{path.name}:{line_no}: malformed record: {exc}")
            report["invalid_samples"] += 1
    if not records and report["samples"] == 0:
        report["errors"].append(f"Manifest is empty: {path.name}")
    return records


def _check_frame_order(refs):
    parsed = []
    for ref in refs:
        match = FRAME_NUMBER.match(Path(ref).name)
        if not match:
            return None
        parsed.append((match.group(1), int(match.group(2)), match.group(3)))
    if len({(prefix, suffix) for prefix, _, suffix in parsed}) != 1:
        return None
    numbers = [number for _, number, _ in parsed]
    return any(a >= b for a, b in zip(numbers, numbers[1:]))


def audit_dataset(data_dir, task=None):
    """Audit all three split manifests. Never downloads or modifies source data."""
    root = Path(data_dir).resolve()
    task = task or root.name
    if task not in TASKS:
        raise ValueError(f"Cannot infer task from {root.name!r}; choose one of {', '.join(TASKS)}")
    kind, classes = TASKS[task]
    result = {"task": task, "kind": kind, "classes": classes, "root": str(root),
              "splits": {}, "leakage": [], "totals": {}}
    seen_within = defaultdict(dict)
    seen_hashes = defaultdict(dict)
    sample_metadata = []

    for split, filename in SPLITS:
        report = _new_split()
        manifest = root / filename
        parsed_rows = _line_records(manifest, report)
        for line_no, row in parsed_rows:
            location = f"{filename}:{line_no}"
            errors_before = len(report["errors"])
            refs = row.get("frames") if kind == "clip" else [row.get("image")]
            if kind == "clip":
                if not isinstance(refs, list) or len(refs) < 2:
                    _error(report, f"{location}: frames must contain at least two paths")
                    refs = []
                else:
                    report["clip_frame_counts"].append(len(refs))
                    if len(set(refs)) != len(refs):
                        report["ordering_problems"] += 1
                        _error(report, f"{location}: repeated frame path in clip")
                    if _check_frame_order(refs):
                        report["ordering_problems"] += 1
                        _error(report, f"{location}: numbered frame filenames are not in ascending order")

            image_info = []
            for ref in refs:
                info = _read_ref(root, ref, report, location)
                if info:
                    path, width, height = info
                    rel = path.relative_to(root).as_posix()
                    image_info.append((rel, width, height))
            content_hashes=[]
            for rel, _, _ in image_info:
                digest=_content_hash(root/rel)
                content_hashes.append(digest)
                if digest in seen_hashes[split] and seen_hashes[split][digest][0] != rel:
                    report["duplicate_samples"] += 1
                    _error(report, f"{location}: duplicate image content also appears at {seen_hashes[split][digest][1]}")
                else:
                    seen_hashes[split][digest]=(rel,location)
            if len(image_info) != len(refs):
                # Missing/bad references have already been described above.
                pass

            if kind == "detection":
                if task == "weapon":
                    if row.get("reviewed") is True:
                        report["reviewed_images"] += 1
                    else:
                        _error(report, f"{location}: weapon record must include reviewed=true")
                if "objects" not in row or not isinstance(row["objects"], list):
                    _error(report, f"{location}: objects must be a list")
                else:
                    objects = row["objects"]
                    if not objects:
                        report["empty_annotations"] += 1
                    valid_box_count = 0
                    image_classes = set()
                    dims = image_info[0][1:] if image_info else None
                    for obj_idx, obj in enumerate(objects):
                        prefix = f"{location}: object {obj_idx}"
                        if not isinstance(obj, dict):
                            _error(report, f"{prefix}: object must be a JSON object")
                            continue
                        cls = obj.get("class")
                        if isinstance(cls, bool) or not isinstance(cls, int) or cls < 0 or cls >= len(classes):
                            _error(report, f"{prefix}: invalid class ID {cls!r}")
                            continue
                        box = obj.get("bbox")
                        if not isinstance(box, list) or len(box) != 4:
                            _error(report, f"{prefix}: bbox must be [x1,y1,x2,y2]", "invalid_bounding_boxes")
                            continue
                        try:
                            if any(isinstance(v, bool) for v in box): raise ValueError("boolean coordinate")
                            x1,y1,x2,y2 = map(float, box)
                            if not all(math.isfinite(v) for v in (x1,y1,x2,y2)): raise ValueError("non-finite coordinate")
                            if not dims or not (0 <= x1 < x2 <= dims[0] and 0 <= y1 < y2 <= dims[1]):
                                raise ValueError("box is reversed, empty, or outside image bounds")
                        except (TypeError, ValueError) as exc:
                            _error(report, f"{prefix}: invalid bbox {box!r}: {exc}", "invalid_bounding_boxes")
                            continue
                        report["class_distribution"][classes[cls]] += 1
                        image_classes.add(classes[cls])
                        report["box_widths"].append(x2-x1); report["box_heights"].append(y2-y1)
                        valid_box_count += 1
                    report["objects"] += valid_box_count
                    for name in image_classes:
                        report["class_images"][name] += 1
                    if task == "weapon":
                        if valid_box_count:
                            report["positive_images"] += 1
                        else:
                            report["negative_images"] += 1
                    report["objects_per_sample"].append(valid_box_count)
                    if valid_box_count != len(objects):
                        report["invalid_annotations"] += 1
            else:
                label = row.get("label")
                if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(classes):
                    _error(report, f"{location}: label must be class ID 0..{len(classes)-1}")
                else:
                    report["class_distribution"][classes[label]] += 1

            # A source record is valid only if its format, labels, annotations,
            # and all referenced image files passed the checks above.
            valid = len(report["errors"]) == errors_before
            key = tuple(item[0] for item in image_info) if kind == "clip" else (image_info[0][0],) if image_info else ()
            if key:
                if key in seen_within[split]:
                    report["duplicate_samples"] += 1
                    _error(report, f"{location}: duplicate sample also appears at {seen_within[split][key]}")
                    valid = False
                else:
                    seen_within[split][key] = location
            sample_metadata.append({"split": split, "location": location,
                                    "refs": [item[0] for item in image_info], "hashes": content_hashes,
                                    "group": row.get("source_group"), "valid": valid})
            if valid:
                report["valid_samples"] += 1
            else:
                report["invalid_samples"] += 1
        result["splits"][split] = report

    # Any source image/frame reused between splits makes each affected record leaky.
    owners = defaultdict(list)
    for sample in sample_metadata:
        for ref in set(sample["refs"]): owners[ref].append(sample)
    leaky_refs = {ref: users for ref, users in owners.items()
                  if len({u["split"] for u in users}) > 1}
    for ref, users in leaky_refs.items():
        split_names = sorted({u["split"] for u in users})
        result["leakage"].append({"path": ref, "splits": split_names})
        for user in users:
            report = result["splits"][user["split"]]
            _error(report, f"{user['location']}: leakage, {ref} appears in {', '.join(split_names)}")
            if user["valid"]:
                report["valid_samples"] -= 1
                report["invalid_samples"] += 1
                user["valid"] = False

    hash_owners=defaultdict(list); group_owners=defaultdict(list)
    for sample in sample_metadata:
        for digest in set(sample["hashes"]): hash_owners[digest].append(sample)
        if sample["group"]: group_owners[sample["group"]].append(sample)
    hash_leaks=[]
    for digest, owners in hash_owners.items():
        if len({owner["split"] for owner in owners})>1 and len({ref for owner in owners for ref in owner["refs"]})>1:
            hash_leaks.append((f"sha256:{digest}",owners))
    for key, owners in hash_leaks+[(f"source_group:{k}",v) for k,v in group_owners.items()]:
        if len({owner["split"] for owner in owners})>1:
            split_names=sorted({owner["split"] for owner in owners})
            result["leakage"].append({"path":key,"splits":split_names})
            for owner in owners:
                report=result["splits"][owner["split"]]
                _error(report,f"{owner['location']}: leakage, {key} appears in {', '.join(split_names)}")
                if owner["valid"]:
                    report["valid_samples"]-=1; report["invalid_samples"]+=1; owner["valid"]=False

    for report in result["splits"].values():
        report["class_distribution"] = dict(report["class_distribution"])
        report["class_images"] = dict(report["class_images"])
        report["average_objects_per_sample"] = (sum(report["objects_per_sample"])/len(report["objects_per_sample"])
                                                if report["objects_per_sample"] else 0.0)
        report["objects_per_sample_range"] = {
            "min": min(report["objects_per_sample"], default=None),
            "max": max(report["objects_per_sample"], default=None),
        }
        report["box_size"] = {
            "min_width": min(report["box_widths"], default=None),
            "max_width": max(report["box_widths"], default=None),
            "min_height": min(report["box_heights"], default=None),
            "max_height": max(report["box_heights"], default=None),
        }
        report["frames_per_clip"] = {
            "min": min(report["clip_frame_counts"], default=None),
            "max": max(report["clip_frame_counts"], default=None),
            "average": sum(report["clip_frame_counts"])/len(report["clip_frame_counts"])
                       if report["clip_frame_counts"] else None,
        }
        for key in ("objects_per_sample", "box_widths", "box_heights", "clip_frame_counts"):
            del report[key]
    result["totals"] = {key: sum(result["splits"][s][key] for s, _ in SPLITS)
                         for key in ("samples", "valid_samples", "invalid_samples", "missing_files",
                                     "invalid_annotations", "invalid_bounding_boxes", "empty_annotations",
                                     "duplicate_samples", "ordering_problems", "objects")}
    result["totals"]["class_distribution"] = dict(sum((Counter(result["splits"][s]["class_distribution"])
                                                        for s, _ in SPLITS), Counter()))
    for key in ("positive_images", "negative_images", "reviewed_images"):
        result["totals"][key] = sum(result["splits"][s][key] for s, _ in SPLITS)
    result["totals"]["class_images"] = dict(sum((Counter(result["splits"][s]["class_images"])
                                                  for s, _ in SPLITS), Counter()))
    object_counts=result["totals"]["class_distribution"]
    nonzero=[count for count in object_counts.values() if count>0]
    result["totals"]["class_imbalance"]={
        "criterion":"largest class has more than 2x the objects of the smallest non-empty class",
        "ratio":max(nonzero)/min(nonzero) if len(nonzero)>1 else None,
        "significant":bool(len(nonzero)>1 and max(nonzero)>2*min(nonzero)),
    }
    return result
