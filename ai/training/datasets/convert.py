"""Convert local COCO detection JSON or classification CSV to project JSONL.

No network access or external detector is used. See `--help` for commands.
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path, PurePosixPath
from .audit import TASKS


def _output_class(value, class_names):
    if isinstance(value,bool): raise ValueError("Boolean is not a valid class ID")
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        idx=int(value)
        if 0 <= idx < len(class_names): return idx
    if isinstance(value,str) and value in class_names: return class_names.index(value)
    raise ValueError(f"Unknown output class {value!r}; expected one of {class_names}")


def coco_to_jsonl(source, output, task, image_prefix="images", class_map=None,
                  skip_unmapped=False, exclude_crowd_images=False):
    kind, class_names=TASKS[task]
    if kind!="detection": raise ValueError("COCO conversion is only supported for person/weapon detection")
    if task=="weapon" and Path(output).name in {"train.jsonl","val.jsonl","test.jsonl"}:
        raise ValueError("Weapon COCO output is an unreviewed proposal; write it to a draft JSONL, never a final split")
    document=json.loads(Path(source).read_text(encoding="utf-8"))
    if not all(isinstance(document.get(key),list) for key in ("images","categories","annotations")):
        raise ValueError("COCO input needs images, categories, and annotations arrays")
    category_names={item["id"]:item["name"] for item in document["categories"]}
    mapping={}
    for source_name, target_name in (class_map or {}).items():
        if task == "weapon" and (not isinstance(target_name,str) or target_name not in class_names):
            raise ValueError("Weapon COCO class maps must name a current class explicitly; numeric/legacy IDs are rejected")
        mapping[source_name]=_output_class(target_name,class_names)
    if class_map is None:
        for name in class_names:
            mapping[name]=class_names.index(name)
    images={}
    for item in document["images"]:
        image_id=item["id"]
        if image_id in images: raise ValueError(f"Duplicate COCO image id {image_id}")
        images[image_id]={"image":_prefix_image(item["file_name"],image_prefix),"objects":[]}
    skipped=0; skipped_crowd=0; skipped_classes=Counter(); crowd_image_ids=set()
    for ann in document["annotations"]:
        image_id=ann["image_id"]
        if image_id not in images: raise ValueError(f"COCO annotation references unknown image_id {image_id}")
        if ann.get("iscrowd",0):
            if not exclude_crowd_images:
                raise ValueError("COCO crowd annotation found; review/annotate individuals or explicitly use --exclude-crowd-images")
            skipped_crowd+=1; crowd_image_ids.add(image_id); continue
        category=category_names.get(ann.get("category_id"))
        if category not in mapping:
            if skip_unmapped: skipped+=1; skipped_classes[category or "<missing>"]+=1; continue
            raise ValueError(f"Unmapped COCO category {category!r}; provide --class-map or --skip-unmapped")
        box=ann.get("bbox")
        if not isinstance(box,list) or len(box)!=4: raise ValueError(f"Invalid COCO bbox: {box!r}")
        x,y,w,h=map(float,box)
        if not all(math.isfinite(v) for v in (x,y,w,h)) or w<=0 or h<=0:
            raise ValueError(f"Invalid COCO XYWH box: {box!r}")
        images[image_id]["objects"].append({"class":mapping[category],"bbox":[x,y,x+w,y+h]})
    output_images=[row for image_id,row in images.items() if image_id not in crowd_image_ids]
    if task == "weapon":
        # A converter cannot attest that a person visually reviewed every box.
        for row in output_images: row["reviewed"]=False
    _write_jsonl(output,output_images)
    return {"samples":len(output_images),"objects":sum(len(row["objects"]) for row in output_images),
            "skipped_unmapped_annotations":skipped,"skipped_categories":dict(skipped_classes),
            "excluded_crowd_images":len(crowd_image_ids),"crowd_annotations":skipped_crowd}


def _prefix_image(filename, prefix):
    p=PurePosixPath(filename)
    if p.is_absolute() or ".." in p.parts: raise ValueError(f"Unsafe image path in annotation file: {filename}")
    prefix=PurePosixPath(prefix) if prefix else PurePosixPath()
    if prefix.parts and p.parts[:len(prefix.parts)]==prefix.parts: return p.as_posix()
    return (prefix/p).as_posix()


def _validate_relative_path(value):
    if not isinstance(value,str) or not value.strip(): raise ValueError("Image/frame paths must be non-empty strings")
    path=PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts: raise ValueError(f"Unsafe image/frame path: {value}")
    return path.as_posix()


def csv_to_jsonl(source, output, task):
    kind,class_names=TASKS[task]
    if kind not in ("classification","clip"): raise ValueError("CSV conversion supports fire and violence tasks")
    rows=[]
    with Path(source).open(newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        required={"label","frames" if kind=="clip" else "image"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV needs columns: {', '.join(sorted(required))}")
        for line_no, row in enumerate(reader,2):
            value=row["label"]
            label=class_names.index(value) if value in class_names else _output_class(value,class_names)
            if kind=="clip":
                try: frames=json.loads(row["frames"])
                except json.JSONDecodeError as exc: raise ValueError(f"CSV row {line_no}: frames must be a JSON array") from exc
                if not isinstance(frames,list): raise ValueError(f"CSV row {line_no}: frames must be a JSON array")
                rows.append({"frames":[_validate_relative_path(frame) for frame in frames],"label":label})
            else:
                rows.append({"image":_validate_relative_path(row["image"]),"label":label})
    _write_jsonl(output,rows)
    return {"samples":len(rows)}


def _write_jsonl(path, rows):
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("w",encoding="utf-8") as stream:
        for row in rows: stream.write(json.dumps(row,separators=(",",":"))+"\n")


def main(argv=None):
    parser=argparse.ArgumentParser(description="Convert local COCO JSON or classification CSV to project JSONL. No downloads.")
    parser.add_argument("--format",choices=("coco","csv"),required=True)
    parser.add_argument("--task",choices=tuple(TASKS),required=True)
    parser.add_argument("--input",required=True); parser.add_argument("--output",required=True)
    parser.add_argument("--image-prefix",default="images",help="COCO image path prefix in dataset root")
    parser.add_argument("--class-map",help="JSON file mapping source COCO class names to target names")
    parser.add_argument("--skip-unmapped",action="store_true",help="Explicitly skip unmapped COCO categories")
    parser.add_argument("--exclude-crowd-images",action="store_true",
                        help="Exclude every image containing a COCO crowd annotation (never turn it into a negative)")
    args=parser.parse_args(argv)
    class_map=json.loads(Path(args.class_map).read_text()) if args.class_map else None
    if args.format=="coco": summary=coco_to_jsonl(args.input,args.output,args.task,args.image_prefix,
                                                   class_map,args.skip_unmapped,args.exclude_crowd_images)
    else: summary=csv_to_jsonl(args.input,args.output,args.task)
    print(json.dumps(summary,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
