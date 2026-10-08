"""Import local COCO detection data into an isolated custom person dataset.

This tool never downloads data or creates annotations. Source files and their
human-authored COCO annotations must already be present locally.
"""
import argparse
from collections import Counter
from datetime import date
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import sys

from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLIT_NAMES = ("train", "val", "test")
PROVENANCE_FIELDS = ("dataset_name", "dataset_url", "license", "annotation_license", "download_date",
                     "images_imported", "person_annotations", "annotation_format")
FORMAT_IMPORTERS = {"coco": "COCO object detection JSON"}


def _safe_filename(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("COCO image file_name must be a non-empty path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Unsafe image path in COCO annotation: {value!r}")
    return path


def _image_source(input_root, file_name):
    relative = _safe_filename(file_name)
    if relative.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image extension for {file_name!r}; supported: JPEG and PNG")
    candidates = (input_root / Path(*relative.parts),
                 input_root / "images" / Path(*relative.parts))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(input_root) and resolved.is_file():
            return resolved
    raise FileNotFoundError(f"COCO image not found under input directory: {file_name}")


def _read_coco(path):
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read COCO annotation file {path}: {exc}") from exc
    if not isinstance(document, dict) or not all(
            isinstance(document.get(key), list) for key in ("images", "categories", "annotations")):
        raise ValueError(f"Unsupported annotation format in {path}: expected COCO JSON with images, categories, and annotations arrays")
    categories = {}
    for category in document["categories"]:
        if (not isinstance(category, dict) or not isinstance(category.get("id"), (int, str))
                or isinstance(category.get("id"), bool) or not isinstance(category.get("name"), str)):
            raise ValueError(f"Malformed COCO category in {path}")
        if category["id"] in categories:
            raise ValueError(f"Duplicate COCO category ID {category['id']!r} in {path}")
        categories[category["id"]] = category["name"].strip().casefold()
    person_ids = {category_id for category_id, name in categories.items() if name == "person"}
    if not person_ids:
        raise ValueError(f"No source category named 'person' in {path}; refusing to invent person labels")

    licenses = {}
    license_rows = document.get("licenses", [])
    if not isinstance(license_rows, list):
        raise ValueError(f"Malformed COCO licenses array in {path}")
    for license_row in license_rows:
        if (not isinstance(license_row, dict) or not isinstance(license_row.get("id"), (int, str))
                or isinstance(license_row.get("id"), bool)):
            raise ValueError(f"Malformed COCO license entry in {path}")
        if license_row["id"] in licenses:
            raise ValueError(f"Duplicate COCO license ID {license_row['id']!r} in {path}")
        licenses[license_row["id"]] = license_row

    images = {}
    for image in document["images"]:
        if (not isinstance(image, dict) or not isinstance(image.get("id"), (int, str))
                or isinstance(image.get("id"), bool) or "file_name" not in image):
            raise ValueError(f"Malformed COCO image record in {path}")
        if image["id"] in images:
            raise ValueError(f"Duplicate COCO image ID {image['id']!r} in {path}")
        image_license = image.get("license")
        if image_license is not None and not isinstance(image_license, str):
            if image_license not in licenses:
                raise ValueError(f"Image {image['id']!r} references unknown COCO license ID {image_license!r}")
            image_license = licenses[image_license]
        images[image["id"]] = {"source": image, "objects": [], "non_person_annotations": 0,
                                "source_license": image_license}

    for annotation in document["annotations"]:
        if (not isinstance(annotation, dict)
                or not isinstance(annotation.get("image_id"), (int, str))
                or isinstance(annotation.get("image_id"), bool)
                or not isinstance(annotation.get("category_id"), (int, str))
                or isinstance(annotation.get("category_id"), bool)):
            raise ValueError(f"Malformed COCO annotation record in {path}")
        image_id = annotation["image_id"]
        if image_id not in images:
            raise ValueError(f"COCO annotation references unknown image_id {image_id!r} in {path}")
        category_id = annotation["category_id"]
        if category_id not in categories:
            raise ValueError(f"COCO annotation references unknown category_id {category_id!r} in {path}")
        if category_id not in person_ids:
            images[image_id]["non_person_annotations"] += 1
            continue
        if annotation.get("iscrowd", 0):
            raise ValueError(f"Person crowd annotation in {path} (image_id={image_id!r}); review or exclude crowd examples before import")
        box = annotation.get("bbox")
        if not isinstance(box, list) or len(box) != 4 or any(isinstance(value, bool) for value in box):
            raise ValueError(f"Invalid COCO XYWH person bbox: {box!r}")
        try:
            x, y, width, height = map(float, box)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid COCO XYWH person bbox: {box!r}") from exc
        if not all(math.isfinite(value) for value in (x, y, width, height)) or width <= 0 or height <= 0:
            raise ValueError(f"Invalid COCO XYWH person bbox: {box!r}")
        images[image_id]["objects"].append({"xywh": [x, y, width, height],
                                             "annotation_id": annotation.get("id")})
    return images


def _annotation_files(input_root, combined=None, split_annotations=None):
    if combined is not None and split_annotations:
        raise ValueError("Use --annotations for a combined COCO file or --train/--val/--test for official splits, not both")
    if combined is not None:
        return None, Path(combined)
    supplied = {name: Path(path) for name, path in (split_annotations or {}).items() if path is not None}
    if supplied:
        if set(supplied) != set(SPLIT_NAMES):
            raise ValueError("Official split import requires all three annotation files: --train, --val, and --test")
        return supplied, None

    candidates = {
        "train": ("annotations/instances_train.json", "annotations/train.json", "train.json"),
        "val": ("annotations/instances_val.json", "annotations/val.json", "val.json"),
        "test": ("annotations/instances_test.json", "annotations/test.json", "test.json"),
    }
    discovered = {}
    for split, values in candidates.items():
        for value in values:
            path = input_root / value
            if path.is_file():
                discovered[split] = path
                break
    if discovered:
        if set(discovered) != set(SPLIT_NAMES):
            raise ValueError("Found an incomplete official split set; provide all --train/--val/--test files")
        return discovered, None
    for value in ("annotations/instances.json", "annotations.json"):
        path = input_root / value
        if path.is_file():
            return None, path
    raise FileNotFoundError("No COCO annotations found. Provide --annotations or all three --train, --val, and --test files.")


def _group_key(image_id, image):
    raw = image.get("source_group", image.get("video_id", image.get("sequence_id", f"image-{image_id}")))
    if isinstance(raw, (dict, list)) or raw is None:
        raise ValueError(f"Invalid source group for image {image_id!r}")
    return str(raw)


def _grouped_split(groups, source_name):
    """Deterministically assign whole groups toward a 70/15/15 image ratio."""
    totals = {"train": 0, "val": 0, "test": 0}
    desired = {"train": .70, "val": .15, "test": .15}
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]),
                    hashlib.sha256(f"{source_name}\0{item[0]}".encode()).hexdigest()))
    assigned = {}
    all_images = sum(len(rows) for _, rows in ordered)
    for group, rows in ordered:
        split = min(SPLIT_NAMES, key=lambda name: (
            ((totals[name] + len(rows)) / max(1, all_images) - desired[name]) ** 2
            - (totals[name] / max(1, all_images) - desired[name]) ** 2,
            SPLIT_NAMES.index(name)))
        assigned[group] = split
        totals[split] += len(rows)
    return assigned


def _dhash(image):
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.tobytes())
    result = 0
    for y in range(8):
        for x in range(8):
            result = (result << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return result


def _image_info(path):
    if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image extension: {path.suffix}")
    digest_builder = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        with Image.open(path) as image:
            visual_hash = _dhash(image)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError(f"Corrupted or unreadable image {path}: {exc}") from exc
    if width < 1 or height < 1:
        raise ValueError(f"Invalid image dimensions for {path}")
    return digest, visual_hash, width, height


class _DuplicateIndex:
    """Exact SHA-256 and banded dHash lookup without quadratic comparisons."""
    def __init__(self, threshold):
        self.threshold = threshold
        self.exact = {}
        self.bands = {}

    def _band_keys(self, value):
        # threshold+1 bands guarantee that images within the dHash distance
        # share at least one unchanged band (multi-index hashing).
        band_count = self.threshold + 1
        return tuple((value >> start) & ((1 << width) - 1)
                     for i in range(band_count)
                     for start, width in [(i * 64 // band_count,
                                           (i + 1) * 64 // band_count - i * 64 // band_count)])

    def add(self, digest, visual_hash, path):
        self.exact.setdefault(digest, path)
        for key in self._band_keys(visual_hash):
            self.bands.setdefault(key, []).append((visual_hash, path))

    def find(self, digest, visual_hash):
        if digest in self.exact:
            return "exact", self.exact[digest]
        candidates = {}
        for key in self._band_keys(visual_hash):
            for old_hash, old_path in self.bands.get(key, ()):
                candidates[old_path] = old_hash
        for old_path, old_hash in candidates.items():
            if (visual_hash ^ old_hash).bit_count() <= self.threshold:
                return "near", old_path
        return None, None


def _existing_images(images_dir, duplicate_index):
    if not images_dir.is_dir():
        return
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES or not path.is_file():
            continue
        digest, visual_hash, _, _ = _image_info(path)
        duplicate_index.add(digest, visual_hash, path.name)


def _write_json_exclusive(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def validate_import(output_root):
    """Check importer provenance, split records, images, and boxes before success."""
    output_root = Path(output_root)
    manifest_path = output_root / "source_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing provenance file: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed provenance file: {manifest_path}: {exc}") from exc
    missing = [field for field in PROVENANCE_FIELDS if field not in manifest or manifest[field] in (None, "")]
    if missing:
        raise ValueError(f"Provenance missing required fields: {', '.join(missing)}")
    image_licenses = manifest.get("image_licenses_used")
    if not isinstance(image_licenses, list):
        raise ValueError("Provenance missing image_licenses_used list")
    total = 0
    for split in SPLIT_NAMES:
        path = output_root / f"{split}.jsonl"
        if not path.is_file():
            raise ValueError(f"Missing split manifest: {path}")
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no}: malformed JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{line_no}: record must be a JSON object")
            for field in ("image", "objects", "source_name", "source_url", "license",
                          "annotation_license", "source_id", "source_group"):
                if field not in row or row[field] in (None, ""):
                    raise ValueError(f"{path.name}:{line_no}: missing provenance/annotation field {field}")
            if not isinstance(row["image"], str):
                raise ValueError(f"{path.name}:{line_no}: image path must be a string")
            image_path = (output_root / row["image"]).resolve()
            if not image_path.is_relative_to(output_root.resolve()) or not image_path.is_file():
                raise ValueError(f"{path.name}:{line_no}: missing or unsafe image {row['image']!r}")
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    image.verify()
            except (OSError, UnidentifiedImageError) as exc:
                raise ValueError(f"{path.name}:{line_no}: unreadable image {row['image']!r}: {exc}") from exc
            if not isinstance(row["objects"], list):
                raise ValueError(f"{path.name}:{line_no}: objects must be a list")
            for obj in row["objects"]:
                if not isinstance(obj, dict) or isinstance(obj.get("class"), bool) or obj.get("class") != 0:
                    raise ValueError(f"{path.name}:{line_no}: imported person class must be 0")
                box = obj.get("bbox")
                if not isinstance(box, list) or len(box) != 4:
                    raise ValueError(f"{path.name}:{line_no}: malformed person bbox")
                if any(isinstance(value, bool) for value in box):
                    raise ValueError(f"{path.name}:{line_no}: boolean person bbox coordinate")
                try:
                    x1, y1, x2, y2 = map(float, box)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path.name}:{line_no}: malformed person bbox {box!r}") from exc
                if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"{path.name}:{line_no}: invalid person bbox {box!r} for {width}x{height}")
            if row.get("reviewed") is not False:
                raise ValueError(f"{path.name}:{line_no}: imported annotations must remain reviewed=false")
            if (row["source_name"] != manifest["dataset_name"]
                    or row["source_url"] != manifest["dataset_url"]
                    or row["annotation_license"] != manifest["annotation_license"]
                    or row["license"] not in image_licenses):
                raise ValueError(f"{path.name}:{line_no}: record provenance does not match source_manifest.json")
            if not row["source_group"].startswith(f"{manifest['dataset_name']}:"):
                raise ValueError(f"{path.name}:{line_no}: source_group is not namespaced to the dataset")
            total += 1
    if total != manifest["images_imported"]:
        raise ValueError(f"Provenance image count {manifest['images_imported']} does not match split records {total}")
    return {"images": total, "person_annotations": manifest["person_annotations"]}


def import_dataset(input_dir, output_dir, source_name, source_url, license_name,
                   annotation_license,
                   source_version=None, combined_annotations=None,
                   split_annotations=None, annotation_format="coco", near_threshold=4,
                   download_date=None):
    if annotation_format not in FORMAT_IMPORTERS:
        raise ValueError(f"Unsupported annotation format {annotation_format!r}; supported formats: {', '.join(FORMAT_IMPORTERS)}")
    if not source_name or not source_name.strip():
        raise ValueError("--source-name is required")
    if not source_url or not source_url.strip():
        raise ValueError("--source-url is required for provenance")
    if not license_name or not license_name.strip() or license_name.strip().casefold() in {"unknown", "unspecified", "n/a"}:
        raise ValueError("A verified source license is required; no license value will be inferred")
    if not annotation_license or not annotation_license.strip() or annotation_license.strip().casefold() in {"unknown", "unspecified", "n/a"}:
        raise ValueError("A verified annotation license is required; no license value will be inferred")
    if not download_date:
        raise ValueError("--download-date is required; the importer will not guess when source files were downloaded")
    try:
        date.fromisoformat(download_date)
    except ValueError as exc:
        raise ValueError("--download-date must be an ISO date in YYYY-MM-DD format") from exc
    if not 0 <= near_threshold <= 16:
        raise ValueError("--near-threshold must be between 0 and 16 dHash bits")
    source_name = source_name.strip()
    source_url = source_url.strip()
    license_name = license_name.strip()
    annotation_license = annotation_license.strip()
    download_date = download_date.strip()
    source_version = source_version.strip() if isinstance(source_version, str) and source_version.strip() else None

    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input dataset directory not found: {input_root}")
    if output_root == input_root or output_root.is_relative_to(input_root):
        raise ValueError("Output must be outside the source directory so source files remain untouched")
    if any((output_root / filename).exists() for filename in
           ("source_manifest.json", "train.jsonl", "val.jsonl", "test.jsonl")):
        raise FileExistsError(f"Output already contains provenance or split manifests: {output_root}")

    official_files, combined_file = _annotation_files(input_root, combined_annotations, split_annotations)
    source_documents = {}
    if official_files:
        for split, path in official_files.items():
            source_documents[split] = _read_coco(path if path.is_absolute() else input_root / path)
    else:
        source_documents["combined"] = _read_coco(combined_file if combined_file.is_absolute() else input_root / combined_file)

    groups = {}
    for original_split, image_map in source_documents.items():
        for image_id, item in image_map.items():
            key = _group_key(image_id, item["source"])
            if original_split != "combined":
                key = f"{original_split}:{key}"
            groups.setdefault(key, []).append((original_split, image_id, item))
    assigned_groups = _grouped_split(groups, source_name) if "combined" in source_documents else None

    output_images = output_root / "images"
    duplicate_index = _DuplicateIndex(near_threshold)
    _existing_images(output_images, duplicate_index)
    next_number = 1
    existing_names = [path.name for path in output_images.iterdir()] if output_images.is_dir() else []
    for name in existing_names:
        match = re.fullmatch(r"online_(\d{6})\.(?:jpg|jpeg|png)", name, re.IGNORECASE)
        if match:
            next_number = max(next_number, int(match.group(1)) + 1)

    records = {split: [] for split in SPLIT_NAMES}
    skipped = []
    duplicate_counts = Counter()
    source_group_names = {}
    annotation_count = 0
    non_person_annotation_count = sum(item["non_person_annotations"]
                                      for image_map in source_documents.values()
                                      for item in image_map.values())

    if assigned_groups is None:
        split_order = {name: index for index, name in enumerate(SPLIT_NAMES)}
        group_order = sorted(groups, key=lambda key: (split_order[groups[key][0][0]], key))
    else:
        group_order = sorted(groups)
    for group_key in group_order:
        values = groups[group_key]
        if assigned_groups is not None:
            target_split = assigned_groups[group_key]
        else:
            # Official source splits stay official; namespace groups by split so
            # validation can detect any source group accidentally crossing splits.
            target_split = values[0][0]
        source_group_names[group_key] = f"{source_name}:{target_split}:{group_key}"
        for original_split, image_id, item in values:
            image = item["source"]
            source_path = _image_source(input_root, image["file_name"])
            digest, visual_hash, width, height = _image_info(source_path)
            duplicate_type, duplicate_path = duplicate_index.find(digest, visual_hash)
            if duplicate_type:
                duplicate_counts[duplicate_type] += 1
                skipped.append({"source_id": str(image_id), "file": image["file_name"],
                                "reason": f"{duplicate_type}_duplicate", "matches": duplicate_path})
                continue

            objects = []
            for source_object in item["objects"]:
                x, y, box_width, box_height = source_object["xywh"]
                x2, y2 = x + box_width, y + box_height
                if not (0 <= x < x2 <= width and 0 <= y < y2 <= height):
                    raise ValueError(f"Person bbox outside image bounds for {image['file_name']!r}: {[x, y, x2, y2]} in {width}x{height}")
                objects.append({"class": 0, "bbox": [x, y, x2, y2]})

            output_name = f"online_{next_number:06d}{source_path.suffix.lower()}"
            next_number += 1
            source_license = item["source_license"]
            if isinstance(source_license, dict):
                image_license_name = source_license.get("name")
                if not isinstance(image_license_name, str) or not image_license_name.strip():
                    raise ValueError(f"COCO license entry for image {image_id!r} has no license name")
                image_license_name = image_license_name.strip()
                source_license_url = source_license.get("url")
                source_license_id = source_license.get("id")
            else:
                image_license_name = source_license.strip() if isinstance(source_license, str) and source_license.strip() else license_name
                source_license_url = None
                source_license_id = None
            row = {"image": f"images/{output_name}", "objects": objects,
                   "reviewed": False, "source_name": source_name,
                   "source_version": source_version, "source_url": source_url,
                   "source_id": str(image_id), "license": image_license_name,
                   "annotation_license": annotation_license,
                   "source_license_url": source_license_url,
                   "source_license_id": source_license_id,
                   "source_image_url": image.get("coco_url") or image.get("flickr_url"),
                   "source_group": source_group_names[group_key]}
            records[target_split].append((row, source_path, digest, visual_hash))
            duplicate_index.add(digest, visual_hash, output_name)
            annotation_count += len(objects)

    output_images.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        for row, source_path, _, _ in records[split]:
            target = output_root / row["image"]
            with target.open("xb") as stream:
                with source_path.open("rb") as source:
                    shutil.copyfileobj(source, stream)
        manifest_path = output_root / f"{split}.jsonl"
        with manifest_path.open("x", encoding="utf-8") as stream:
            for row, _, _, _ in records[split]:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    summary = {"dataset_name": source_name, "dataset_url": source_url,
               "version": source_version, "license": license_name,
               "annotation_license": annotation_license,
               "download_date": download_date,
               "images_imported": sum(len(values) for values in records.values()),
               "person_annotations": annotation_count,
               "non_person_annotations_discarded": non_person_annotation_count,
               "annotation_format": annotation_format,
               "image_licenses_used": sorted({row["license"] for split_rows in records.values()
                                               for row, _, _, _ in split_rows}),
               "split_counts": {split: len(records[split]) for split in SPLIT_NAMES},
               "source_groups": len(set(source_group_names.values())),
               "exact_duplicates": duplicate_counts["exact"],
               "near_duplicates": duplicate_counts["near"],
               "skipped_files": skipped,
               "review_status": "unreviewed; human review required"}
    _write_json_exclusive(output_root / "source_manifest.json", summary)
    validate_import(output_root)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Import local COCO person annotations; performs no downloads or automatic labeling.")
    parser.add_argument("--input", required=True, help="Downloaded dataset root directory")
    parser.add_argument("--output", required=True, help="Separate destination dataset root (recommended: data/person/online/<name>)")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-url", required=True, help="Original dataset landing page or source URL")
    parser.add_argument("--source-version", help="Version/release, if documented")
    parser.add_argument("--download-date", required=True, help="Verified source download date (YYYY-MM-DD)")
    parser.add_argument("--license", required=True, dest="license_name",
                        help="Verified image/source license; per-image COCO license entries take precedence")
    parser.add_argument("--annotation-license", required=True,
                        help="Verified license for the source annotations")
    parser.add_argument("--format", choices=tuple(FORMAT_IMPORTERS), default="coco")
    parser.add_argument("--annotations", help="Single combined COCO JSON; groups are split deterministically 70/15/15")
    parser.add_argument("--train", help="Official COCO train annotation JSON (use with --val and --test)")
    parser.add_argument("--val", help="Official COCO validation annotation JSON")
    parser.add_argument("--test", help="Official COCO test annotation JSON")
    parser.add_argument("--near-threshold", type=int, default=4, help="dHash Hamming distance for near duplicates (default: 4)")
    args = parser.parse_args(argv)
    split_annotations = {name: getattr(args, name) for name in SPLIT_NAMES}
    try:
        summary = import_dataset(args.input, args.output, args.source_name,
                                 args.source_url, args.license_name, args.annotation_license,
                                 args.source_version, args.annotations,
                                 split_annotations, args.format, args.near_threshold,
                                 args.download_date)
    except (ValueError, FileNotFoundError, FileExistsError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    print("Imported annotations remain reviewed=false; human review is required before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
