"""Shared checks for source metadata attached to person detection records."""
import json
from pathlib import Path


RECORD_FIELDS = ("source_name", "source_url", "source_id", "license",
                "annotation_license", "source_group")
MANIFEST_FIELDS = ("dataset_name", "dataset_url", "license", "annotation_license",
                   "download_date", "images_imported", "person_annotations",
                   "annotation_format", "image_licenses_used")
SOURCE_MARKERS = {"source_name", "source_url", "source_id", "license", "annotation_license"}


def read_source_manifest(root):
    """Return (manifest, error); no manifest is fine until imported rows appear."""
    path = Path(root) / "source_manifest.json"
    if not path.exists():
        return None, f"missing provenance file: {path}"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"malformed provenance file {path}: {exc}"
    if not isinstance(manifest, dict):
        return None, f"provenance file must contain a JSON object: {path}"
    missing = [key for key in MANIFEST_FIELDS if key not in manifest or manifest[key] in (None, "")]
    if missing:
        return None, f"provenance missing required fields: {', '.join(missing)}"
    if not isinstance(manifest["image_licenses_used"], list):
        return None, "provenance image_licenses_used must be a list"
    return manifest, None


def record_provenance_errors(row, manifest, manifest_error=None):
    """Return errors for imported person records; local/manual rows are unaffected."""
    if not SOURCE_MARKERS.intersection(row):
        return []
    if manifest is None:
        return [manifest_error or "missing source provenance"]
    missing = [key for key in RECORD_FIELDS if key not in row or row[key] in (None, "")]
    if missing:
        return [f"record provenance missing fields: {', '.join(missing)}"]
    errors = []
    if row["source_name"] != manifest["dataset_name"]:
        errors.append("source_name does not match source_manifest.json")
    if row["source_url"] != manifest["dataset_url"]:
        errors.append("source_url does not match source_manifest.json")
    if row["annotation_license"] != manifest["annotation_license"]:
        errors.append("annotation_license does not match source_manifest.json")
    if row["license"] not in manifest["image_licenses_used"]:
        errors.append("image license is not listed in source_manifest.json")
    if not isinstance(row["source_group"], str) or not row["source_group"].startswith(f"{manifest['dataset_name']}:"):
        errors.append("source_group is not namespaced to the source dataset")
    return errors
