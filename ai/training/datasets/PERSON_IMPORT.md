# Local person dataset import

`import_person_dataset` imports already-downloaded **COCO object-detection JSON** and its original image files. It does not access the network, create labels, run a detector, or train a model. The currently implemented format is deliberately explicit; other annotation formats are rejected until a separate parser is added and tested.

## Expected source layout

For official source splits:

```text
source_dataset/
  images/
    train/...
    val/...
    test/...
  annotations/
    instances_train.json
    instances_val.json
    instances_test.json
```

COCO image `file_name` paths must resolve within the input directory (either directly or below `images/`). JPEG and PNG are accepted. Each annotation document must contain `images`, `categories`, and `annotations`; a category named `person`; image IDs and file names; and person bounding boxes in COCO `[x, y, width, height]` pixel coordinates. A person annotation is converted to `[x1, y1, x2, y2]`. All person boxes are retained, including multiple people in one image. Other categories are discarded. A source with no `person` category is rejected rather than treated as a set of negative images. Person crowd annotations are rejected for explicit source review.

Example import with official source splits:

```bash
python3 -m ai.training.datasets.import_person_dataset \
  --input /path/to/source_dataset \
  --output data/person/online/source_name \
  --source-name source_name \
  --source-url https://dataset.example.org/ \
  --source-version v1 \
  --download-date 2026-10-08 \
  --license 'verified license identifier' \
  --annotation-license 'verified annotation license identifier' \
  --train annotations/instances_train.json \
  --val annotations/instances_val.json \
  --test annotations/instances_test.json
```

The image/source license, annotation license, URL, and download date must come from the dataset's actual documentation/download record. The importer requires them and never guesses them. Do not use a placeholder license. If the same verified license covers both images and annotations, pass it to both license options. If a COCO document has an image-level license registry, the importer preserves each referenced license name/URL and the image's source URL; unresolved license IDs stop the import. Otherwise the supplied `--license` is copied to every source record. Version is optional when the source does not provide one.

For a single combined COCO file, pass `--annotations annotations/instances.json`. Images are assigned to train/validation/test in a deterministic 70/15/15 group split. COCO `video_id`, `sequence_id`, or `source_group` values define groups; otherwise each image ID is its own group. Official split files are kept in their original splits. Group IDs are namespaced with the source name and output split so the existing leakage validator can detect accidental cross-split reuse.

## Output and review

Write imports to a separate root such as `data/person/online/source_name`; do not point it at `data/person`, whose `images/` folder contains webcam captures. Output includes `images/`, `train.jsonl`, `val.jsonl`, `test.jsonl`, and `source_manifest.json`. Each record contains the original source name/version/URL/license, source ID and source group, `reviewed: false`, and only class `0` person boxes. Imported annotations remain pending human review. Webcam images are neither read nor changed by this importer.

The importer checks image readability and box bounds, refuses existing split/provenance manifests, uses unique names without overwriting, and reports exact/near duplicates and skipped files. Exact duplicates use SHA-256; near duplicates use 64-bit dHash Hamming distance (default threshold 4). The source manifest records both supplied license declarations, per-image licenses, download date, counts, format, split counts, duplicate counts, and an explicit unreviewed status. Each record preserves its image license, annotation license, source identifier, and available image attribution URL. Import validation checks every record's provenance, image and person boxes before success.

No public dataset has been selected or imported as part of this implementation. MS COCO is a format-compatible candidate, not an automatically approved source: its official terms state that annotations are CC BY 4.0, while image copyrights remain with image owners and image use must follow Flickr terms. Verify the image rights and attribution per image before importing. The `--annotation-license` and image-level `--license`/COCO license catalog are kept distinct for this reason.

After a licensed source is selected and imported, inspect it separately with:

```bash
python3 -m ai.training.datasets.validate --data data/person/online/source_name --task person
python3 -m ai.training.datasets.stats --data data/person/online/source_name --task person
```

Do not consider the dataset ready for training until imported annotations and webcam annotations have both been reviewed and the final train/validation/test dataset passes validation.
