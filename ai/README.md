# Custom AI surveillance pipeline

This project defines its own compact PyTorch models and does not use YOLO or
Ultralytics. Models initialize from random weights. Training is a separate
command and never runs when the camera pipeline starts. **Model architecture
implemented, but trained inference has not yet been validated.** No dataset or
trained checkpoints are currently included, so pipeline model statuses are
unavailable until you train and supply weights.

## Architecture and status

The person and weapon models use a compact stride-16 CNN with a grid prediction
head (objectness, normalized box, and class logits). Person has one class;
weapon classes are `knife`, `gun`, `arivaal`, `baseball_bat`, and `other_weapon`. Fire is a custom binary
image classifier with `normal` and `fire`. Violence is a custom binary clip
classifier with `normal` and `violence`; it extracts per-frame features and
passes the ordered feature sequence through a GRU. These are starter trainable architectures, not
validated detectors/classifiers. Detection training currently assigns one
object per grid cell and rejects annotations that collide in a cell. Accuracy,
speed, and thresholds require evaluation and tuning on representative data.
For a 320x320 input, the weapon output is `[B,10,20,20]`: objectness, four box
channels, and five class logits. Inference returns `class_id`, `class_name`,
`bbox`, and `confidence`. Existing three-class weapon checkpoints are rejected;
their class IDs are never remapped automatically.

## Dataset preparation

The empty workspace is prepared for real, user-provided data:

```text
data/
├── person/{images/,train.jsonl,val.jsonl,test.jsonl}
├── weapon/{images/,train.jsonl,val.jsonl,test.jsonl}
├── fire/{images/,train.jsonl,val.jsonl,test.jsonl}
└── violence/{frames/,train.jsonl,val.jsonl,test.jsonl}
```

Put JPEG or PNG files under the task's `images/` folder; violence uses
`frames/`. Use stable descriptive names such as
`camera03_scene12_frame000145.jpg` and
`scene12_clip04_frame000145.jpg`. Keep original source/clip identifiers in
filenames so you can assign entire sessions to one split. Do not overwrite a
source image with an augmented copy. Training preprocessing resizes RGB images
to 320x320 and scales pixel values to [0,1]; there is no automatic augmentation.
This workflow never scrapes, downloads, or trains on data automatically.

Create three non-empty JSON Lines manifests per task: `train.jsonl`, `val.jsonl`,
and `test.jsonl`. Each line is one JSON object. Use a recommended starting split
of 70% train, 15% validation, and 15% test. Split by original video, person/session, and camera scene before
extracting frames. Never put adjacent frames or clips from one source in
different splits; the validator can catch repeated file paths, but cannot
recognize two separately copied files from the same scene.

### Person and weapon detection

Detection layout:

```text
data/person/
  images/0001.jpg
  train.jsonl
  val.jsonl
  test.jsonl
```

Each manifest line has an image path relative to the dataset root and pixel
XYXY boxes in the original image. Person class is always 0. Weapon class IDs
are 0=`knife`, 1=`gun`, 2=`arivaal`, 3=`baseball_bat`, and
4=`other_weapon` (`arivaal` is the traditional Tamil sickle-like weapon).
Every final weapon row must include `"reviewed":true`; an
empty object list then means that a human reviewed the image and found no
target object. The dataset loader rejects
missing files, malformed rows, out-of-range class IDs, invalid/out-of-image
boxes, and split path overlap.

Use tight pixel boxes in `[x1,y1,x2,y2]` order, measured on the original image:
top-left first, bottom-right second, with `x2>x1` and `y2>y1`. Include the
visible person or weapon when partly occluded; follow one consistent policy for
truncation and difficult visibility. Person has only class ID `0=person`.
Weapon IDs are `0=knife`, `1=gun`, `2=arivaal`, `3=baseball_bat`,
`4=other_weapon`. Do not map unknown categories automatically. A reviewed
image with no target uses `"objects": []` and `"reviewed":true`.

Representative records (one JSON object per line):

```json
{"image":"images/person_001.jpg","objects":[{"class":0,"bbox":[42,30,190,286]}]}
{"image":"images/person_empty_001.jpg","objects":[]}
{"image":"images/weapon_001.jpg","objects":[{"class":2,"bbox":[122,91,208,173]}],"reviewed":true,"source_group":"session_01"}
{"image":"images/fire_001.jpg","label":1}
{"frames":["frames/clip001_frame000100.jpg","frames/clip001_frame000140.jpg","frames/clip001_frame000180.jpg"],"label":1}
```

The first two records are person samples, the third a weapon sample, the fourth
a fire sample (`label` 0=`normal`, 1=`fire`), and the fifth a violence clip
(`label` 0=`normal`, 1=`violence`). Clip paths must be ordered oldest to newest
and should come from one source video. Violence labels describe visible
physical violence only, not intent or a prediction. Use JPEG/PNG frames and
include varied non-violent clips as negatives.

### Fire classification

Each image is labeled as a whole: `{"image":"images/0001.jpg","label":0}`
where 0=`normal` and 1=`fire`. Use `normal` examples with difficult negatives
such as sunsets, lamps, orange clothing, and reflections, along with varied
fire conditions and camera distances. Keep images from one burst/video in one
split.

### Violence clip classification

Extract ordered frames from video into `frames/`. Each line references frames
from one clip and labels 0=`normal`, 1=`violence`:

```json
{"frames":["frames/clip1_000.jpg","frames/clip1_001.jpg"],"label":1}
```

The loader samples eight frames across each clip. Choose representative
timestamps across a clip and keep filenames numerically ordered. Keep all clips
from a source recording in one split. Label observable physical violence only;
labels must not encode intent or predicted future behavior.

### Suggested first collection targets

These are practical starting targets for a prototype, not accuracy guarantees:

- Person: 300–800 varied labeled frames from multiple rooms, distances,
  lighting conditions, and source sessions; include reviewed person-free frames.
- Weapon: collect varied examples across all five classes (`knife`, `gun`,
  `arivaal`, `baseball_bat`, `other_weapon`) plus verified negatives.
  If a class is unavailable, leave it out of training plans rather than
  relabeling unrelated objects.
- Fire: 300–1,000 images per class, with diverse fire examples and hard normal
  negatives.
- Violence: 100–300 clips per class to begin, from varied source sessions,
  with ordered frames and clear normal clips as well as clips showing visible
  physical violence.

Start smaller if needed, inspect validation errors, then expand the weak
classes/scenes. Dataset size alone does not establish accuracy.

### Local person annotation and conversion

For direct person annotation, put real images in `data/person/images/` and run:

```bash
python3 -m ai.training.datasets.annotate_person --data data/person --split train
```

The local Tk window displays each image at a fitted scale while converting drawn
coordinates back to original-image pixels. Drag around every visible person;
use **Clear boxes** to redraw. If an image contains no people, explicitly check
the reviewed-empty box. Save or move to the next image to write a `class: 0`
record directly into the selected split JSONL. Person has only one class, so no
class selector is needed. Run separately with `--split val` and `--split test`
after assigning source scenes to those splits. Tk must be available in the
Python environment; the tool is local and does not send images anywhere. The
current workspace interpreter does not include Tk, so either add the OS Tk
package for that Python or use the COCO-export/conversion workflow below.

For an annotation GUI that exports COCO JSON, keep the annotation work local
and convert each split separately. The converter does not require a YOLO
format. COCO `[x,y,width,height]` boxes are converted to project
`[x1,y1,x2,y2]` without scaling. Put image files under the corresponding
dataset `images/` directory with paths matching COCO `file_name` after the
`images/` prefix. For COCO sources with extra categories, provide an explicit
class map and use `--skip-unmapped` only after reviewing ignored categories.
Crowd annotations are rejected by default: they may describe a group rather
than individual people. If you choose `--exclude-crowd-images`, every image
containing a crowd annotation is omitted from the JSONL (never made a negative).
Review the converter summary and resulting manifest before use.

Example class map file for a person dataset:

```json
{"person":"person"}
```

```bash
python3 -m ai.training.datasets.convert --format coco --task person \
  --input annotations_train.json --output data/person/train.jsonl \
  --class-map person_class_map.json --skip-unmapped
```

If a reviewed export contains crowd annotations you cannot individually box,
exclude those whole images explicitly by adding `--exclude-crowd-images`.

Run the same command for validation and test annotation exports, changing the
input and output paths. For fire, use a CSV with `image,label` columns. For
violence use `frames,label`; the `frames` CSV cell must contain a JSON array of
ordered relative frame paths. Labels can be integer IDs or exact class names.
For example, the violence cell value can be `"[\"frames/clip01_001.jpg\",\"frames/clip01_002.jpg\"]"`.

### Local weapon candidate and annotation workflow

Import local candidate images into an isolated review queue. Importing never
adds an image to a final split:

```bash
python3 -m ai.training.datasets.prepare_weapon_dataset --input candidate_weapon_images/
```

Candidates are copied to `data/weapon/candidates/images/`, deduplicated by
SHA-256, and indexed with source filename and group metadata. For local
collections, place each source/session in its own subfolder so the validator
can keep the group together across splits. Capture session names are also used
automatically as their group. Review locally:

```bash
python3 -m ai.training.datasets.annotate_weapon --data data/weapon --split train
```

Use `0`–`4` to choose a class, drag on empty space to draw, drag inside a box
to move it, and drag its lower-right handle to resize it. Select a box and
press a class key to edit its class; `X` deletes it. Press `A` to explicitly
approve a positive annotation, `N` to approve a negative after checking the
full image, `S` to leave it pending, and `Q` to quit. Only `A` and `N` copy an
image to `data/weapon/images/` and append a row with `reviewed:true` to the
selected split. COCO conversion marks weapon records `reviewed:false`, so they
cannot train until a human verifies them. Keep each source/capture session in
one split; choose split per group to target approximately 70/15/15. No AI box
proposals are generated.

For laptop-camera capture use only safe, appropriate training props; do not
handle functional weapons for collection. Captures go to an inbox, not a
final split:

```bash
python3 -m ai.training.datasets.capture_weapon_webcam --camera 0 --session weapon_session_01
python3 -m ai.training.datasets.prepare_weapon_dataset --input data/weapon/candidates/inbox
python3 -m ai.training.datasets.annotate_weapon --data data/weapon --split train
```

The capture window saves only when Space is pressed; `P` pauses/resumes and `Q`
quits.

### Validate and inspect statistics

Run the validator for each dataset before training:

```bash
python3 -m ai.training.datasets.validate --data data/person
python3 -m ai.training.datasets.validate --data data/weapon
python3 -m ai.training.datasets.validate --data data/fire
python3 -m ai.training.datasets.validate --data data/violence
```

The validator reports samples/valid/invalid, class counts, missing/unreadable
files, malformed annotations, empty detection annotations, invalid boxes,
duplicates, and train/validation/test leakage by path, image content, or source
group. Detection reports include objects per image and box width/height ranges.
Weapon reports also require review markers and include reviewed, positive,
negative, images/objects per class, and a 2x class-imbalance warning check.
Violence reports include clips, frame counts, missing frames, and
numbered-filename ordering issues. It exits
non-zero if a split is empty or samples are invalid. It cannot identify
semantic leakage when a scene has been copied under a new filename.

Get a compact summary with:

```bash
python3 -m ai.training.datasets.stats --data data/person
python3 -m ai.training.datasets.stats --data data/weapon
python3 -m ai.training.datasets.stats --data data/fire
python3 -m ai.training.datasets.stats --data data/violence
```

The requested folders and empty manifest files have been created, but no
images, clips, labels, or fake training samples have been added. Validation of
these empty manifests is expected to report zero samples until real data is
provided.

### Capture person images from the laptop webcam

Check camera 0 without opening a window or saving an image:

```bash
python3 -m ai.training.datasets.capture_person_webcam --test-only --camera 0
```

Start the local preview/capture window:

```bash
python3 -m ai.training.datasets.capture_person_webcam --camera 0 --session session_01
```

Use `--camera 1` (or another index) if camera 0 is unavailable. The tool checks
both `isOpened()` and the first frame before starting. It reports a clear error
and suggests another index if either check fails. It saves only when you press
Space; it never records video continuously. Press **P** to pause/resume, **D**
to review the last saved image, and **Q** or Escape to quit. The preview shows
the live frame number, images captured this run, session, and destination.

Images are written under `data/person/images/` as
`session_01_000001.jpg`, with collision-safe numbering so existing images are
not overwritten. Without `--session`, names use `person_cam_000001.jpg`. Use a
new session name when you change camera position, room, participants, or capture
conditions. Capture after a meaningful change rather than holding Space: vary
camera height/angle, distance, center/left/right placement, people count, poses,
occlusion, indoor/outdoor setting, and lighting. Include reviewed empty scenes
and different person scales. Avoid saving bursts of near-identical frames just
to reach 300–800; that target is a starting point, not an accuracy guarantee.

After capture, annotate every image, including negative frames. The local
person GUI writes class-0 boxes directly; alternatively use a local COCO
annotation export and the converter documented above. Keep each capture session
entirely in one of train, validation, or test before annotating; a recommended
starting split is 70/15/15. Then validate and inspect statistics with the
commands above. No images are captured unless you run the capture command and
press Space.

For all tasks, split at the source video/person/session level where possible.
Include varied lighting, viewpoints, distances, occlusions, and representative
negative examples. Review labels before training. Expected checkpoint defaults
are `checkpoints/person.pt`, `checkpoints/weapon.pt`, `checkpoints/fire.pt`,
and `checkpoints/violence.pt` relative to the working directory.

## Install and train

Install `requirements-ai.txt` in the chosen Python environment. Then train from
the repository root; each training command validates train, validation, and test JSONL,
prints validation loss (and accuracy for classifiers), and saves the best
validation-loss state dict. Training does not use or fetch pretrained weights.

```bash
python3 -m pip install -r requirements-ai.txt
python3 -m ai.training.train_person --data data/person --checkpoint checkpoints/person.pt --epochs 20
python3 -m ai.training.train_weapon --data data/weapon --checkpoint checkpoints/weapon.pt --epochs 20
python3 -m ai.training.train_fire --data data/fire --checkpoint checkpoints/fire.pt --epochs 20
python3 -m ai.training.train_violence --data data/violence --checkpoint checkpoints/violence.pt --epochs 20
```

Validate a saved checkpoint on the validation manifest with the same command
and `--validate-only`:

```bash
python3 -m ai.training.train_person --data data/person --checkpoint checkpoints/person.pt --validate-only
```

Run `python3 -m ai.training.smoke_test` to check model shapes, dataset rejection
cases, gradient updates, checkpoint round trips, and inference mechanics on
temporary synthetic data. Its metrics have no accuracy meaning and its files
are removed when the command exits.

Change `train_person` to another task's module and use that task's data and
checkpoint path. Keep the held-out test split untouched for final evaluation;
the starter CLI does not evaluate test sets.

## Inference and pipeline integration

Pass trained checkpoint paths explicitly. Missing or invalid checkpoints are
not replaced with another model, and no predictions are emitted for unloaded
models. The status object indicates `available`, `loaded`, and a reason.

```python
from ai import CameraStream, RestrictedZone, SurveillancePipeline

pipeline = SurveillancePipeline(
    zone=RestrictedZone([(0.2, 0.2), (0.8, 0.2), (0.8, 0.9), (0.2, 0.9)], is_normalized=True),
    person_checkpoint="checkpoints/person.pt",
    weapon_checkpoint="checkpoints/weapon.pt",
    violence_checkpoint="checkpoints/violence.pt",
    fire_checkpoint="checkpoints/fire.pt",
    device="cpu",  # set to "cuda" when available
)
with CameraStream(0) as camera:  # replace 0 with a video filename for a file
    ok, frame, timestamp = camera.read()
    if ok:
        result = pipeline.process(frame, timestamp)
```

The structured output includes per-model status and detections/results, plus
`persons`, `weapons`, `violence`, and `fire` compatibility fields. Zone checks
use the tracked person's feet point. `ViolenceInference.classify(frames)`
accepts a list of OpenCV BGR frames; the pipeline keeps a rolling clip history.
The pipeline waits for the configured clip length (eight frames by default)
before returning a violence classification.
For custom integration, import model definitions from `ai.models` and inference
adapters from `ai.inference`.

## Limitations and current operational state

The camera stream and polygon-zone geometry are operational. The custom model
architectures, training interfaces, inference adapters, and orchestration are
implemented, but **all AI predictions require datasets and trained weights**.
No detection or classification quality claim is made. The starter grid detector
has limited capacity and simple postprocessing, and the violence classifier
needs representative ordered clips. ID tracking may switch during occlusion.
Results are experimental signals for human review, not verified events or
determinations of intent. The model definitions and training code need dataset
validation, tuning, and held-out evaluation before operational use.
