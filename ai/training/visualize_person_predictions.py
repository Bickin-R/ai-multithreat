"""Save test-split images with Person CNN predictions and ground truth overlaid."""
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from ai.device import resolve_device
from ai.inference.common import decode_grid
from ai.models import LegacyPersonDetector20, PersonDetectorModel
from ai.training.datasets import DetectionDataset, PERSON_CLASSES
from ai.training.datasets.formats import load_rgb
from ai.training.train_utils import load_model_checkpoint
from ai.inference.legacy_person_inference import load_legacy_person_checkpoint


def annotate_image(image, ground_truth, predictions):
    """Return a copy with green GT boxes and red, confidence-labeled predictions."""
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    for box in ground_truth:
        draw.rectangle(tuple(map(float, box)), outline=(0, 255, 0), width=3)
    for item in predictions:
        x1, y1, x2, y2 = map(float, item["bbox"])
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
        label = f"{float(item['confidence']):.2f}"
        text_y = max(0, y1 - 14)
        draw.text((x1, text_y), label, fill=(255, 0, 0), stroke_width=1,
                  stroke_fill=(255, 255, 255))
    return result


def run_visualization(data="data/person", checkpoint="checkpoints/person_augmented.pt",
                      device="cuda", confidence=0.1, output="debug/person_predictions",
                      architecture="legacy20"):
    """Run inference on test.jsonl only and save annotated copies."""
    root = Path(data)
    manifest = root / "test.jsonl"
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Person checkpoint not found: {checkpoint}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    dataset = DetectionDataset(root, manifest, PERSON_CLASSES, augment=False)
    resolved_device = resolve_device(device)
    if architecture == "legacy20":
        model = load_legacy_person_checkpoint(
            LegacyPersonDetector20(), checkpoint, resolved_device)
    elif architecture == "current40":
        model = load_model_checkpoint(PersonDetectorModel(), checkpoint, resolved_device)
    else:
        raise ValueError("architecture must be 'legacy20' or 'current40'")
    model.eval()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    total_ground_truth = total_predictions = 0
    counts = []
    with torch.no_grad():
        for index, row in enumerate(dataset.rows):
            image_path = root / row["image"]
            input_tensor = load_rgb(image_path).unsqueeze(0).to(resolved_device)
            with Image.open(image_path) as source:
                original = source.convert("RGB")
            predictions = decode_grid(model(input_tensor),
                                      (original.height, original.width, 3),
                                      PERSON_CLASSES, threshold=confidence)
            ground_truth = [obj["bbox"] for obj in row["objects"]]
            annotated = annotate_image(original, ground_truth, predictions)
            destination = output / f"{index:03d}_{Path(row['image']).stem}_predictions.png"
            annotated.save(destination)
            total_ground_truth += len(ground_truth)
            total_predictions += len(predictions)
            counts.append((destination.name, len(predictions)))

    print(f"images_processed: {len(counts)}")
    print(f"total_ground_truth_boxes: {total_ground_truth}")
    print(f"total_predictions: {total_predictions}")
    print("prediction_count_per_image:")
    for filename, count in counts:
        print(f"  {filename}: {count}")
    print(f"output_directory: {output}")
    return {"images_processed": len(counts), "total_ground_truth_boxes": total_ground_truth,
            "total_predictions": total_predictions,
            "prediction_count_per_image": dict(counts), "output_directory": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Visualize Person CNN predictions on the test split.")
    parser.add_argument("--data", default="data/person")
    parser.add_argument("--checkpoint", default="checkpoints/person_augmented.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--output", default="debug/person_predictions")
    parser.add_argument("--architecture", choices=("legacy20", "current40"), default="legacy20",
                        help="Explicit model architecture; default matches person_augmented.pt")
    args = parser.parse_args(argv)
    try:
        return run_visualization(args.data, args.checkpoint, args.device,
                                 args.confidence, args.output, args.architecture)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Visualization failed: {exc}\n")


if __name__ == "__main__":
    main()
