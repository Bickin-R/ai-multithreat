"""Final evaluation of the custom Person detector on the held-out test split."""
import argparse
from pathlib import Path

import torch
from PIL import Image

from ai.device import resolve_device
from ai.inference.common import decode_grid
from ai.models import LegacyPersonDetector20, PersonDetectorModel
from ai.training.datasets import DetectionDataset, PERSON_CLASSES
from ai.training.datasets.formats import load_rgb
from ai.training.train_utils import load_model_checkpoint
from ai.inference.legacy_person_inference import load_legacy_person_checkpoint


def box_iou(first, second):
    """IoU for XYXY boxes."""
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def match_at_iou(predictions, ground_truth, iou_threshold=0.5,
                 confidence_threshold=0.5):
    """Greedily match score-ordered predictions to at most one GT each."""
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have the same image count")
    true_positives = false_positives = total_ground_truth = 0
    matched_ious = []
    for image_predictions, image_truth in zip(predictions, ground_truth):
        total_ground_truth += len(image_truth)
        unmatched = set(range(len(image_truth)))
        scored = sorted((item for item in image_predictions
                         if float(item["confidence"]) >= confidence_threshold),
                        key=lambda item: float(item["confidence"]), reverse=True)
        for item in scored:
            if unmatched:
                best_index = max(unmatched, key=lambda index:
                                 box_iou(item["bbox"], image_truth[index]))
                overlap = box_iou(item["bbox"], image_truth[best_index])
            else:
                best_index, overlap = None, 0.0
            if best_index is not None and overlap >= iou_threshold:
                unmatched.remove(best_index)
                true_positives += 1
                matched_ious.append(overlap)
            else:
                false_positives += 1
    false_negatives = total_ground_truth - true_positives
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / total_ground_truth if total_ground_truth else 0.0
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    return {"ground_truth": total_ground_truth,
            "predictions": true_positives + false_positives,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision, "recall": recall,
            "mean_iou": mean_iou}


def average_precision(predictions, ground_truth, iou_threshold=0.5):
    """COCO-style 101-point interpolated AP for the single person class."""
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have the same image count")
    total_ground_truth = sum(len(boxes) for boxes in ground_truth)
    if total_ground_truth == 0:
        return 0.0
    ranked = []
    for image_index, image_predictions in enumerate(predictions):
        for item in image_predictions:
            ranked.append((float(item["confidence"]), image_index, item["bbox"]))
    ranked.sort(key=lambda item: item[0], reverse=True)
    matched = [set() for _ in ground_truth]
    tp = fp = 0
    recalls, precisions = [], []
    for _, image_index, predicted_box in ranked:
        candidates = [index for index in range(len(ground_truth[image_index]))
                      if index not in matched[image_index]]
        if candidates:
            best_index = max(candidates, key=lambda index:
                             box_iou(predicted_box, ground_truth[image_index][index]))
            overlap = box_iou(predicted_box, ground_truth[image_index][best_index])
        else:
            best_index, overlap = None, 0.0
        if best_index is not None and overlap >= iou_threshold:
            matched[image_index].add(best_index)
            tp += 1
        else:
            fp += 1
        recalls.append(tp / total_ground_truth)
        precisions.append(tp / (tp + fp))
    if not precisions:
        return 0.0
    interpolated = []
    for recall_level in (index / 100 for index in range(101)):
        interpolated.append(max((precision for recall, precision in zip(recalls, precisions)
                                 if recall >= recall_level), default=0.0))
    return sum(interpolated) / len(interpolated)


def evaluate_predictions(predictions, ground_truth, confidence_threshold=0.5):
    """Compute thresholded counts, AP@0.50, and mean AP@0.50:0.95."""
    counts = match_at_iou(predictions, ground_truth, 0.5, confidence_threshold)
    ap50 = average_precision(predictions, ground_truth, 0.5)
    thresholds = [round(0.50 + 0.05 * index, 2) for index in range(10)]
    aps = {f"{threshold:.2f}": average_precision(predictions, ground_truth, threshold)
           for threshold in thresholds}
    return {"images": len(ground_truth), **counts,
            "confidence_threshold": confidence_threshold,
            "ap50_person": ap50,
            "map50_person": ap50,
            "map50_95_person": sum(aps.values()) / len(aps),
            "ap_by_iou_person": aps}


def run_evaluation(data_root="data/person", checkpoint="checkpoints/person_augmented.pt",
                   device="cuda" if torch.cuda.is_available() else "cpu",
                   confidence_threshold=0.5, architecture="legacy20"):
    """Evaluate only the supplied root's test.jsonl. No training data is opened."""
    root = Path(data_root)
    manifest = root / "test.jsonl"
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Person checkpoint not found: {checkpoint}")
    dataset = DetectionDataset(root, manifest, PERSON_CLASSES)
    device = resolve_device(device)
    if architecture == "legacy20":
        model = load_legacy_person_checkpoint(LegacyPersonDetector20(), checkpoint, device)
    elif architecture == "current40":
        model = load_model_checkpoint(PersonDetectorModel(), checkpoint, device)
    else:
        raise ValueError("architecture must be 'legacy20' or 'current40'")
    model.eval()

    predictions, ground_truth = [], []
    with torch.no_grad():
        for index, row in enumerate(dataset.rows):
            image_path = root / row["image"]
            image = load_rgb(image_path).unsqueeze(0).to(device)
            with Image.open(image_path) as source:
                width, height = source.size
            decoded = decode_grid(model(image), (height, width, 3), PERSON_CLASSES,
                                  threshold=0.0)
            predictions.append(decoded)
            ground_truth.append([obj["bbox"] for obj in row["objects"]])
    return evaluate_predictions(predictions, ground_truth, confidence_threshold)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the custom Person CNN on data/person/test.jsonl only.")
    parser.add_argument("--data", default="data/person", help="Person dataset root; only test.jsonl is loaded")
    parser.add_argument("--checkpoint", default="checkpoints/person_augmented.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--architecture", choices=("legacy20", "current40"), default="legacy20",
                        help="Explicit model architecture; default matches person_augmented.pt")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold for precision/recall counts; AP ranks all decoded predictions")
    args = parser.parse_args(argv)
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    try:
        report = run_evaluation(args.data, args.checkpoint, args.device, args.confidence,
                                args.architecture)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")
    for key, value in report.items():
        print(f"{key}: {value}")
    print("class: person (single-class detector; mAP is for person)")
    return report


if __name__ == "__main__":
    main()
