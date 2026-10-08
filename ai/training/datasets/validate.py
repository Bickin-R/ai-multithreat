"""Run with: python3 -m ai.training.datasets.validate --data data/person"""
import argparse
import json
from .audit import audit_dataset, SPLITS


def render(report):
    print(f"Dataset: {report['root']}  task={report['task']}  classes={', '.join(report['classes'])}")
    for split, _ in SPLITS:
        data=report["splits"][split]
        print(f"\n{split.title()}: samples={data['samples']} valid={data['valid_samples']} invalid={data['invalid_samples']}")
        print(f"  class distribution: {data['class_distribution']}")
        print(f"  missing files/frames={data['missing_files']} invalid annotations={data['invalid_annotations']} "
              f"invalid boxes={data['invalid_bounding_boxes']} duplicate samples={data['duplicate_samples']}")
        if report["kind"]=="detection":
            if report["task"]=="weapon":
                print(f"  reviewed={data['reviewed_images']} positive={data['positive_images']} negative={data['negative_images']}")
                print(f"  images per class: {data['class_images']}")
            print(f"  objects={data['objects']} empty annotations={data['empty_annotations']} "
                  f"objects/image range={data['objects_per_sample_range']} "
                  f"average objects/image={data['average_objects_per_sample']:.3f}")
            print(f"  box size pixels: {data['box_size']}")
        if report["kind"]=="clip":
            print(f"  clips={data['samples']} frames/clip={data['frames_per_clip']} "
                  f"ordering problems={data['ordering_problems']}")
        for error in data["errors"]:
            print(f"  ERROR: {error}")
    print(f"\nTotals: {report['totals']}")
    if report["task"]=="weapon":
        imbalance=report["totals"]["class_imbalance"]
        if imbalance["significant"]:
            print(f"CLASS IMBALANCE WARNING: object-count ratio={imbalance['ratio']:.2f}; {imbalance['criterion']}.")
        else:
            print(f"Class imbalance: no significant ratio detected (ratio={imbalance['ratio']}).")
    print(f"Train/validation/test leakage entries: {len(report['leakage'])}")
    for item in report["leakage"]:
        print(f"  LEAKAGE: {item['path']} appears in {', '.join(item['splits'])}")


def main(argv=None):
    parser=argparse.ArgumentParser(description="Validate a local custom-AI dataset without modifying or downloading it.")
    parser.add_argument("--data",required=True,help="Dataset root directory")
    parser.add_argument("--task",choices=("person","weapon","fire","violence"),help="Defaults to directory name")
    parser.add_argument("--json",action="store_true",help="Print machine-readable JSON")
    args=parser.parse_args(argv)
    report=audit_dataset(args.data,args.task)
    if args.json: print(json.dumps(report,indent=2))
    else: render(report)
    if report["totals"]["invalid_samples"] or any(report["splits"][s]["samples"]==0 for s,_ in SPLITS):
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
