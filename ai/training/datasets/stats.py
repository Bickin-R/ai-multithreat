"""Run with: python3 -m ai.training.datasets.stats --data data/person"""
import argparse
from .audit import audit_dataset, SPLITS


def main(argv=None):
    parser=argparse.ArgumentParser(description="Summarize local dataset sizes, labels, and boxes.")
    parser.add_argument("--data",required=True)
    parser.add_argument("--task",choices=("person","weapon","fire","violence"))
    args=parser.parse_args(argv); report=audit_dataset(args.data,args.task)
    splits=report["splits"]; totals=report["totals"]
    print(f"Dataset: {report['root']} ({report['task']})")
    print(f"Samples: {totals['samples']}  Valid: {totals['valid_samples']}  Invalid: {totals['invalid_samples']}")
    print(f"Objects: {totals['objects']}")
    print(f"Classes: {totals['class_distribution']}")
    print(f"Train: {splits['train']['samples']}  Validation: {splits['validation']['samples']}  Test: {splits['test']['samples']}")
    print(f"Missing files/frames: {totals['missing_files']}  Invalid annotations: {totals['invalid_annotations']}  "
          f"Duplicates: {totals['duplicate_samples']}  Leakage entries: {len(report['leakage'])}")
    if report["kind"]=="detection":
        avg=totals["objects"]/totals["samples"] if totals["samples"] else 0
        print(f"Average objects/image: {avg:.3f}")
        boxes=[splits[s]["box_size"] for s,_ in SPLITS]
        bounds={key:([b[key] for b in boxes if b[key] is not None]) for key in ("min_width","max_width","min_height","max_height")}
        print("Bounding-box size px: " + ", ".join(f"{name}={min(values) if name.startswith('min') else max(values):.1f}" if values else f"{name}=n/a" for name,values in bounds.items()))
        print(f"Empty negative images: {totals['empty_annotations']}")
        if report["task"] == "weapon":
            print(f"Reviewed: {totals['reviewed_images']}  Positive: {totals['positive_images']}  Negative: {totals['negative_images']}")
            print("Class              Objects       Images")
            print("------------------------------------------------")
            for name in report["classes"]:
                print(f"{name:<18} {totals['class_distribution'].get(name, 0):>7} {totals['class_images'].get(name, 0):>11}")
            print(f"{'negative':<18} {'—':>7} {totals['negative_images']:>11}")
            counts=[n for n in totals["class_distribution"].values() if n]
            if len(counts)>1 and max(counts)>2*min(counts):
                print("IMBALANCE WARNING: largest weapon class has over 2x the objects of the smallest non-empty class.")
    if report["kind"]=="clip":
        clips=[splits[s]["frames_per_clip"] for s,_ in SPLITS if splits[s]["frames_per_clip"]["average"] is not None]
        frame_counts=[v for data in clips for v in (data["min"],data["max"]) if v is not None]
        print(f"Clips: {totals['samples']}  Frames per clip range: "
              f"{(min(frame_counts),max(frame_counts)) if frame_counts else 'n/a'}")
        print(f"Missing frames: {totals['missing_files']}  Ordering problems: "
              f"{sum(splits[s]['ordering_problems'] for s,_ in SPLITS)}")
    if report["leakage"]: print(f"Train/validation/test leakage entries: {len(report['leakage'])}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
