import argparse
from pathlib import Path
import torch
from ai.models import PersonDetectorModel, WeaponDetectorModel, ViolenceClassifier, FireClassifier
from ai.training.datasets import (ImageClassificationDataset, DetectionDataset, ClipClassificationDataset,
                                  validate_dataset_splits, PERSON_CLASSES, WEAPON_CLASSES,
                                  FIRE_CLASSES, VIOLENCE_CLASSES)
from ai.training.train_utils import train_classifier, train_detector, validate_checkpoint


def main(kind):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset root directory")
    parser.add_argument("--train", default="train.jsonl"); parser.add_argument("--val", default="val.jsonl")
    parser.add_argument("--test", default="test.jsonl")
    parser.add_argument("--checkpoint", default=f"checkpoints/{kind}.pt")
    parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3); parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(); root=Path(args.data)
    model_cls = {"person":PersonDetectorModel,"weapon":WeaponDetectorModel,"violence":ViolenceClassifier,"fire":FireClassifier}[kind]
    model=model_cls()
    class_names = {"person":PERSON_CLASSES,"weapon":WEAPON_CLASSES,"violence":VIOLENCE_CLASSES,"fire":FIRE_CLASSES}[kind]
    dataset_kind = "detection" if kind in ("person", "weapon") else "clip" if kind == "violence" else "classification"
    print("validated dataset splits:", validate_dataset_splits(root, root/args.train, root/args.val,
                                                                  root/args.test, dataset_kind, class_names))
    if kind in ("person","weapon"):
        train=DetectionDataset(root,root/args.train,class_names); val=DetectionDataset(root,root/args.val,class_names)
        if args.validate_only:
            from ai.training.train_utils import _detector_loss
            from ai.training.train_utils import load_model_checkpoint
            load_model_checkpoint(model,args.checkpoint,args.device)
            model.to(args.device)
            print({"validation_loss":_detector_loss(model,val,args.batch_size,args.device)})
        else: train_detector(model,train,val,args.checkpoint,args.epochs,args.batch_size,args.lr,args.device)
    else:
        DatasetClass=ClipClassificationDataset if kind=="violence" else ImageClassificationDataset
        train=DatasetClass(root,root/args.train,class_names); val=DatasetClass(root,root/args.val,class_names)
        if args.validate_only: validate_checkpoint(model,val,args.checkpoint,args.batch_size,args.device)
        else: train_classifier(model,train,val,args.checkpoint,args.epochs,args.batch_size,args.lr,args.device)
