"""Mechanics-only smoke test on temporary synthetic data; never measures accuracy."""
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from PIL import Image

from ai.models import PersonDetectorModel, WeaponDetectorModel, FireClassifier, ViolenceClassifier
from ai.training.datasets import (DetectionDataset, ImageClassificationDataset, ClipClassificationDataset,
                                 validate_dataset_splits, PERSON_CLASSES, WEAPON_CLASSES,
                                 FIRE_CLASSES, VIOLENCE_CLASSES)
from ai.training.train_utils import train_detector, train_classifier, load_model_checkpoint
from ai.inference import PersonInference, WeaponInference, FireInference, ViolenceInference
from ai.inference.common import decode_grid
from ai.pipeline import SurveillancePipeline


def _write_manifest(path, rows):
    path.write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8")


def _snapshot(model):
    return next(model.parameters()).detach().clone()


def _changed(model, before):
    return not torch.equal(before, next(model.parameters()).detach())


def _expect_rejected(call):
    try: call()
    except (ValueError, FileNotFoundError): return
    raise AssertionError("invalid dataset input was accepted")


def main():
    torch.manual_seed(7)
    x=torch.zeros((2,3,320,320))
    assert PersonDetectorModel()(x).shape==(2,10,40,40)
    assert WeaponDetectorModel()(x).shape==(2,10,20,20)
    assert FireClassifier()(x).shape==(2,2)
    assert ViolenceClassifier()(x.unsqueeze(1).repeat(1,2,1,1,1)).shape==(2,2)
    # Deterministic postprocessing check: class/confidence/box and NMS only.
    raw=torch.full((1,8,20,20),-12.0)
    box_logits=torch.logit(torch.tensor([0.5,0.5,0.2,0.4]))
    for ix in (10,11):
        raw[0,0,10,ix]=12; raw[0,1:5,10,ix]=box_logits; raw[0,5,10,ix]=12
    decoded=decode_grid(raw,(320,320,3),WeaponDetectorModel.class_names,threshold=0.5)
    assert len(decoded)==1 and decoded[0]["class"]=="knife" and decoded[0]["bbox"]==[128,96,192,224]
    assert decoded[0]["confidence"]>0.99
    status=SurveillancePipeline().process(np.zeros((64,80,3),dtype=np.uint8),timestamp=0.0)
    assert all(status[key]["available"] is False for key in ("person_detection","weapon_detection","violence_detection","fire_detection"))
    assert status["persons"]==[] and status["weapons"]==[] and status["violence"] is None and status["fire"] is None
    with tempfile.TemporaryDirectory(prefix="custom-ai-smoke-") as tmp:
        root=Path(tmp)
        # Make disjoint train, validation, and test samples.
        for i in range(18):
            array=np.full((64,80,3),i*7,dtype=np.uint8)
            Image.fromarray(array).save(root/f"frame_{i:02}.png")

        bad=root/"bad.jsonl"
        _write_manifest(bad,[{"image":"frame_00.png","objects":[{"class":0,"bbox":[-1,0,10,10]}]}])
        _expect_rejected(lambda:DetectionDataset(root,bad,PERSON_CLASSES))
        _write_manifest(bad,[{"image":"frame_00.png","objects":[{"class":1,"bbox":[0,0,10,10]}]}])
        _expect_rejected(lambda:DetectionDataset(root,bad,PERSON_CLASSES))
        _write_manifest(bad,[{"image":"missing.png","objects":[]}])
        _expect_rejected(lambda:DetectionDataset(root,bad,PERSON_CLASSES))
        bad.write_text("{malformed json\n",encoding="utf-8")
        _expect_rejected(lambda:DetectionDataset(root,bad,PERSON_CLASSES))
        _write_manifest(bad,[{"image":"frame_00.png"}])
        _expect_rejected(lambda:ImageClassificationDataset(root,bad,FIRE_CLASSES))
        good=root/"one.jsonl"; _write_manifest(good,[{"image":"frame_00.png","label":0}])
        _expect_rejected(lambda:validate_dataset_splits(root,good,good,good,"classification",FIRE_CLASSES))

        outputs={}
        for kind, model_cls, classes in (("person",PersonDetectorModel,PERSON_CLASSES),
                                          ("weapon",WeaponDetectorModel,WEAPON_CLASSES)):
            manifests={s:[] for s in ("train","val","test")}
            for split, indices in (("train",range(0,6)),("val",range(6,8)),("test",range(8,10))):
                for j,i in enumerate(indices):
                    objects=[] if kind=="person" and j==0 else [{"class":(j%len(classes) if kind=="weapon" else 0),
                                      "bbox":[8+j*2,7,38+j*2,50]}]
                    manifests[split].append({"image":f"frame_{i:02}.png","objects":objects})
            for split,rows in manifests.items(): _write_manifest(root/f"{kind}_{split}.jsonl",rows)
            validate_dataset_splits(root,root/f"{kind}_train.jsonl",root/f"{kind}_val.jsonl",root/f"{kind}_test.jsonl","detection",classes)
            train=DetectionDataset(root,root/f"{kind}_train.jsonl",classes)
            val=DetectionDataset(root,root/f"{kind}_val.jsonl",classes)
            model=model_cls(); before=_snapshot(model); ckpt=root/f"{kind}.pt"
            train_detector(model,train,val,ckpt,epochs=1,batch_size=2,lr=1e-3,device="cpu")
            assert _changed(model,before),f"{kind}: optimizer did not change model weights"
            assert ckpt.is_file(),f"{kind}: checkpoint not saved"
            loaded=model_cls(); load_model_checkpoint(loaded,ckpt,"cpu")
            adapter=(PersonInference if kind=="person" else WeaponInference)(ckpt,device="cpu")
            assert adapter.loaded and isinstance(adapter.detect(np.zeros((64,80,3),dtype=np.uint8)),list)
            outputs[kind]={"weights_changed":True,"checkpoint_loaded":True,"inference_ran":True}

        rows_by_split={s:[] for s in ("train","val","test")}
        for split, indices in (("train",range(10,12)),("val",range(12,14)),("test",range(14,16))):
            for j,i in enumerate(indices): rows_by_split[split].append({"image":f"frame_{i:02}.png","label":j%2})
        for split,rows in rows_by_split.items(): _write_manifest(root/f"fire_{split}.jsonl",rows)
        validate_dataset_splits(root,root/"fire_train.jsonl",root/"fire_val.jsonl",root/"fire_test.jsonl","classification",FIRE_CLASSES)
        model=FireClassifier(); before=_snapshot(model); ckpt=root/"fire.pt"
        train_classifier(model,ImageClassificationDataset(root,root/"fire_train.jsonl",FIRE_CLASSES),
                         ImageClassificationDataset(root,root/"fire_val.jsonl",FIRE_CLASSES),ckpt,1,2,1e-3,"cpu")
        assert _changed(model,before) and ckpt.is_file()
        load_model_checkpoint(FireClassifier(),ckpt,"cpu")
        assert FireInference(ckpt).classify(np.zeros((64,80,3),dtype=np.uint8)) is not None
        outputs["fire"]={"weights_changed":True,"checkpoint_loaded":True,"inference_ran":True}

        clip_rows={s:[] for s in ("train","val","test")}
        ranges={"train":range(0,2),"val":range(2,4),"test":range(4,6)}
        for split, clips in ranges.items():
            for j,k in enumerate(clips):
                clip_rows[split].append({"frames":[f"frame_{k*2:02}.png",f"frame_{k*2+1:02}.png"],"label":j%2})
        for split,rows in clip_rows.items(): _write_manifest(root/f"violence_{split}.jsonl",rows)
        validate_dataset_splits(root,root/"violence_train.jsonl",root/"violence_val.jsonl",root/"violence_test.jsonl","clip",VIOLENCE_CLASSES)
        model=ViolenceClassifier(); before=_snapshot(model); ckpt=root/"violence.pt"
        train_classifier(model,ClipClassificationDataset(root,root/"violence_train.jsonl",VIOLENCE_CLASSES),
                         ClipClassificationDataset(root,root/"violence_val.jsonl",VIOLENCE_CLASSES),ckpt,1,2,1e-3,"cpu")
        assert _changed(model,before) and ckpt.is_file()
        load_model_checkpoint(ViolenceClassifier(),ckpt,"cpu")
        clip=[np.zeros((64,80,3),dtype=np.uint8) for _ in range(4)]
        assert ViolenceInference(ckpt).classify(clip) is not None
        outputs["violence"]={"weights_changed":True,"checkpoint_loaded":True,"sequence_inference_ran":True}
        pipeline=SurveillancePipeline(person_checkpoint=root/"person.pt",weapon_checkpoint=root/"weapon.pt",
                                      fire_checkpoint=root/"fire.pt",violence_checkpoint=root/"violence.pt")
        first=pipeline.process(np.zeros((64,80,3),dtype=np.uint8),timestamp=1.0)
        assert first["violence"] is None and "Waiting" in first["violence_detection"]["reason"]
        for ts in range(2,9): second=pipeline.process(np.zeros((64,80,3),dtype=np.uint8),timestamp=float(ts))
        assert all(second[key]["available"] for key in ("person_detection","weapon_detection","violence_detection","fire_detection"))
        assert second["violence"] is not None
        outputs["pipeline"]={"missing_checkpoint_status_checked":True,"loaded_checkpoint_pipeline_ran":True}
        print("TRAINING MECHANICS ONLY; NO ACCURACY CLAIM:",outputs)


if __name__=="__main__": main()
