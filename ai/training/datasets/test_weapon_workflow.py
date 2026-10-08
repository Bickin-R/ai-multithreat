"""Weapon model/data safeguards; fixtures live in temporary directories only."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from PIL import Image

from ai.models import WeaponDetectorModel
from ai.inference.common import decode_grid, load_checkpoint
from ai.training.datasets.audit import audit_dataset
from ai.training.datasets.formats import WEAPON_CLASSES, validate_dataset_splits
from ai.training.datasets.prepare_weapon_dataset import import_candidates
from ai.training.datasets.annotate_weapon import WeaponAnnotator
from ai.training.train_utils import load_model_checkpoint


class WeaponWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="weapon-workflow-test-")
        self.root=Path(self.temp.name); self.data=self.root/"data"; self.data.mkdir()
        for i in range(3): Image.new("RGB",(64,48),(i*30,0,0)).save(self.data/f"image_{i}.jpg")

    def tearDown(self): self.temp.cleanup()

    def rows(self,split,rows):
        path=self.data/f"{split}.jsonl"
        path.write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8")
        return path

    def test_five_class_model_and_detection_output(self):
        self.assertEqual(WEAPON_CLASSES,["knife","gun","arivaal","baseball_bat","other_weapon"])
        model=WeaponDetectorModel().eval()
        with torch.no_grad(): output=model(torch.zeros(2,3,320,320))
        self.assertEqual(tuple(output.shape),(2,10,20,20))
        raw=torch.full((1,10,20,20),-8.0); raw[0,0,10,10]=8; raw[0,1:5,10,10]=0; raw[0,7,10,10]=8
        detection=decode_grid(raw,(320,320,3),WEAPON_CLASSES,threshold=.5)[0]
        self.assertEqual(detection["class_id"],2); self.assertEqual(detection["class_name"],"arivaal")
        self.assertEqual(detection["class"],"arivaal"); self.assertEqual(len(detection["bbox"]),4)
        self.assertGreater(detection["confidence"],.5)

    def test_weapon_manifests_require_review_and_report_class_counts(self):
        positive={"image":"image_0.jpg","objects":[{"class":2,"bbox":[1,2,20,30]}],
                  "reviewed":True,"source_group":"session_a"}
        self.rows("train",[positive]); self.rows("val",[{"image":"image_1.jpg","objects":[],"reviewed":True,"source_group":"session_b"}])
        self.rows("test",[{"image":"image_2.jpg","objects":[{"class":4,"bbox":[2,2,21,31]}],
                            "reviewed":True,"source_group":"session_c"}])
        report=audit_dataset(self.data,"weapon")
        self.assertEqual(report["totals"]["reviewed_images"],3)
        self.assertEqual(report["totals"]["positive_images"],2)
        self.assertEqual(report["totals"]["negative_images"],1)
        self.assertEqual(report["totals"]["class_images"],{"arivaal":1,"other_weapon":1})
        self.assertFalse(report["totals"]["class_imbalance"]["significant"])
        self.assertEqual(validate_dataset_splits(self.data,self.data/"train.jsonl",self.data/"val.jsonl",
                                                  self.data/"test.jsonl","detection",WEAPON_CLASSES),
                         {"train":1,"validation":1,"test":1})
        self.rows("val",[{"image":"image_1.jpg","objects":[]}])
        with self.assertRaisesRegex(ValueError,"reviewed=true"):
            validate_dataset_splits(self.data,self.data/"train.jsonl",self.data/"val.jsonl",
                                    self.data/"test.jsonl","detection",WEAPON_CLASSES)

    def test_unknown_class_and_cross_split_same_content_fail_audit(self):
        duplicate=self.data/"copy.jpg"; duplicate.write_bytes((self.data/"image_0.jpg").read_bytes())
        self.rows("train",[{"image":"image_0.jpg","objects":[{"class":8,"bbox":[1,1,4,4]}],"reviewed":True}])
        self.rows("val",[{"image":"copy.jpg","objects":[],"reviewed":True}])
        self.rows("test",[{"image":"image_2.jpg","objects":[],"reviewed":True}])
        report=audit_dataset(self.data,"weapon")
        self.assertGreater(report["totals"]["invalid_samples"],0)
        self.assertTrue(any("invalid class ID" in error for error in report["splits"]["train"]["errors"]))
        self.assertTrue(any(item["path"].startswith("sha256:") for item in report["leakage"]))

    def test_candidate_import_deduplicates_and_never_writes_final_manifests(self):
        source=self.root/"candidates"; source.mkdir()
        Image.new("RGB",(20,20),(1,2,3)).save(source/"first.jpg")
        (source/"duplicate.jpg").write_bytes((source/"first.jpg").read_bytes())
        report=import_candidates(source,self.data)
        self.assertEqual(report["added"],1); self.assertEqual(report["duplicates"],1)
        self.assertFalse((self.data/"train.jsonl").exists())
        self.assertTrue((self.data/"candidates"/"index.jsonl").is_file())

    def test_only_explicit_review_promotes_candidate_to_final_manifest(self):
        candidate_dir=self.data/"candidates"/"images"; candidate_dir.mkdir(parents=True)
        image=candidate_dir/"candidate.jpg"; Image.new("RGB",(40,30),(5,6,7)).save(image)
        index=self.data/"candidates"/"index.jsonl"
        from ai.training.datasets.prepare_weapon_dataset import sha256
        index.write_text(json.dumps({"candidate":"candidates/images/candidate.jpg","source_name":"candidate.jpg",
                                     "source_group":"session_one","sha256":sha256(image),"status":"pending"})+"\n")
        annotator=WeaponAnnotator(self.data,"train")
        annotator.boxes=[{"class":3,"bbox":[2,3,20,25]}]
        annotator._promote(False)
        record=json.loads((self.data/"train.jsonl").read_text())
        self.assertIs(record["reviewed"],True); self.assertEqual(record["objects"][0]["class"],3)
        self.assertTrue((self.data/record["image"]).is_file())
        self.assertEqual(json.loads(index.read_text())["status"],"reviewed_positive")

    def test_legacy_three_class_checkpoint_is_explicitly_rejected(self):
        path=self.root/"legacy.pt"
        torch.save({"format_version":1,"model_name":"WeaponDetectorModel",
                    "class_names":["knife","gun","other_weapon"],"input_size":[320,320],"state_dict":{}},path)
        with self.assertRaisesRegex(ValueError,"legacy 3-class weapon checkpoint"):
            load_model_checkpoint(WeaponDetectorModel(),path)
        with self.assertRaisesRegex(ValueError,"legacy 3-class weapon checkpoint"):
            load_checkpoint(WeaponDetectorModel(),path,"cpu")


if __name__=="__main__": unittest.main()
