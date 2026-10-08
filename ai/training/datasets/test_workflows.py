"""Tooling tests use temporary fixture images only; they do not train models."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image

from ai.training.datasets.audit import audit_dataset
from ai.training.datasets.convert import coco_to_jsonl, csv_to_jsonl
from ai.training.datasets.stats import main as stats_main
from ai.training.datasets.validate import main as validate_main
from ai.training.datasets.annotate_person import person_record
from ai.training.datasets.capture_person_webcam import _next_filename, _session_name


class DatasetWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="dataset-tools-test-")
        self.root=Path(self.temp.name)
        for i in range(8): Image.new("RGB",(64,48),(i*10,0,0)).save(self.root/f"frame_{i:03}.jpg")

    def tearDown(self): self.temp.cleanup()

    def write_rows(self,name,rows):
        path=self.root/name
        path.write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8")
        return path

    def test_detection_audit_and_stats(self):
        self.write_rows("train.jsonl",[{"image":"frame_000.jpg","objects":[{"class":0,"bbox":[1,2,20,30]}]},
                                         {"image":"frame_001.jpg","objects":[]}])
        self.write_rows("val.jsonl",[{"image":"frame_002.jpg","objects":[]}])
        self.write_rows("test.jsonl",[{"image":"frame_003.jpg","objects":[{"class":0,"bbox":[4,5,32,40]}]}])
        report=audit_dataset(self.root,"person")
        self.assertEqual(report["totals"]["samples"],4)
        self.assertEqual(report["totals"]["valid_samples"],4)
        self.assertEqual(report["totals"]["objects"],2)
        self.assertEqual(report["totals"]["empty_annotations"],2)
        self.assertEqual(report["totals"]["class_distribution"],{"person":2})
        output=io.StringIO()
        with contextlib.redirect_stdout(output): code=stats_main(["--data",str(self.root),"--task","person"])
        self.assertEqual(code,0); self.assertIn("Average objects/image",output.getvalue())
        self.assertIn("Bounding-box size",output.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(validate_main(["--data",str(self.root),"--task","person","--json"]),0)

    def test_invalid_annotation_and_missing_file_are_counted(self):
        self.write_rows("train.jsonl",[{"image":"missing.jpg","objects":[{"class":0,"bbox":[-1,2,5,8]}]}])
        self.write_rows("val.jsonl",[{"image":"frame_002.jpg","objects":[{"class":9,"bbox":[1,1,5,5]}]}])
        self.write_rows("test.jsonl",[{"image":"frame_003.jpg","objects":[]}])
        report=audit_dataset(self.root,"person")
        self.assertEqual(report["totals"]["invalid_samples"],2)
        self.assertGreaterEqual(report["totals"]["missing_files"],1)
        self.assertGreaterEqual(report["totals"]["invalid_bounding_boxes"],1)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertNotEqual(validate_main(["--data",str(self.root),"--task","person"]),0)

    def test_split_leakage_and_duplicate_detection(self):
        same={"image":"frame_000.jpg","label":0}
        self.write_rows("train.jsonl",[same,same])
        self.write_rows("val.jsonl",[same])
        self.write_rows("test.jsonl",[{"image":"frame_001.jpg","label":1}])
        report=audit_dataset(self.root,"fire")
        self.assertEqual(report["splits"]["train"]["duplicate_samples"],1)
        self.assertEqual(len(report["leakage"]),1)
        output=io.StringIO()
        with contextlib.redirect_stdout(output): stats_main(["--data",str(self.root),"--task","fire"])
        self.assertIn("Classes:",output.getvalue())

    def test_violence_clip_frame_counts_and_order(self):
        self.write_rows("train.jsonl",[{"frames":["frame_002.jpg","frame_001.jpg"],"label":1}])
        self.write_rows("val.jsonl",[{"frames":["frame_003.jpg","frame_004.jpg"],"label":0}])
        self.write_rows("test.jsonl",[{"frames":["frame_005.jpg","frame_006.jpg"],"label":0}])
        report=audit_dataset(self.root,"violence")
        self.assertEqual(report["totals"]["samples"],3)
        self.assertEqual(report["splits"]["train"]["ordering_problems"],1)
        self.assertEqual(report["splits"]["validation"]["frames_per_clip"]["average"],2)
        output=io.StringIO()
        with contextlib.redirect_stdout(output): stats_main(["--data",str(self.root),"--task","violence"])
        self.assertIn("Frames per clip range",output.getvalue())

    def test_coco_and_csv_annotation_converters(self):
        coco={"images":[{"id":1,"file_name":"person.jpg"},{"id":2,"file_name":"empty.jpg"}],
              "categories":[{"id":1,"name":"person"}],
              "annotations":[{"image_id":1,"category_id":1,"bbox":[2,3,10,12]},
                             {"image_id":1,"category_id":1,"bbox":[20,30,4,8]}]}
        source=self.root/"coco.json"; source.write_text(json.dumps(coco))
        out=self.root/"converted.jsonl"
        self.assertEqual(coco_to_jsonl(source,out,"person"),{"samples":2,"objects":2,
                         "skipped_unmapped_annotations":0,"skipped_categories":{},
                         "excluded_crowd_images":0,"crowd_annotations":0})
        rows=[json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(rows[0],{"image":"images/person.jpg","objects":[
            {"class":0,"bbox":[2.0,3.0,12.0,15.0]},
            {"class":0,"bbox":[20.0,30.0,24.0,38.0]}]})
        self.assertEqual(rows[1],{"image":"images/empty.jpg","objects":[]})
        csv_path=self.root/"clips.csv"
        csv_path.write_text('frames,label\n"[""a.jpg"",""b.jpg""]",violence\n')
        csv_out=self.root/"clips.jsonl"
        self.assertEqual(csv_to_jsonl(csv_path,csv_out,"violence"),{"samples":1})
        self.assertEqual(json.loads(csv_out.read_text()),{"frames":["a.jpg","b.jpg"],"label":1})
        weapon={"images":[{"id":4,"file_name":"knife.jpg"}],
                "categories":[{"id":7,"name":"knife"},{"id":8,"name":"handgun"}],
                "annotations":[{"image_id":4,"category_id":7,"bbox":[1,2,4,5]},
                               {"image_id":4,"category_id":8,"bbox":[10,12,3,6]}]}
        weapon_path=self.root/"weapon_coco.json"; weapon_path.write_text(json.dumps(weapon))
        weapon_out=self.root/"weapon.jsonl"
        coco_to_jsonl(weapon_path,weapon_out,"weapon",class_map={"knife":"knife","handgun":"gun"})
        converted=json.loads(weapon_out.read_text())
        self.assertEqual([obj["class"] for obj in converted["objects"]],[0,1])
        self.assertIs(converted["reviewed"],False)
        with self.assertRaisesRegex(ValueError,"numeric/legacy IDs are rejected"):
            coco_to_jsonl(weapon_path,weapon_out,"weapon",class_map={"knife":2})
        with self.assertRaisesRegex(ValueError,"never a final split"):
            coco_to_jsonl(weapon_path,self.root/"train.jsonl","weapon")

    def test_person_annotation_record_format(self):
        record=person_record("images/multi.jpg",[[42,30,190,286],[20,40,35,70]],320,300)
        self.assertEqual(record,{"image":"images/multi.jpg","objects":[
            {"class":0,"bbox":[42,30,190,286]},{"class":0,"bbox":[20,40,35,70]}]})
        self.assertEqual(person_record("images/empty.jpg",[],320,300),{"image":"images/empty.jpg","objects":[]})
        with self.assertRaises(ValueError): person_record("images/bad.jpg",[[0,0,321,5]],320,300)

    def test_capture_names_are_unique_and_session_safe(self):
        captures=self.root/"captures"; captures.mkdir()
        (captures/"person_cam_000001.jpg").touch()
        (captures/"person_cam_000003.jpg").touch()
        self.assertEqual(_next_filename(captures,None).name,"person_cam_000004.jpg")
        self.assertEqual(_next_filename(captures,"session_01").name,"session_01_000001.jpg")
        self.assertEqual(_session_name("session_01"),"session_01")
        with self.assertRaises(ValueError): _session_name("../outside")

    def test_crowd_conversion_requires_explicit_exclusion(self):
        coco={"images":[{"id":1,"file_name":"crowd.jpg"}],
              "categories":[{"id":1,"name":"person"}],
              "annotations":[{"image_id":1,"category_id":1,"bbox":[0,0,20,20],"iscrowd":1}]}
        source=self.root/"crowd.json"; source.write_text(json.dumps(coco)); out=self.root/"crowd.jsonl"
        with self.assertRaisesRegex(ValueError,"crowd annotation"):
            coco_to_jsonl(source,out,"person")
        summary=coco_to_jsonl(source,out,"person",exclude_crowd_images=True)
        self.assertEqual(summary["samples"],0); self.assertEqual(summary["excluded_crowd_images"],1)
        self.assertEqual(out.read_text(),"")


if __name__=="__main__": unittest.main()
