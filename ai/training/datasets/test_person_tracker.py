"""Unit tests for persistent Person detection IDs."""
import unittest
from unittest import mock

import numpy as np

from ai.tracking.person_tracker import PersonTracker
from ai.inference.person_webcam import annotate_frame


def det(box, confidence=.9):
    return {"bbox": box, "confidence": confidence, "class": "person"}


class PersonTrackerTests(unittest.TestCase):
    def test_one_person_keeps_id_across_frames(self):
        tracker = PersonTracker()
        first = tracker.update([det([10, 10, 50, 90])])[0]
        second = tracker.update([det([12, 11, 52, 91])])[0]
        self.assertEqual(first["person_id"], "P001")
        self.assertEqual(second["person_id"], first["person_id"])
        self.assertEqual(second["tracking_state"], "tracked")

    def test_two_people_keep_separate_ids(self):
        tracker = PersonTracker()
        tracks = tracker.update([det([0, 0, 30, 60]), det([200, 0, 230, 60])])
        self.assertEqual([track["person_id"] for track in tracks], ["P001", "P002"])
        matched = tracker.update([det([202, 1, 232, 61]), det([1, 0, 31, 60])])
        self.assertEqual([track["person_id"] for track in matched], ["P001", "P002"])

    def test_temporary_miss_retains_and_recovers_same_id(self):
        tracker = PersonTracker(max_missed_frames=2)
        initial = tracker.update([det([10, 10, 50, 90])])[0]
        missing = tracker.update([])[0]
        recovered = tracker.update([det([11, 10, 51, 90])])[0]
        self.assertEqual(missing["tracking_state"], "missing")
        self.assertEqual(recovered["person_id"], initial["person_id"])

    def test_track_expires_after_max_missed_frames(self):
        tracker = PersonTracker(max_missed_frames=2)
        tracker.update([det([10, 10, 50, 90])])
        tracker.update([])
        tracker.update([])
        self.assertEqual(len(tracker.update([])), 0)

    def test_new_person_gets_new_non_reused_id(self):
        tracker = PersonTracker(max_missed_frames=0)
        first = tracker.update([det([0, 0, 30, 60])])[0]["person_id"]
        tracker.update([])
        next_person = tracker.update([det([200, 0, 230, 60])])[0]["person_id"]
        self.assertEqual(first, "P001")
        self.assertEqual(next_person, "P002")

    def test_webcam_demo_renders_person_id(self):
        tracked = PersonTracker().update([det([5, 6, 40, 70])])
        with mock.patch("ai.inference.person_webcam.cv2.putText") as put_text:
            annotate_frame(np.zeros((80, 80, 3), dtype=np.uint8), tracked, 20.0)
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertTrue(any(label.startswith("ID: P001") for label in labels))


if __name__ == "__main__":
    unittest.main()
