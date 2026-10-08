"""Synthetic tests for Person evaluation matching and AP calculations."""
import unittest

from ai.training.evaluate_person import (average_precision, box_iou,
                                         evaluate_predictions, match_at_iou)


def prediction(box, confidence=0.9):
    return {"class": "person", "confidence": confidence, "bbox": box}


class PersonEvaluationTests(unittest.TestCase):
    def test_iou_for_overlapping_xyxy_boxes(self):
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)

    def test_greedy_matching_counts_duplicate_and_missed_ground_truth(self):
        ground_truth = [[[0, 0, 10, 10], [20, 20, 30, 30]]]
        predictions = [[prediction([0, 0, 10, 10], .9),
                        prediction([0, 0, 10, 10], .8)]]
        result = match_at_iou(predictions, ground_truth, .5, .5)
        self.assertEqual(result["ground_truth"], 2)
        self.assertEqual(result["predictions"], 2)
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["false_negatives"], 1)
        self.assertEqual(result["precision"], .5)
        self.assertEqual(result["recall"], .5)
        self.assertEqual(result["mean_iou"], 1.0)

    def test_ap_and_single_class_map(self):
        ground_truth = [[[0, 0, 10, 10]]]
        predictions = [[prediction([0, 0, 10, 10])]]
        self.assertEqual(average_precision(predictions, ground_truth, .5), 1.0)
        report = evaluate_predictions(predictions, ground_truth)
        self.assertEqual(report["ap50_person"], 1.0)
        self.assertEqual(report["map50_person"], 1.0)
        self.assertEqual(report["map50_95_person"], 1.0)

    def test_ap_uses_confidence_ranking(self):
        ground_truth = [[[0, 0, 10, 10]]]
        predictions = [[prediction([20, 20, 30, 30], .99),
                        prediction([0, 0, 10, 10], .8)]]
        self.assertAlmostEqual(average_precision(predictions, ground_truth, .5),
                               .5)


if __name__ == "__main__":
    unittest.main()
