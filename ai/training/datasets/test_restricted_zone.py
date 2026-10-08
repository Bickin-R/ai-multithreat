"""Tests for polygon membership using tracked-person foot points."""
import unittest

from ai.security.restricted_zone import RestrictedZone
from ai.inference.person_webcam import parse_zone_points


def tracked(person_id, bbox, confidence=.8):
    return {"person_id": person_id, "bbox": bbox, "confidence": confidence}


class RestrictedZoneTests(unittest.TestCase):
    def setUp(self):
        self.zone = RestrictedZone([(100, 100), (500, 100), (500, 400), (100, 400)])

    def test_person_foot_clearly_inside(self):
        result = self.zone.evaluate([tracked("P001", [150, 150, 200, 300])])[0]
        self.assertTrue(result["inside_zone"])
        self.assertEqual(result["person_id"], "P001")
        self.assertEqual(result["bbox"], [150, 150, 200, 300])
        self.assertEqual(result["confidence"], .8)

    def test_person_foot_clearly_outside(self):
        self.assertFalse(self.zone.evaluate([tracked("P001", [600, 100, 650, 300])])[0]["inside_zone"])

    def test_bbox_overlap_does_not_count_when_foot_is_outside(self):
        # Box overlaps the polygon near its lower-left corner, but its foot is (90, 110).
        person = tracked("P001", [70, 80, 110, 110])
        result = self.zone.evaluate([person])[0]
        self.assertFalse(result["inside_zone"])

    def test_multiple_people_get_independent_zone_states(self):
        results = self.zone.evaluate([
            tracked("P001", [150, 150, 200, 300]),
            tracked("P002", [600, 100, 650, 300]),
        ])
        self.assertEqual([item["inside_zone"] for item in results], [True, False])

    def test_polygon_boundary_is_inclusive(self):
        # Foot point is exactly on the left polygon boundary.
        result = self.zone.evaluate([tracked("P001", [80, 150, 120, 200])])[0]
        self.assertTrue(result["inside_zone"])

    def test_cli_zone_points_configuration(self):
        configured = parse_zone_points("100,100;500,100;500,400;100,400")
        self.assertEqual(configured.points, self.zone.points)
        with self.assertRaisesRegex(ValueError, "--zone-points"):
            parse_zone_points(None)


if __name__ == "__main__":
    unittest.main()
