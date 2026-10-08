"""Polygon restricted-zone membership for tracked people."""
import math


def _point_on_segment(point, start, end, tolerance=1e-9):
    px, py = point
    x1, y1 = start
    x2, y2 = end
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    scale = max(1.0, abs(x2 - x1), abs(y2 - y1))
    if abs(cross) > tolerance * scale:
        return False
    return (min(x1, x2) - tolerance <= px <= max(x1, x2) + tolerance and
            min(y1, y2) - tolerance <= py <= max(y1, y2) + tolerance)


def point_in_polygon(point, polygon):
    """Return membership using an inclusive boundary rule."""
    x, y = point
    inside = False
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        if _point_on_segment((x, y), start, end):
            return True
        x1, y1 = start
        x2, y2 = end
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing_x:
                inside = not inside
    return inside


class RestrictedZone:
    """A polygon zone in camera-frame pixel coordinates.

    Zone membership uses each person's bottom-center point. Polygon vertices
    may later be loaded from camera configuration or supplied by a UI without
    changing this geometry or evaluation API.
    """

    def __init__(self, points, name=None):
        try:
            polygon = tuple((float(point[0]), float(point[1])) for point in points)
        except (TypeError, IndexError, ValueError) as exc:
            raise ValueError("zone points must be coordinate pairs") from exc
        if len(polygon) < 3:
            raise ValueError("a restricted zone requires at least three points")
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in polygon):
            raise ValueError("zone coordinates must be finite")
        if len(set(polygon)) < 3:
            raise ValueError("a restricted zone requires at least three distinct points")
        area_twice = sum(
            polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
            - polygon[(i + 1) % len(polygon)][0] * polygon[i][1]
            for i in range(len(polygon))
        )
        if abs(area_twice) < 1e-9:
            raise ValueError("restricted zone polygon must have non-zero area")
        self.points = polygon
        self.name = name or "restricted_zone"

    def contains_point(self, point):
        """Check whether a frame-coordinate point lies inside/on the polygon."""
        if len(point) != 2 or not all(math.isfinite(float(value)) for value in point):
            raise ValueError("point must contain two finite coordinates")
        return point_in_polygon((float(point[0]), float(point[1])), self.points)

    @staticmethod
    def foot_point(person):
        """Return the bottom-center of a tracked person's bbox [x1,y1,x2,y2]."""
        try:
            x1, y1, x2, y2 = (float(value) for value in person["bbox"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("tracked person must have bbox [x1,y1,x2,y2]") from exc
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError("tracked person bbox coordinates must be finite")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("tracked person bbox must have positive width and height")
        return ((x1 + x2) / 2.0, y2)

    def evaluate(self, tracked_people):
        """Return zone status alongside identity, bbox and confidence per person."""
        results = []
        for person in tracked_people:
            if "person_id" not in person:
                raise ValueError("tracked person must have person_id")
            foot = self.foot_point(person)
            result = {
                "person_id": person["person_id"],
                "inside_zone": self.contains_point(foot),
                "bbox": list(person["bbox"]),
                "confidence": float(person["confidence"]),
            }
            for key in ("tracking_state", "missed_frames"):
                if key in person:
                    result[key] = person[key]
            results.append(result)
        return results
