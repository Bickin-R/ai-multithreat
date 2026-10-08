"""
Restricted Zone Detection Module.
Evaluates whether tracked individuals breach a user-defined restricted boundary.
"""

from typing import List, Tuple, Union, Dict, Any, Optional
import cv2
import numpy as np


class RestrictedZone:
    """
    Represents a user-defined restricted monitoring area.
    Can be configured via a polygon (list of 3+ points) or a rectangle [x1, y1, x2, y2].
    Supports both pixel coordinates and normalized coordinates [0.0 - 1.0].
    """

    def __init__(
        self,
        coordinates: Union[List[Tuple[float, float]], List[List[float]], Tuple[float, float, float, float]],
        is_normalized: bool = False,
        zone_name: str = "restricted_area",
    ):
        """
        :param coordinates: Either a list of (x, y) polygon vertices OR a rectangle tuple (x1, y1, x2, y2).
        :param is_normalized: True if coordinates are normalized [0.0, 1.0] relative to frame dimensions.
        :param zone_name: Optional identifier for the zone.
        """
        self.zone_name = zone_name
        self.is_normalized = is_normalized
        self.polygon_pts = self._parse_coordinates(coordinates)

    def _parse_coordinates(self, coords) -> np.ndarray:
        """Parses polygon or rectangle into numpy vertex array of shape (N, 2)."""
        if isinstance(coords, (tuple, list)) and len(coords) == 4 and not isinstance(coords[0], (tuple, list)):
            # Rectangle format: [x1, y1, x2, y2]
            x1, y1, x2, y2 = coords
            pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        else:
            # Polygon vertices
            pts = [[float(p[0]), float(p[1])] for p in coords]

        if len(pts) < 3:
            raise ValueError(f"RestrictedZone requires at least 3 points, got {len(pts)}")

        return np.array(pts, dtype=np.float32)

    def get_pixel_polygon(self, frame_width: int, frame_height: int) -> np.ndarray:
        """Returns polygon coordinates in integer pixel values for the given frame dimensions."""
        if self.is_normalized:
            pixel_pts = self.polygon_pts * np.array([frame_width, frame_height], dtype=np.float32)
            return pixel_pts.astype(np.int32)
        return self.polygon_pts.astype(np.int32)

    def contains_point(
        self,
        point: Union[List[float], Tuple[float, float]],
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """
        Tests whether a 2D point (x, y) is inside the restricted polygon.

        :param point: (x, y) coordinates of the point to test.
        :param frame_shape: Optional (height, width) if coordinates need pixel scaling.
        :return: True if point is inside or on the boundary of the zone.
        """
        x, y = float(point[0]), float(point[1])

        if self.is_normalized and frame_shape is not None:
            h, w = frame_shape[:2]
            poly = self.get_pixel_polygon(w, h)
            # Check using cv2.pointPolygonTest
            dist = cv2.pointPolygonTest(poly, (float(x), float(y)), measureDist=False)
            return dist >= 0
        elif not self.is_normalized:
            poly = self.polygon_pts.astype(np.int32)
            dist = cv2.pointPolygonTest(poly, (float(x), float(y)), measureDist=False)
            return dist >= 0
        else:
            # Normalized coordinates without frame shape
            poly = self.polygon_pts.astype(np.float32)
            # Ray casting algorithm
            n = len(poly)
            inside = False
            p1x, p1y = poly[0]
            for i in range(n + 1):
                p2x, p2y = poly[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            return inside

    def contains_person(
        self,
        person: Dict[str, Any],
        frame_shape: Optional[Tuple[int, int]] = None,
        anchor: str = "feet",
    ) -> bool:
        """
        Checks whether a detected/tracked person is inside the restricted zone.

        Surveillance Best Practice:
        The 'feet' anchor (bottom-center of bounding box) accurately reflects the ground plane
        location of a walking/standing human, minimizing perspective false-positives.

        :param person: Dict containing at least 'bbox': [x1, y1, x2, y2] or 'center': [cx, cy].
        :param frame_shape: (height, width) of the video frame.
        :param anchor: 'feet' (bottom-center [cx, y2]), 'center' ([cx, cy]), or 'bbox_any'.
        :return: True if inside the restricted zone.
        """
        bbox = person.get("bbox")
        center = person.get("center")

        if anchor == "feet" and bbox is not None:
            x1, y1, x2, y2 = bbox
            test_pt = [(x1 + x2) / 2.0, float(y2)]
            return self.contains_point(test_pt, frame_shape)

        elif anchor == "center":
            if center is not None:
                return self.contains_point(center, frame_shape)
            elif bbox is not None:
                x1, y1, x2, y2 = bbox
                test_pt = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
                return self.contains_point(test_pt, frame_shape)

        elif anchor == "bbox_any" and bbox is not None:
            # Check corners and center
            x1, y1, x2, y2 = bbox
            pts = [
                [(x1 + x2) / 2.0, float(y2)],  # feet
                [(x1 + x2) / 2.0, (y1 + y2) / 2.0],  # center
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ]
            return any(self.contains_point(p, frame_shape) for p in pts)

        # Fallback to center
        if center is not None:
            return self.contains_point(center, frame_shape)
        return False

    def draw(
        self,
        frame: np.ndarray,
        is_breached: bool = False,
        alpha: float = 0.25,
        color_normal: Tuple[int, int, int] = (255, 180, 0),   # Cyan/amber
        color_breached: Tuple[int, int, int] = (0, 0, 255),   # Red
    ) -> np.ndarray:
        """
        Renders a semi-transparent filled polygon overlay with a glowing border onto the frame.
        """
        h, w = frame.shape[:2]
        pts = self.get_pixel_polygon(w, h).reshape((-1, 1, 2))
        color = color_breached if is_breached else color_normal

        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw thick border
        thickness = 3 if is_breached else 2
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

        # Draw zone label
        pt_label = pts[0][0]
        label = f"RESTRICTED ZONE: {'[BREACH]' if is_breached else '[SECURE]'}"
        cv2.putText(
            frame,
            label,
            (int(pt_label[0]), max(20, int(pt_label[1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        return frame


def check_zone(
    person: Union[Dict[str, Any], List[float], Tuple[float, float]],
    zone_coordinates: Union[RestrictedZone, List, Tuple],
    frame_shape: Optional[Tuple[int, int]] = None,
    anchor: str = "feet",
) -> bool:
    """
    Direct function interface for Team Member 2.

    Determines: "Is this person's position inside the restricted area?"
    Returns: inside_zone = true / false

    :param person: Either a person dict (with 'bbox' or 'center') or an (x, y) point.
    :param zone_coordinates: Either a RestrictedZone instance or list/tuple of coordinates.
    :param frame_shape: Optional (height, width) of the frame.
    :param anchor: Anchor point to test ('feet', 'center').
    :return: bool
    """
    if isinstance(zone_coordinates, RestrictedZone):
        zone = zone_coordinates
    else:
        zone = RestrictedZone(zone_coordinates)

    if isinstance(person, dict):
        return zone.contains_person(person, frame_shape=frame_shape, anchor=anchor)
    elif isinstance(person, (list, tuple)):
        return zone.contains_point(person, frame_shape=frame_shape)
    return False


def annotate_zone_status(
    persons: List[Dict[str, Any]],
    zone: Optional[Union[RestrictedZone, List, Tuple]],
    frame_shape: Optional[Tuple[int, int]] = None,
    anchor: str = "feet",
) -> List[Dict[str, Any]]:
    """
    Iterates through a list of detected/tracked persons and populates 'inside_zone': bool.
    """
    if zone is None:
        for p in persons:
            p["inside_zone"] = False
        return persons

    rz = zone if isinstance(zone, RestrictedZone) else RestrictedZone(zone)
    for p in persons:
        p["inside_zone"] = rz.contains_person(p, frame_shape=frame_shape, anchor=anchor)
    return persons
