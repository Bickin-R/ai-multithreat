"""
Camera and Video Stream Interface.
Provides unified access to laptop webcams and recorded video files.
"""

import time
import threading
import logging
from typing import Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraStream:
    """
    Robust video capture stream supporting laptop webcams (0, 1, ...)
    and recorded video files (for demonstration backups).

    Features:
    - Threaded background frame reading to prevent buffer lag on webcams.
    - Automatic video file looping for repeatable hackathon demonstrations.
    - Clean context manager support (`with CameraStream(...) as cam:`).
    - Monotonic timestamps for accurate tracking and speed calculation.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        threaded: bool = False,
        loop: bool = True,
    ):
        """
        Initialize the camera or video stream.

        :param source: Webcam device index (int, e.g. 0) or path to video file (str).
        :param width: Optional desired frame width.
        :param height: Optional desired frame height.
        :param fps: Optional desired target FPS.
        :param threaded: Run reader thread in background to prevent buffer latency.
        :param loop: Loop video playback when EOF is reached (applies to video files).
        """
        self.source = source
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self.threaded = threaded
        self.loop = loop

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_video_file = isinstance(source, str)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._current_frame: Optional[np.ndarray] = None
        self._current_timestamp: float = 0.0
        self._has_frame: bool = False

        self.open()

    def open(self) -> bool:
        """Opens or re-opens the capture stream."""
        if self.cap is not None:
            self.cap.release()

        # Open capture
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(str(self.source))

        if not self.cap.isOpened():
            logger.error("Failed to open video source: %s", self.source)
            return False

        # Apply settings if provided
        if self.requested_width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        if self.requested_height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        if self.requested_fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.requested_fps)

        if self.threaded:
            self._running = True
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()

        return True

    def _reader_loop(self):
        """Background thread to drain capture buffer and keep the latest frame ready."""
        while self._running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                if self.is_video_file and self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    with self._lock:
                        self._has_frame = False
                    break

            ts = time.time()
            with self._lock:
                self._current_frame = frame
                self._current_timestamp = ts
                self._has_frame = True

            # Small sleep to yield CPU
            time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Read the latest available frame.

        :return: (success: bool, frame: np.ndarray, timestamp: float)
        """
        if self.threaded:
            with self._lock:
                if not self._has_frame or self._current_frame is None:
                    return False, None, time.time()
                return True, self._current_frame.copy(), self._current_timestamp

        if self.cap is None or not self.cap.isOpened():
            return False, None, time.time()

        ret, frame = self.cap.read()
        ts = time.time()

        if not ret:
            if self.is_video_file and self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return False, None, ts
            else:
                return False, None, ts

        return True, frame, ts

    @property
    def fps(self) -> float:
        """Returns detected or stream FPS."""
        if self.cap and self.cap.isOpened():
            val = self.cap.get(cv2.CAP_PROP_FPS)
            return val if val > 0 else 30.0
        return 30.0

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Returns (width, height) of frames."""
        if self.cap and self.cap.isOpened():
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return 0, 0

    def is_opened(self) -> bool:
        """Checks if stream is active."""
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        """Releases the camera or file resource."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def list_available_cameras(max_tested: int = 5) -> list:
    """
    Utility function to discover connected webcams on the host system.

    :param max_tested: Maximum device index to check.
    :return: List of valid camera indices.
    """
    available = []
    for idx in range(max_tested):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
            cap.release()
    return available
