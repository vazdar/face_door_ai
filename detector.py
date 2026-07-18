from __future__ import annotations

import cv2
import numpy as np

from config import (
    DETECTION_DOWNSCALE_WIDTH,
    DETECTION_MIN_NEIGHBORS,
    DETECTION_SCALE_FACTOR,
    MIN_FACE_SIZE,
)


class FaceDetector:
    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError("Không tải được Haar face cascade")

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        height, width = frame.shape[:2]

        # Chỉ downscale cho bước detect để tăng tốc Haar cascade.
        # Các bước sau (recognize, anti-spoof, liveness) vẫn nhận bbox
        # đã quy đổi lại về toạ độ ảnh gốc và luôn crop trên frame
        # full-res -> độ chính xác nhận diện/chống giả mạo không đổi.
        scale = 1.0
        search_frame = frame
        if DETECTION_DOWNSCALE_WIDTH and width > DETECTION_DOWNSCALE_WIDTH:
            scale = DETECTION_DOWNSCALE_WIDTH / float(width)
            new_size = (int(width * scale), int(height * scale))
            search_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(search_frame, cv2.COLOR_BGR2GRAY)

        min_size = (
            max(1, int(MIN_FACE_SIZE[0] * scale)),
            max(1, int(MIN_FACE_SIZE[1] * scale)),
        )

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=DETECTION_SCALE_FACTOR,
            minNeighbors=DETECTION_MIN_NEIGHBORS,
            minSize=min_size,
        )

        if scale != 1.0:
            faces = [
                (
                    int(x / scale),
                    int(y / scale),
                    int(w / scale),
                    int(h / scale),
                )
                for (x, y, w, h) in faces
            ]

        return [tuple(map(int, face)) for face in faces]
