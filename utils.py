from __future__ import annotations

import time
from datetime import datetime
from typing import Iterable

import cv2
import numpy as np


def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def decode_jpeg(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes | None:
    ok, buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    return buffer.tobytes() if ok else None


def draw_label(image, box: tuple[int, int, int, int], text: str, color) -> None:
    x, y, w, h = box
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    label_top = max(0, y - 30)
    cv2.rectangle(image, (x, label_top), (x + w, y), color, -1)
    cv2.putText(
        image,
        text,
        (x + 4, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )


class FPSCounter:
    def __init__(self) -> None:
        self._count = 0
        self._fps = 0
        self._last_tick = time.monotonic()

    def tick(self) -> int:
        self._count += 1
        now = time.monotonic()
        if now - self._last_tick >= 1.0:
            self._fps = self._count
            self._count = 0
            self._last_tick = now
        return self._fps
