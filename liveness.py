from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from config import LIVENESS_CLOSED_FRAMES, LIVENESS_TIMEOUT_SECONDS


@dataclass
class _BlinkState:
    phase: str = "WAIT_OPEN"
    closed_frames: int = 0
    verified_at: float = 0.0
    updated_at: float = 0.0


class BlinkLivenessDetector:
    """Liveness nhẹ, không cần model ngoài: yêu cầu một lần chớp mắt.

    Đây chỉ là lớp chống ảnh tĩnh cơ bản, chưa thay thế anti-spoofing bằng model.
    """

    def __init__(self) -> None:
        path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        self._eye_cascade = cv2.CascadeClassifier(path)
        self._states: dict[str, _BlinkState] = {}

    def check(self, frame: np.ndarray, box: tuple[int, int, int, int], key: str) -> tuple[bool, str]:
        now = time.monotonic()
        state = self._states.setdefault(key, _BlinkState(updated_at=now))
        if now - state.updated_at > LIVENESS_TIMEOUT_SECONDS:
            state.phase = "WAIT_OPEN"
            state.closed_frames = 0
            state.verified_at = 0.0
        state.updated_at = now

        if state.verified_at and now - state.verified_at <= LIVENESS_TIMEOUT_SECONDS:
            return True, "LIVE"

        x, y, w, h = box
        face = frame[y : y + h, x : x + w]
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        upper = gray[: max(1, int(h * 0.60)), :]
        eyes = self._eye_cascade.detectMultiScale(upper, 1.1, 4, minSize=(15, 15))
        eyes_open = len(eyes) >= 1

        if state.phase == "WAIT_OPEN":
            if eyes_open:
                state.phase = "WAIT_CLOSED"
            return False, "NHIN CAMERA"

        if state.phase == "WAIT_CLOSED":
            if eyes_open:
                state.closed_frames = 0
            else:
                state.closed_frames += 1
                if state.closed_frames >= LIVENESS_CLOSED_FRAMES:
                    state.phase = "WAIT_REOPEN"
            return False, "CHOP MAT"

        if state.phase == "WAIT_REOPEN" and eyes_open:
            state.verified_at = now
            return True, "LIVE"

        return False, "MO MAT"
