from __future__ import annotations

from dataclasses import dataclass
import math
import time

from config import (
    CENTER_FACE_MAX_OFFSET,
    MIN_PRIMARY_FACE_RATIO,
    PRESENCE_LOST_SECONDS,
)


@dataclass(frozen=True)
class PresenceResult:
    valid: bool
    primary_index: int | None
    status: str


class PersonPresenceTracker:
    """Chọn đúng người đứng trước camera: một mặt, đủ lớn và gần tâm ảnh."""

    def __init__(self) -> None:
        self.last_present_at = 0.0
        self.current_name = "NONE"

    def select(self, frame_shape, faces: list[tuple[int, int, int, int]]) -> PresenceResult:
        height, width = frame_shape[:2]
        if not faces:
            if time.monotonic() - self.last_present_at > PRESENCE_LOST_SECONDS:
                self.current_name = "NONE"
            return PresenceResult(False, None, "KHONG CO NGUOI")
        if len(faces) > 1:
            self.current_name = "MULTIPLE"
            return PresenceResult(False, None, "CHI DUOC 1 NGUOI")

        x, y, w, h = faces[0]
        face_ratio = (w * h) / float(width * height)
        face_cx, face_cy = x + w / 2.0, y + h / 2.0
        offset = math.hypot((face_cx - width / 2) / width, (face_cy - height / 2) / height)
        if face_ratio < MIN_PRIMARY_FACE_RATIO:
            return PresenceResult(False, 0, "LAI GAN CAMERA")
        if offset > CENTER_FACE_MAX_OFFSET:
            return PresenceResult(False, 0, "DUNG GIUA MAN HINH")
        self.last_present_at = time.monotonic()
        return PresenceResult(True, 0, "PERSON PRESENT")

    def set_name(self, name: str) -> None:
        self.current_name = name
