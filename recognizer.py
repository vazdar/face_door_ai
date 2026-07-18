from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import face_recognition
import numpy as np

from config import DATASET_DIR, FACE_TOLERANCE


@dataclass(frozen=True)
class RecognitionResult:
    name: str
    distance: float | None

    @property
    def known(self) -> bool:
        return self.name != "UNKNOWN"


class FaceRecognizer:
    def __init__(self, dataset_dir: Path = DATASET_DIR) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.known_encodings: list[np.ndarray] = []
        self.known_names: list[str] = []
        self.load_dataset()

    def load_dataset(self) -> None:
        self.known_encodings.clear()
        self.known_names.clear()
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        for person_dir in sorted(self.dataset_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            for image_path in sorted(person_dir.iterdir()):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                try:
                    image = face_recognition.load_image_file(str(image_path))
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_encodings.append(encodings[0])
                        self.known_names.append(person_dir.name)
                    else:
                        print(f"[DATASET] Không tìm thấy mặt: {image_path}")
                except Exception as exc:
                    print(f"[DATASET] Bỏ qua {image_path}: {exc}")

        print(f"[DATASET] Đã tải {len(self.known_encodings)} encoding")

    def recognize(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> RecognitionResult:
        if not self.known_encodings:
            return RecognitionResult("UNKNOWN", None)

        x, y, w, h = box
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            return RecognitionResult("UNKNOWN", None)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        encodings = face_recognition.face_encodings(
            rgb, known_face_locations=[(0, width, height, 0)]
        )
        if not encodings:
            return RecognitionResult("UNKNOWN", None)

        distances = face_recognition.face_distance(self.known_encodings, encodings[0])
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])
        if best_distance <= FACE_TOLERANCE:
            return RecognitionResult(self.known_names[best_index], best_distance)
        return RecognitionResult("UNKNOWN", best_distance)
