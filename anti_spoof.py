from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
import threading

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.data_io import transform as trans
from src.utility import parse_model_name

from config import (
    ANTI_SPOOF_MAX_WORKERS,
    ANTI_SPOOF_MODEL_PATH,
    ANTI_SPOOF_THRESHOLD,
)


@dataclass(frozen=True)
class AntiSpoofResult:
    is_live: bool
    live_score: float
    label: str
    available: bool


@dataclass
class LoadedModel:
    """Lưu predictor và thông tin crop của từng model."""

    path: Path
    predictor: AntiSpoofPredict
    input_height: int
    input_width: int
    scale: float | None


class AIAntiSpoofDetector:
    """
    Chạy các model MiniFASNet PyTorch .pth.

    Hỗ trợ:
    - Một file .pth.
    - Một thư mục chứa nhiều file .pth.
    - Ensemble nhiều model để tăng độ ổn định.
    - Fail-safe: thiếu model hoặc inference lỗi thì không mở cửa.

    Theo Silent-Face-Anti-Spoofing:
    - class 1: khuôn mặt thật
    - class 0 và class 2: khuôn mặt giả

    Hiệu năng: khi ensemble có nhiều model, các model được chạy SONG
    SONG bằng ThreadPoolExecutor thay vì tuần tự. PyTorch nhả GIL
    trong lúc chạy các phép tính conv ở backend C++, nên nhiều model
    có thể thực sự chạy đồng thời trên nhiều core. Kết quả trả về
    (điểm số, ngưỡng, nhãn) không đổi so với chạy tuần tự -> độ chính
    xác không bị ảnh hưởng, chỉ có độ trễ giảm.
    """

    LIVE_CLASS_INDEX = 1

    def __init__(
        self,
        model_path: Path = ANTI_SPOOF_MODEL_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self._models: list[LoadedModel] = []
        self._cropper = CropImage()
        self._lock = threading.Lock()
        self._load_error: str | None = None
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

        self.reload()

    @property
    def available(self) -> bool:
        return len(self._models) > 0

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _find_model_files(self) -> list[Path]:
        """
        Tìm model .pth.

        Nếu ANTI_SPOOF_MODEL_PATH đang trỏ tới anti_spoof.onnx cũ,
        chương trình tự tìm các file .pth trong cùng thư mục models.
        """

        path = self.model_path

        if path.is_file() and path.suffix.lower() == ".pth":
            return [path]

        if path.is_dir():
            return sorted(path.glob("*.pth"))

        # Hỗ trợ config cũ đang trỏ tới models/anti_spoof.onnx.
        parent = path.parent

        if parent.exists():
            return sorted(parent.glob("*.pth"))

        return []

    def _rebuild_executor(self) -> None:
        """Tạo lại thread pool theo đúng số model hiện có."""

        if self._executor is not None:
            self._executor.shutdown(wait=False)

        worker_count = ANTI_SPOOF_MAX_WORKERS or max(1, len(self._models))
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="anti-spoof",
        )

        # Khi chạy nhiều model song song trên nhiều thread, giới hạn
        # intra-op threads của mỗi lần forward về 1 để tránh các thread
        # tranh nhau CPU (oversubscription) và làm chậm hơn thay vì
        # nhanh hơn. Chỉ áp dụng khi thực sự có >1 model chạy song song.
        if len(self._models) > 1:
            torch.set_num_threads(1)

    def reload(self) -> bool:
        """Nạp toàn bộ model một lần khi khởi động."""

        self._models.clear()
        self._load_error = None

        model_files = self._find_model_files()

        if not model_files:
            self._load_error = (
                "Không tìm thấy model .pth trong: "
                f"{self.model_path if self.model_path.is_dir() else self.model_path.parent}"
            )
            print(f"[ANTI-SPOOF] {self._load_error}")
            self._rebuild_executor()
            return False

        errors: list[str] = []

        for model_file in model_files:
            try:
                height, width, _, scale = parse_model_name(model_file.name)

                predictor = AntiSpoofPredict(device_id=0)

                # Chỉ load model một lần, không load lại ở mỗi frame.
                predictor._load_model(str(model_file))
                predictor.model.eval()

                loaded = LoadedModel(
                    path=model_file,
                    predictor=predictor,
                    input_height=height,
                    input_width=width,
                    scale=scale,
                )

                self._models.append(loaded)

                print(
                    "[ANTI-SPOOF] Đã tải "
                    f"{model_file.name} | "
                    f"input={width}x{height} | "
                    f"scale={scale} | "
                    f"device={predictor.device}"
                )

            except Exception as exc:
                message = f"{model_file.name}: {exc}"
                errors.append(message)
                print(f"[ANTI-SPOOF] Không tải được {message}")

        if not self._models:
            self._load_error = "; ".join(errors) or "Không tải được model."
            self._rebuild_executor()
            return False

        if errors:
            self._load_error = (
                f"Đã tải {len(self._models)} model, "
                f"nhưng có lỗi: {'; '.join(errors)}"
            )
        else:
            self._load_error = None

        self._rebuild_executor()

        print(
            f"[ANTI-SPOOF] Sẵn sàng với "
            f"{len(self._models)} model | "
            f"chạy song song với {ANTI_SPOOF_MAX_WORKERS or len(self._models)} worker."
        )

        return True

    @staticmethod
    def _validate_box(
        frame: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int] | None:
        """Chuẩn hóa bbox và giới hạn trong kích thước frame."""

        if frame is None or frame.size == 0:
            return None

        x, y, width, height = [int(value) for value in box]

        frame_height, frame_width = frame.shape[:2]

        x = max(0, x)
        y = max(0, y)

        width = min(width, frame_width - x)
        height = min(height, frame_height - y)

        if width <= 1 or height <= 1:
            return None

        return x, y, width, height

    def _prepare_crop(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int],
        model: LoadedModel,
    ) -> np.ndarray | None:
        """Crop vùng khuôn mặt đúng theo scale và input của model."""

        try:
            crop = self._cropper.crop(
                org_img=frame,
                bbox=list(box),
                scale=model.scale,
                out_w=model.input_width,
                out_h=model.input_height,
                crop=model.scale is not None,
            )

            if crop is None or crop.size == 0:
                return None

            return crop

        except (ValueError, cv2.error) as exc:
            print(
                f"[ANTI-SPOOF] Crop lỗi với "
                f"{model.path.name}: {exc}"
            )
            return None

    @staticmethod
    def _to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        Áp dụng preprocessing giống repository gốc.

        Không đổi BGR sang RGB và không normalize -1..1 vì code gốc
        chỉ sử dụng ToTensor().
        """

        transform = trans.Compose([
            trans.ToTensor(),
        ])

        tensor = transform(image)
        tensor = tensor.unsqueeze(0)
        return tensor.to(device)

    def _predict_one(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int],
        model: LoadedModel,
    ) -> np.ndarray | None:
        """Chạy inference cho một model. An toàn để gọi từ nhiều thread
        cùng lúc vì mỗi lời gọi chỉ đọc model, không ghi state chung."""

        crop = self._prepare_crop(frame, box, model)

        if crop is None:
            return None

        tensor = self._to_tensor(
            crop,
            model.predictor.device,
        )

        with torch.no_grad():
            logits = model.predictor.model(tensor)
            probabilities = F.softmax(logits, dim=1)

        result = (
            probabilities
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float32)
        )

        if result.size <= self.LIVE_CLASS_INDEX:
            raise RuntimeError(
                f"Model {model.path.name} trả về "
                f"{result.size} class, không có class LIVE index "
                f"{self.LIVE_CLASS_INDEX}."
            )

        return result

    def _run_ensemble(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> list[np.ndarray]:
        """Chạy tất cả model trong ensemble song song và gom kết quả.

        Kết quả cuối cùng (danh sách các vector xác suất) giống hệt
        như khi chạy tuần tự -> việc lấy trung bình ensemble ở predict()
        không đổi, chỉ tốc độ thay đổi.
        """

        if self._executor is None:
            # fallback an toàn nếu executor chưa sẵn sàng
            return [
                result
                for model in self._models
                if (result := self._predict_one(frame, box, model)) is not None
            ]

        futures = [
            self._executor.submit(self._predict_one, frame, box, model)
            for model in self._models
        ]

        predictions: list[np.ndarray] = []
        for future in futures:
            result = future.result()
            if result is not None:
                predictions.append(result)

        return predictions

    def predict(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> AntiSpoofResult:
        """Chạy ensemble và trả kết quả REAL hoặc FAKE."""

        if not self.available:
            return AntiSpoofResult(
                is_live=False,
                live_score=0.0,
                label="MODEL MISSING",
                available=False,
            )

        valid_box = self._validate_box(frame, box)

        if valid_box is None:
            return AntiSpoofResult(
                is_live=False,
                live_score=0.0,
                label="INVALID FACE",
                available=True,
            )

        try:
            # Lock chỉ để đảm bảo không có 2 request submit ensemble
            # chồng lên nhau (ví dụ Flask threaded phục vụ 2 client).
            # Bên trong, các model của CÙNG một request vẫn chạy song
            # song với nhau qua ThreadPoolExecutor.
            with self._lock:
                predictions = self._run_ensemble(frame, valid_box)

            if not predictions:
                return AntiSpoofResult(
                    is_live=False,
                    live_score=0.0,
                    label="INVALID FACE",
                    available=True,
                )

            # Trung bình xác suất từ các model.
            ensemble = np.mean(
                np.stack(predictions, axis=0),
                axis=0,
            )

            live_score = float(
                ensemble[self.LIVE_CLASS_INDEX]
            )

            is_live = (
                live_score >= ANTI_SPOOF_THRESHOLD
            )

            return AntiSpoofResult(
                is_live=is_live,
                live_score=live_score,
                label="REAL" if is_live else "FAKE",
                available=True,
            )

        except Exception as exc:
            print(f"[ANTI-SPOOF] Inference error: {exc}")

            return AntiSpoofResult(
                is_live=False,
                live_score=0.0,
                label="MODEL ERROR",
                available=False,
            )
