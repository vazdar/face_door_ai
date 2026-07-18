from __future__ import annotations

import threading
import time
from collections import defaultdict

import cv2
from flask import Flask, Response, jsonify, render_template, request

from anti_spoof import AIAntiSpoofDetector
from config import (
    ANTI_SPOOF_ENABLED,
    ENABLE_TIMING_LOG,
    HOST,
    JPEG_QUALITY,
    LIVENESS_ENABLED,
    PORT,
    RECOGNITION_RESET_SECONDS,
    REQUIRED_STABLE_FRAMES,
    REQUIRE_ACTIVE_CHALLENGE,
)
from database import AccessDatabase
from detector import FaceDetector
from door_controller import DoorController
from liveness import BlinkLivenessDetector
from presence import PersonPresenceTracker
from recognizer import FaceRecognizer
from utils import FPSCounter, decode_jpeg, draw_label, encode_jpeg

app = Flask(__name__)

detector = FaceDetector()
recognizer = FaceRecognizer()
anti_spoof = AIAntiSpoofDetector()
liveness = BlinkLivenessDetector()
presence = PersonPresenceTracker()
database = AccessDatabase()
door = DoorController()
fps_counter = FPSCounter()

latest_frame: bytes | None = None
frame_condition = threading.Condition()
recognition_streak: dict[str, int] = defaultdict(int)
last_seen: dict[str, float] = {}
last_result = {
    "person": "NONE",
    "presence": "KHONG CO NGUOI",
    "anti_spoof": "WAITING",
    "live_score": 0.0,
    "challenge": "WAITING",
}


class _StageTimer:
    """Đo thời gian từng bước xử lý một frame để tìm bottleneck thực tế.

    Chỉ hoạt động khi ENABLE_TIMING_LOG=True trong config.py. Không
    thay đổi bất kỳ logic xử lý nào, chỉ ghi lại và in ra thời gian.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._stage_start = 0.0
        self._marks: list[tuple[str, float]] = []
        self._total_start = 0.0

    def start(self) -> None:
        if not self.enabled:
            return
        self._total_start = time.perf_counter()
        self._stage_start = self._total_start
        self._marks.clear()

    def mark(self, stage_name: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        self._marks.append((stage_name, (now - self._stage_start) * 1000.0))
        self._stage_start = now

    def report(self) -> None:
        if not self.enabled:
            return
        total_ms = (time.perf_counter() - self._total_start) * 1000.0
        breakdown = " | ".join(f"{name}={ms:.1f}ms" for name, ms in self._marks)
        print(f"[TIMING] total={total_ms:.1f}ms | {breakdown}")


def reset_streaks() -> None:
    recognition_streak.clear()
    last_seen.clear()


def update_streak(name: str) -> int:
    now = time.monotonic()
    for person in list(last_seen):
        if now - last_seen[person] > RECOGNITION_RESET_SECONDS:
            recognition_streak.pop(person, None)
            last_seen.pop(person, None)
    if name == "UNKNOWN":
        return 0
    last_seen[name] = now
    recognition_streak[name] += 1
    return recognition_streak[name]


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/upload")
def upload():
    global latest_frame, last_result

    timer = _StageTimer(ENABLE_TIMING_LOG)
    timer.start()

    image = decode_jpeg(request.get_data(cache=False))
    timer.mark("decode")
    if image is None:
        return jsonify(error="Dữ liệu JPEG không hợp lệ"), 400

    faces = detector.detect(image)
    timer.mark("detect")
    presence_result = presence.select(image.shape, faces)
    timer.mark("presence")
    last_result = {
        "person": presence.current_name,
        "presence": presence_result.status,
        "anti_spoof": "WAITING",
        "live_score": 0.0,
        "challenge": "WAITING",
    }

    if not presence_result.valid:
        reset_streaks()
        if presence_result.primary_index is not None:
            draw_label(image, faces[presence_result.primary_index], presence_result.status, (0, 165, 255))
    else:
        box = faces[presence_result.primary_index]
        result = recognizer.recognize(image, box)
        timer.mark("recognize")
        presence.set_name(result.name)

        spoof_result = anti_spoof.predict(image, box) if ANTI_SPOOF_ENABLED else None
        timer.mark("anti_spoof")
        ai_live = bool(spoof_result and spoof_result.is_live) if ANTI_SPOOF_ENABLED else True
        ai_label = (
            f"{spoof_result.label} {spoof_result.live_score:.2f}"
            if spoof_result is not None
            else "AI OFF"
        )

        challenge_ok, challenge_text = (
            liveness.check(image, box, result.name)
            if LIVENESS_ENABLED and result.known
            else (not REQUIRE_ACTIVE_CHALLENGE, "CHALLENGE OFF")
        )
        timer.mark("liveness")
        streak = update_streak(result.name) if result.known and ai_live else 0

        access_ok = (
            result.known
            and ai_live
            and (challenge_ok or not REQUIRE_ACTIVE_CHALLENGE)
            and streak >= REQUIRED_STABLE_FRAMES
        )

        if not result.known:
            color = (0, 0, 255)
            label = f"UNKNOWN | {ai_label}"
        elif not ai_live:
            color = (0, 0, 255)
            label = f"{result.name} | {ai_label} | TU CHOI"
            reset_streaks()
        elif not challenge_ok and REQUIRE_ACTIVE_CHALLENGE:
            color = (0, 255, 255)
            label = f"{result.name} | {ai_label} | {challenge_text}"
        else:
            color = (0, 255, 0)
            label = f"{result.name} | {ai_label} | {streak}/{REQUIRED_STABLE_FRAMES}"

        if access_ok:
            opened, reason = door.open_for(result.name)
            if opened:
                database.add_log(result.name, "OPENED", result.distance, spoof_result.live_score)
                recognition_streak[result.name] = 0
            elif reason == "esp32_error":
                database.add_log(result.name, "ESP32_ERROR", result.distance, spoof_result.live_score)

        last_result = {
            "person": result.name,
            "presence": presence_result.status,
            "anti_spoof": spoof_result.label if spoof_result else "OFF",
            "live_score": spoof_result.live_score if spoof_result else 1.0,
            "challenge": challenge_text,
        }
        draw_label(image, box, label, color)

    fps = fps_counter.tick()
    cv2.putText(image, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(
        image,
        f"DOOR: {door.status()} | FACES: {len(faces)} | {last_result['presence']}",
        (10, max(30, image.shape[0] - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 0) if door.status() == "OPEN" else (255, 255, 255),
        2,
    )

    encoded = encode_jpeg(image, JPEG_QUALITY)
    timer.mark("encode")
    if encoded is None:
        return jsonify(error="Không encode được JPEG"), 500

    with frame_condition:
        latest_frame = encoded
        frame_condition.notify_all()

    timer.report()
    return Response(encoded, mimetype="image/jpeg")


def generate_frames():
    previous = None
    while True:
        with frame_condition:
            frame_condition.wait_for(lambda: latest_frame is not None and latest_frame is not previous, timeout=1)
            frame = latest_frame
        if frame is None:
            continue
        previous = frame
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


@app.get("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
def api_status():
    return jsonify(
        door=door.status(),
        known_encodings=len(recognizer.known_encodings),
        liveness_enabled=LIVENESS_ENABLED,
        anti_spoof_enabled=ANTI_SPOOF_ENABLED,
        anti_spoof_available=anti_spoof.available,
        anti_spoof_error=anti_spoof.load_error,
        **last_result,
    )


@app.get("/api/logs")
def api_logs():
    limit = min(max(request.args.get("limit", default=20, type=int), 1), 200)
    return jsonify(database.recent_logs(limit))


@app.post("/api/reload-dataset")
def reload_dataset():
    recognizer.load_dataset()
    model_ok = anti_spoof.reload()
    return jsonify(ok=True, known_encodings=len(recognizer.known_encodings), model_ok=model_ok)


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, threaded=True)
