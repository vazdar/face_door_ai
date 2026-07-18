from __future__ import annotations

import threading
import time

import requests

from config import (
    DOOR_OPEN_DURATION,
    ESP32_DOOR_IP,
    ESP32_OPEN_PATH,
    PERSON_COOLDOWN_SECONDS,
    REQUEST_TIMEOUT,
)


class DoorController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = "CLOSED"
        self._opened_at = 0.0
        self._last_person_open: dict[str, float] = {}

    def _refresh_state(self, now: float) -> None:
        if self._state == "OPEN" and now - self._opened_at >= DOOR_OPEN_DURATION:
            self._state = "CLOSED"

    def status(self) -> str:
        with self._lock:
            self._refresh_state(time.monotonic())
            return self._state

    def open_for(self, person_name: str) -> tuple[bool, str]:
        now = time.monotonic()
        with self._lock:
            self._refresh_state(now)
            if person_name == "UNKNOWN":
                return False, "unknown"
            if self._state != "CLOSED":
                return False, "door_busy"
            last_open = self._last_person_open.get(person_name, 0.0)
            if now - last_open < PERSON_COOLDOWN_SECONDS:
                return False, "person_cooldown"

            try:
                response = requests.get(
                    f"http://{ESP32_DOOR_IP}{ESP32_OPEN_PATH}", timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                print(f"[DOOR] Lỗi kết nối ESP32: {exc}")
                return False, "esp32_error"

            self._state = "OPEN"
            self._opened_at = now
            self._last_person_open[person_name] = now
            print(f"[DOOR] Đã mở cho {person_name}")
            return True, "opened"
