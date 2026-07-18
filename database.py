from __future__ import annotations

import sqlite3
from pathlib import Path

from config import DATABASE_PATH
from utils import now_string


class AccessDatabase:
    def __init__(self, path: Path = DATABASE_PATH) -> None:
        self.path = Path(path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS access_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_name TEXT NOT NULL,
                    event_time TEXT NOT NULL,
                    result TEXT NOT NULL,
                    distance REAL,
                    live_score REAL
                )
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(access_logs)")}
            if "live_score" not in columns:
                db.execute("ALTER TABLE access_logs ADD COLUMN live_score REAL")

    def add_log(self, name: str, result: str, distance: float | None = None, live_score: float | None = None) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO access_logs(person_name, event_time, result, distance, live_score) VALUES (?, ?, ?, ?, ?)",
                (name, now_string(), result, distance, live_score),
            )

    def recent_logs(self, limit: int = 50) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM access_logs ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(row) for row in rows]
