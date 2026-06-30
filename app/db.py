"""SQLite 저장소 — 탐지 결과 1건 = 테이블 1행.

추론 서버(server.py)가 사진을 추론한 결과를 여기에 저장하고,
웹 대시보드가 여기서 이력을 읽어 갑니다.
DB 파일(data.db)은 처음 실행할 때 자동으로 만들어집니다.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 컬럼명을 키로 접근 가능
    return conn


def init_db() -> None:
    """테이블이 없으면 생성(있으면 그대로)."""
    with _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT    NOT NULL,   -- 촬영/저장 시각
                image       TEXT    NOT NULL,   -- uploads/ 안의 파일명
                status      TEXT    NOT NULL,   -- '정상' 또는 '이상'
                pest_count  INTEGER NOT NULL,   -- 탐지된 해충 마릿수
                classes     TEXT    NOT NULL,   -- 클래스별 개수 (JSON 문자열)
                detections  TEXT    NOT NULL    -- 박스 상세 목록 (JSON 문자열)
            )
            """
        )


def insert(rec: dict) -> int:
    """탐지 결과 1건 저장 후 새 id 반환."""
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO detections (created_at, image, status, pest_count, classes, detections)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                rec["created_at"],
                rec["image"],
                rec["status"],
                rec["pest_count"],
                json.dumps(rec["classes"], ensure_ascii=False),
                json.dumps(rec["detections"], ensure_ascii=False),
            ),
        )
        return cur.lastrowid


def recent(limit: int = 50) -> list[dict]:
    """최근 기록을 최신순으로 반환(JSON 컬럼은 객체로 복원)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [
        {
            "id": r["id"],
            "created_at": r["created_at"],
            "image": r["image"],
            "status": r["status"],
            "pest_count": r["pest_count"],
            "classes": json.loads(r["classes"]),
            "detections": json.loads(r["detections"]),
        }
        for r in rows
    ]
