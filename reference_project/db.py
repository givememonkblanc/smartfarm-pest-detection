"""SQLite 저장소 — ERD 명세(USERS / FARMS / ANALYSIS_HISTORIES / DETECTION_RESULTS) 구현.

정답 프로젝트용. 명세서의 4개 테이블과 관계를 그대로 따른다.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "service.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                role         TEXT NOT NULL DEFAULT 'FARMER',   -- FARMER / ADMIN
                name         TEXT NOT NULL,
                phone_number TEXT,
                created_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS farms (
                farm_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL REFERENCES users(user_id),
                farm_name  TEXT NOT NULL,
                location   TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS analysis_histories (
                history_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id            INTEGER NOT NULL REFERENCES users(user_id),
                original_image_url TEXT NOT NULL,
                result_image_url   TEXT,
                status             TEXT NOT NULL,              -- NORMAL / ABNORMAL
                admin_memo         TEXT,
                device_info        TEXT,
                captured_at        TEXT NOT NULL,
                created_at         TEXT NOT NULL,
                saved              INTEGER NOT NULL DEFAULT 0  -- 0:임시 1:최종저장
            );

            CREATE TABLE IF NOT EXISTS detection_results (
                detection_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                history_id       INTEGER NOT NULL REFERENCES analysis_histories(history_id),
                pest_type        TEXT NOT NULL,                -- APHID / THRIPS / WHITEFLY
                confidence_score REAL NOT NULL,
                bounding_box     TEXT NOT NULL                 -- JSON {"x","y","w","h"}
            );
            """
        )


def seed_demo() -> int:
    """데모용 농업인 1명 + 농장 1개. 이미 있으면 그대로 두고 user_id 반환."""
    from datetime import datetime
    now = datetime.now().isoformat(timespec="seconds")
    with _conn() as c:
        row = c.execute("SELECT user_id FROM users ORDER BY user_id LIMIT 1").fetchone()
        if row:
            return row["user_id"]
        uid = c.execute(
            "INSERT INTO users (role, name, phone_number, created_at) VALUES (?,?,?,?)",
            ("FARMER", "김농부", "010-1234-5678", now),
        ).lastrowid
        c.execute(
            "INSERT INTO farms (user_id, farm_name, location, created_at) VALUES (?,?,?,?)",
            (uid, "행복농원 1구역", "전남 나주시", now),
        )
        return uid


# ---------- 분석 이력 ----------
def insert_history(rec: dict) -> int:
    with _conn() as c:
        return c.execute(
            "INSERT INTO analysis_histories "
            "(user_id, original_image_url, result_image_url, status, admin_memo, device_info, captured_at, created_at, saved)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (rec["user_id"], rec["original_image_url"], rec.get("result_image_url"),
             rec["status"], rec.get("admin_memo"), rec.get("device_info"),
             rec["captured_at"], rec["created_at"], rec.get("saved", 0)),
        ).lastrowid


def insert_detection(history_id: int, pest_type: str, score: float, bbox: dict) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO detection_results (history_id, pest_type, confidence_score, bounding_box)"
            " VALUES (?,?,?,?)",
            (history_id, pest_type, score, json.dumps(bbox)),
        )


def get_history(history_id: int) -> dict | None:
    with _conn() as c:
        h = c.execute("SELECT * FROM analysis_histories WHERE history_id=?", (history_id,)).fetchone()
        if not h:
            return None
        dets = c.execute(
            "SELECT pest_type, confidence_score, bounding_box FROM detection_results WHERE history_id=?",
            (history_id,),
        ).fetchall()
    out = dict(h)
    out["detections"] = [
        {"pest_type": d["pest_type"], "confidence_score": d["confidence_score"],
         "bounding_box": json.loads(d["bounding_box"])}
        for d in dets
    ]
    return out


def mark_saved(history_id: int) -> None:
    with _conn() as c:
        c.execute("UPDATE analysis_histories SET saved=1 WHERE history_id=?", (history_id,))


def update_memo(history_id: int, memo: str) -> None:
    with _conn() as c:
        c.execute("UPDATE analysis_histories SET admin_memo=? WHERE history_id=?", (memo, history_id))


def list_histories(date_from=None, date_to=None, pest=None, status=None, limit=100, offset=0) -> list[dict]:
    """필터(날짜·해충·상태) 지원 이력 리스트(최신순). 해충 필터는 detection_results 조인."""
    sql = ["SELECT DISTINCT h.* FROM analysis_histories h"]
    params, where = [], []
    if pest:
        sql.append("JOIN detection_results d ON d.history_id = h.history_id")
        where.append("d.pest_type = ?"); params.append(pest)
    if status:
        where.append("h.status = ?"); params.append(status)
    if date_from:
        where.append("h.captured_at >= ?"); params.append(date_from)
    if date_to:
        where.append("h.captured_at <= ?"); params.append(date_to + " 23:59:59")
    if where:
        sql.append("WHERE " + " AND ".join(where))
    sql.append("ORDER BY h.history_id DESC LIMIT ? OFFSET ?")
    params += [limit, offset]
    with _conn() as c:
        rows = c.execute(" ".join(sql), params).fetchall()
        # 각 이력의 해충 요약
        out = []
        for r in rows:
            pests = c.execute(
                "SELECT pest_type, COUNT(*) n FROM detection_results WHERE history_id=? GROUP BY pest_type",
                (r["history_id"],),
            ).fetchall()
            d = dict(r)
            d["pest_summary"] = {p["pest_type"]: p["n"] for p in pests}
            out.append(d)
    return out


def stats() -> dict:
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) n FROM analysis_histories").fetchone()["n"]
        abnormal = c.execute("SELECT COUNT(*) n FROM analysis_histories WHERE status='ABNORMAL'").fetchone()["n"]
        by_pest = c.execute(
            "SELECT pest_type, COUNT(*) n FROM detection_results GROUP BY pest_type ORDER BY n DESC"
        ).fetchall()
        # 일자별 추이(최근 14일)
        trend = c.execute(
            "SELECT substr(captured_at,1,10) d, COUNT(*) n FROM analysis_histories "
            "GROUP BY d ORDER BY d DESC LIMIT 14"
        ).fetchall()
    return {
        "total": total,
        "abnormal": abnormal,
        "abnormal_ratio": round(abnormal / total * 100, 1) if total else 0.0,
        "by_pest": {r["pest_type"]: r["n"] for r in by_pest},
        "trend": [{"date": r["d"], "count": r["n"]} for r in reversed(trend)],
    }
