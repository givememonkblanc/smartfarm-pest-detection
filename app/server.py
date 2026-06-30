"""해충 탐지 추론 서버 (Flask).

전체 흐름(아키텍처):
    [모바일 촬영 capture.html]  --사진 업로드-->  POST /predict
        -> YOLO 추론 -> SQLite 저장 -> JSON 응답
    [웹 대시보드 dashboard.html]  --조회-->  GET /api/history

실행:
    pip install -r requirements.txt
    python3 server.py            # http://0.0.0.0:8000

환경변수:
    MODEL_PATH  사용할 가중치(.pt) 경로. 기본 ../weights/pests.pt
                없으면 yolo11n.pt(사전학습)로 대체하여 파이프라인만 시연.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from ultralytics import YOLO

import db

BASE = Path(__file__).resolve().parent
UPLOAD_DIR = BASE / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# --- 모델 로드 (서버 시작 시 1회만! 요청마다 로드하면 매우 느림) ---
MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE.parent / "weights" / "pests.pt"))
if not Path(MODEL_PATH).exists():
    print(f"[경고] 가중치 없음: {MODEL_PATH}\n      -> yolo11n.pt(사전학습)로 대체합니다. "
          "학습한 pests.pt 를 weights/ 에 두면 해충을 탐지합니다.")
    MODEL_PATH = "yolo11n.pt"
print(f"[모델] {MODEL_PATH} 로드 중...")
model = YOLO(MODEL_PATH)
CONF = float(os.environ.get("CONF", "0.25"))
# DEVICE 미지정 시 자동 선택. GPU 문제(예: 일부 AMD ROCm)나 CPU 강제 시 DEVICE=cpu
DEVICE = os.environ.get("DEVICE") or None

app = Flask(__name__)
db.init_db()


@app.route("/")
def index():
    """모바일 촬영 화면 (서비스 진입점)."""
    return render_template("capture.html")


@app.route("/dashboard")
def dashboard():
    """웹 결과 확인 대시보드."""
    return render_template("dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():
    """사진 1장을 받아 추론 -> 저장 -> 결과 JSON 반환."""
    if "image" not in request.files:
        return jsonify({"error": "image 파일이 필요합니다"}), 400

    f = request.files["image"]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"{ts}.jpg"
    f.save(UPLOAD_DIR / fname)

    # 추론
    r = model.predict(str(UPLOAD_DIR / fname), conf=CONF, device=DEVICE, verbose=False)[0]
    dets, counts = [], {}
    for b in r.boxes:
        cid = int(b.cls[0])
        name = model.names[cid]
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        dets.append({
            "class": name,
            "confidence": round(float(b.conf[0]), 3),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "center": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
        })
        counts[name] = counts.get(name, 0) + 1

    record = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "image": fname,
        "status": "이상" if dets else "정상",   # 해충이 1마리라도 있으면 '이상'
        "pest_count": len(dets),
        "classes": counts,
        "detections": dets,
    }
    record["id"] = db.insert(record)
    return jsonify(record)


@app.route("/api/history")
def history():
    """최근 탐지 이력 (대시보드가 호출)."""
    return jsonify(db.recent(limit=50))


@app.route("/uploads/<path:fname>")
def uploads(fname):
    """업로드된 원본 이미지 제공."""
    return send_from_directory(UPLOAD_DIR, fname)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    # use_reloader=False: 리로더가 모델을 두 번 로드하는 것을 방지
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
