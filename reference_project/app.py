"""AI 해충 탐지 서비스 — 정답(reference) 구현.

명세 충실 구현:
 - 화면: M-01(촬영/업로드) · M-02(분석결과) · W-01(대시보드) · W-02(상세)
 - API : /api/v1/analysis/{upload,result,save} · /api/v1/dashboard/{stats,history,export} · memo
 - DB  : USERS / FARMS / ANALYSIS_HISTORIES / DETECTION_RESULTS (db.py, ERD)

실행:
    pip install -r requirements.txt
    python app.py          # http://localhost:8100
"""
from __future__ import annotations

import datetime
import io
import os
from pathlib import Path

from flask import (Flask, jsonify, render_template, request,
                   send_file, send_from_directory)
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

import db

BASE = Path(__file__).resolve().parent
UPLOAD_DIR = BASE / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# 모델 1회 로드 (학습한 weights/pests.pt 우선, 없으면 yolo11n 데모)
MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE.parent / "weights" / "pests.pt"))
if not Path(MODEL_PATH).exists():
    print(f"[경고] {MODEL_PATH} 없음 → yolo11n.pt(데모)로 대체")
    MODEL_PATH = "yolo11n.pt"
model = YOLO(MODEL_PATH)
CONF = float(os.environ.get("CONF", "0.25"))
DEVICE = os.environ.get("DEVICE") or None

# 모델 클래스명 → 명세 pest_type 코드
PEST_CODE = {"Aphids": "APHID", "Thrips": "THRIPS", "Whiteflies": "WHITEFLY"}
PEST_KO = {"APHID": "진딧물", "THRIPS": "총채벌레", "WHITEFLY": "가루이"}
PEST_COLOR = {"APHID": (46, 160, 67), "THRIPS": (255, 143, 0), "WHITEFLY": (33, 118, 220)}

app = Flask(__name__)
db.init_db()
DEMO_USER = db.seed_demo()


# ============== 화면 라우트 ==============
@app.route("/")
def m01():
    return render_template("m01_capture.html")


@app.route("/result/<int:hid>")
def m02(hid):
    return render_template("m02_result.html", hid=hid)


@app.route("/dashboard")
def w01():
    return render_template("w01_dashboard.html")


@app.route("/dashboard/<int:hid>")
def w02(hid):
    return render_template("w02_detail.html", hid=hid)


@app.route("/uploads/<path:fname>")
def uploads(fname):
    return send_from_directory(UPLOAD_DIR, fname)


# ============== 분석 API ==============
@app.route("/api/v1/analysis/upload", methods=["POST"])
def analysis_upload():
    """F-03/F-04: 이미지 업로드 → YOLO 추론 → BBox 결과이미지 + 이력/탐지 저장."""
    if "image" not in request.files:
        return jsonify({"error": "image 필요"}), 400
    f = request.files["image"]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    orig_name = f"orig_{ts}.jpg"
    f.save(UPLOAD_DIR / orig_name)

    # 추론
    r = model.predict(str(UPLOAD_DIR / orig_name), conf=CONF, device=DEVICE, verbose=False)[0]
    img = Image.open(UPLOAD_DIR / orig_name).convert("RGB")
    drawer = ImageDraw.Draw(img)
    W, H = img.size
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(14, W // 36))
    except Exception:
        font = ImageFont.load_default()

    detections = []
    for b in r.boxes:
        name = model.names[int(b.cls[0])]
        code = PEST_CODE.get(name, name.upper())
        score = round(float(b.conf[0]), 4)
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        bbox = {"x": round(x1, 1), "y": round(y1, 1),
                "w": round(x2 - x1, 1), "h": round(y2 - y1, 1)}
        detections.append({"pest_type": code, "confidence_score": score, "bounding_box": bbox})
        col = PEST_COLOR.get(code, (229, 57, 53))
        drawer.rectangle([x1, y1, x2, y2], outline=col, width=max(2, W // 300))
        label = f"{PEST_KO.get(code, code)} {int(score*100)}%"
        tb = drawer.textbbox((0, 0), label, font=font)
        drawer.rectangle([x1, y1 - (tb[3]-tb[1]) - 5, x1 + (tb[2]-tb[0]) + 6, y1], fill=col)
        drawer.text((x1 + 3, y1 - (tb[3]-tb[1]) - 4), label, fill="white", font=font)

    result_name = f"result_{ts}.jpg"
    img.save(UPLOAD_DIR / result_name, quality=88)

    status = "ABNORMAL" if detections else "NORMAL"
    now = datetime.datetime.now().isoformat(timespec="seconds")
    hid = db.insert_history({
        "user_id": DEMO_USER,
        "original_image_url": f"/uploads/{orig_name}",
        "result_image_url": f"/uploads/{result_name}",
        "status": status,
        "device_info": request.form.get("device_info") or request.headers.get("User-Agent", "")[:100],
        "captured_at": request.form.get("captured_at") or now,
        "created_at": now,
        "saved": 0,
    })
    for d in detections:
        db.insert_detection(hid, d["pest_type"], d["confidence_score"], d["bounding_box"])

    return jsonify({
        "history_id": hid,
        "status": status,
        "result_image_url": f"/uploads/{result_name}",
        "original_image_url": f"/uploads/{orig_name}",
        "detections": detections,
    })


@app.route("/api/v1/analysis/result/<int:hid>")
def analysis_result(hid):
    h = db.get_history(hid)
    return (jsonify(h), 200) if h else (jsonify({"error": "not found"}), 404)


@app.route("/api/v1/analysis/save", methods=["POST"])
def analysis_save():
    """F-05: 분석 결과 최종 저장 확정."""
    hid = (request.get_json(silent=True) or {}).get("history_id") or request.form.get("history_id")
    if not hid:
        return jsonify({"error": "history_id 필요"}), 400
    db.mark_saved(int(hid))
    return jsonify({"history_id": int(hid), "saved": True})


# ============== 대시보드 API ==============
@app.route("/api/v1/dashboard/stats")
def dashboard_stats():
    return jsonify(db.stats())


def _filters():
    return dict(
        date_from=request.args.get("date_from"),
        date_to=request.args.get("date_to"),
        pest=request.args.get("pest"),
        status=request.args.get("status"),
    )


@app.route("/api/v1/dashboard/history")
def dashboard_history():
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))
    rows = db.list_histories(limit=limit, offset=offset, **_filters())
    return jsonify(rows)


@app.route("/api/v1/dashboard/history/<int:hid>")
def dashboard_history_detail(hid):
    h = db.get_history(hid)
    return (jsonify(h), 200) if h else (jsonify({"error": "not found"}), 404)


@app.route("/api/v1/dashboard/history/<int:hid>/memo", methods=["PUT"])
def dashboard_memo(hid):
    """F-11: 관리자 메모 업데이트."""
    memo = (request.get_json(silent=True) or {}).get("memo", "")
    db.update_memo(hid, memo)
    return jsonify({"history_id": hid, "memo": memo})


@app.route("/api/v1/dashboard/export")
def dashboard_export():
    """F-12: 필터된 이력을 .xlsx 로 추출."""
    import openpyxl
    rows = db.list_histories(limit=10000, **_filters())
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "분석이력"
    ws.append(["ID", "촬영일시", "판정", "탐지해충", "관리자메모", "기기"])
    for r in rows:
        pests = ", ".join(f"{PEST_KO.get(k,k)}×{v}" for k, v in r["pest_summary"].items()) or "-"
        ws.append([r["history_id"], r["captured_at"],
                   "정상" if r["status"] == "NORMAL" else "이상",
                   pests, r.get("admin_memo") or "", r.get("device_info") or ""])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="pest_history.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8100"))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
