"""
학습된 YOLO 가중치로 해충을 탐지하고, 강의 자료(6장 좌표·7장 JSON) 형식으로
결과를 저장하는 추론 스크립트.

사용 예:
    python3 predict.py --weights runs/detect/pests/weights/best.pt \
                       --source Pests/test/images
    python3 predict.py --weights best.pt --source 한장.jpg --conf 0.4

출력:
  - runs/detect/predict*/ : 바운딩박스가 그려진 결과 이미지
  - detections.json       : 각 탐지의 클래스 / 신뢰도 / 박스좌표 / 중심좌표
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pests YOLO 추론")
    p.add_argument("--weights", default="runs/detect/pests/weights/best.pt")
    p.add_argument("--source", required=True, help="이미지 파일 또는 폴더")
    p.add_argument("--conf", type=float, default=0.25, help="신뢰도 임계값")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="detections.json", help="JSON 저장 경로")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.weights).exists():
        raise SystemExit(f"가중치가 없습니다: {args.weights} (먼저 train.py 실행)")
    if args.device is None:
        args.device = "0" if torch.cuda.is_available() else "cpu"

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,        # 박스 그려진 이미지 저장
    )

    payload = []
    for r in results:
        dets = []
        for b in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            cls_id = int(b.cls[0])
            dets.append({
                "class_id": cls_id,
                "class_name": model.names[cls_id],
                "confidence": round(float(b.conf[0]), 4),
                # 6.1 바운딩박스 좌표 (좌상단/우하단, 픽셀)
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                # 6.2 중심좌표 (로봇 연계용)
                "center": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
            })
        payload.append({
            "image": Path(r.path).name,
            "count": len(dets),
            "detections": dets,
        })

    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = sum(p["count"] for p in payload)
    print(f"\n이미지 {len(payload)}장에서 탐지 {total}건 -> {args.out}")
    print("박스 이미지: runs/detect/predict* 폴더 확인")


if __name__ == "__main__":
    main()
