"""
YOLO 해충 탐지 모델 학습 스크립트 (Pests: Aphids / Thrips / Whiteflies).

전제: 먼저 `python3 prepare_dataset.py` 를 실행해
      Pests/data.yaml 과 YOLO 라벨이 생성되어 있어야 합니다.

사용 예:
    python3 train.py                       # 기본값으로 학습
    python3 train.py --model yolo11s.pt --epochs 150 --imgsz 640
    python3 train.py --device cpu          # GPU 없이 CPU 학습

학습이 끝나면 가중치는 runs/detect/pests/weights/best.pt 에 저장됩니다.
이 best.pt 를 predict.py 또는 강의 자료의 모델 선택 드롭다운에 연결하면 됩니다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pests YOLO 학습")
    p.add_argument("--data", default="Pests/data.yaml", help="data.yaml 경로")
    p.add_argument("--model", default="yolo11n.pt",
                   help="사전학습 가중치 (yolo11n/s/m.pt 또는 yolov8n.pt 등)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16,
                   help="배치 크기 (-1 이면 GPU 메모리에 맞춰 자동)")
    p.add_argument("--device", default=None,
                   help="'0', 'cpu' 등. 미지정 시 GPU 있으면 자동 사용")
    p.add_argument("--name", default="pests", help="실행 이름 (runs/detect/<name>)")
    p.add_argument("--patience", type=int, default=20,
                   help="개선 없을 때 조기 종료까지 대기 에폭")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"data.yaml 이 없습니다: {data_path}\n"
            f"-> 먼저 `python3 prepare_dataset.py` 를 실행하세요."
        )

    if args.device is None:
        args.device = "0" if torch.cuda.is_available() else "cpu"
    print(f"학습 장치: {args.device} (cuda available={torch.cuda.is_available()})")
    print(f"베이스 모델: {args.model} · epochs={args.epochs} · imgsz={args.imgsz}")

    model = YOLO(args.model)  # 사전학습 가중치 로드 (없으면 자동 다운로드)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=args.patience,
        # 데이터가 적당량(≈2k)이라 가벼운 증강 위주
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, flipud=0.0,
        mosaic=1.0,
    )

    # 검증 세트 성능 출력
    metrics = model.val(data=str(data_path), split="val", device=args.device)
    print("\n=== 검증 성능 ===")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")

    best = Path("runs/detect") / args.name / "weights" / "best.pt"
    print(f"\n최종 가중치: {best}")
    print("추론 예:  python3 predict.py --weights "
          f"{best} --source Pests/test/images")


if __name__ == "__main__":
    main()
