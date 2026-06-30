# weights/ — 모델 가중치

추론 서버(`app/server.py`)가 사용하는 YOLO 가중치를 두는 곳입니다.

| 파일 | 설명 |
|------|------|
| `pests.pt` | 해충 탐지 학습 모델 (Aphids·Thrips·Whiteflies) — 서버 기본값 |

## 서버가 찾는 경로

`app/server.py` 는 기본으로 `weights/pests.pt` 를 로드합니다.
다른 가중치를 쓰려면 환경변수로 지정하세요.

```bash
MODEL_PATH=weights/best.pt python app/server.py
```

`weights/pests.pt` 가 없으면 서버는 사전학습 `yolo11n.pt`(데모)로 자동 대체합니다.

## 직접 학습한 모델 넣기

Colab(`training/Pests_YOLO_Colab.ipynb`) 또는 `training/train.py` 로 학습하면
`runs/detect/pests/weights/best.pt` 가 생성됩니다. 이를 여기로 복사하세요.

```bash
cp runs/detect/pests/weights/best.pt weights/pests.pt
```
