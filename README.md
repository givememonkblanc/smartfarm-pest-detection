# 스마트팜 해충 탐지 AI 서비스

농작물 잎을 **모바일로 촬영 → AI가 해충 탐지 → 웹 대시보드에서 확인**하는 전체 AI 서비스 실습 프로젝트입니다.
해충 3종(진딧물·총채벌레·가루이)을 YOLO 객체 탐지로 진단합니다.

## 📂 저장소 구성

| 폴더 | 내용 |
|------|------|
| `training/` | **학습 코드** — 데이터 변환·YOLO 학습·추론·Colab 노트북 |
| `app/` | **추론 서버 코드** — Flask + SQLite + 모바일/대시보드 화면 |
| `weights/` | **가중치** — `pests.pt` (학습된 해충 탐지 모델) |
| `assets/` | 문서용 예시 이미지 |

> **데이터셋(약 116MB)** 은 용량 때문에 레포가 아니라 **[Releases](../../releases)** 에 있습니다.

## 🚀 빠른 시작

### 1) 데이터셋 받기 (학습용)
```bash
curl -L -o pests.zip \
  https://github.com/givememonkblanc/smartfarm-pest-detection/releases/download/v1.0/Pests.v3i.coco.zip
mkdir -p Pests && unzip -q pests.zip -d Pests
```

### 2) 학습 (training/)
```bash
pip install -r training/requirements.txt
python training/prepare_dataset.py          # COCO → YOLO 변환 + data.yaml
python training/train.py                     # → runs/detect/pests/weights/best.pt
cp runs/detect/pests/weights/best.pt weights/pests.pt
```
> GPU 없이/Colab은 `training/Pests_YOLO_Colab.ipynb` 사용.

### 3) 추론 서버 실행 (app/)
```bash
pip install -r app/requirements.txt
cd app && python server.py          # http://localhost:8000
```
- `/` 모바일 촬영·진단 · `/dashboard` 웹 대시보드 · `/predict` 추론 API
- `weights/pests.pt` 가 있으면 해충을 탐지, 없으면 `yolo11n.pt`(데모)로 대체.

### 4) 배포 (Cloudflare Tunnel)
```bash
cloudflared tunnel --url http://localhost:8000   # 공개 https 주소 발급
```

## 📊 데이터셋

- 클래스 3종: `0 Aphids(진딧물)` · `1 Thrips(총채벌레)` · `2 Whiteflies(가루이)`
- 분할: train 1,493 / valid 442 / test 213 (총 2,148장, 640×640, COCO→YOLO)
- ⚠️ 진딧물 라벨이 많은 **클래스 불균형** — `prepare_dataset.py --oversample` 로 보강 가능

## 🧩 아키텍처

```
📱 모바일(capture.html) → POST /predict → 🧠 추론서버(Flask+YOLO) → 🗄️ SQLite
                                                      ↓
💻 대시보드(dashboard.html) ← GET /api/history ←──────┘
        전체를 ☁️ Cloudflare Tunnel 로 인터넷에 공개
```
