# 스마트팜 AI — 해충 탐지(Pests) 실습

7회차 스마트팜 객체탐지 강의 실습 자료입니다.
YOLO로 농작물 해충(**진딧물·총채벌레·가루이**)을 탐지하는 모델을 학습합니다.

## 📂 구성

| 파일 | 설명 |
|------|------|
| `07_스마트팜_AI_서비스_1.html` | 강의 자료(웹페이지) |
| `Pests_YOLO_Colab.ipynb` | **Colab 학습 노트북 (이거 하나로 끝)** |
| `prepare_dataset.py` | COCO → YOLO 변환 + `data.yaml` 생성 |
| `train.py` | YOLO 학습 |
| `predict.py` | 추론 + 좌표·JSON 출력 |
| `requirements.txt` | 의존성 |

> 데이터셋은 용량(약 116MB) 때문에 레포가 아니라 **[Releases](../../releases)** 에 있습니다.

## 🚀 가장 쉬운 방법 — Google Colab

1. `Pests_YOLO_Colab.ipynb` 를 [Colab](https://colab.research.google.com/) 에서 엽니다.
2. 런타임 → 런타임 유형 변경 → **GPU(T4)** 선택
3. 셀을 위에서부터 차례로 실행 (데이터 다운로드 → 변환 → 학습 → 추론)

## 💻 로컬에서 직접 하기

### 1) 데이터셋 받기 (Releases 에서)

```bash
# 레포 클론
git clone https://github.com/givememonkblanc/smartfarm-pest-detection.git
cd smartfarm-pest-detection

# 데이터셋 다운로드 (Release 에셋)
curl -L -o Pests.v3i.coco.zip \
  https://github.com/givememonkblanc/smartfarm-pest-detection/releases/download/v1.0/Pests.v3i.coco.zip
mkdir -p Pests && unzip -q Pests.v3i.coco.zip -d Pests
```

### 2) 환경 설치

```bash
pip install -r requirements.txt
```

### 3) 변환 → 학습 → 추론

```bash
python3 prepare_dataset.py          # COCO -> YOLO, data.yaml 생성
python3 train.py                    # runs/detect/pests/weights/best.pt 생성
python3 predict.py --weights runs/detect/pests/weights/best.pt \
                   --source Pests/test/images
```

## 📊 데이터셋 정보

- 출처: Roboflow (COCO 포맷, 이미지 640×640)
- 클래스: `0 Aphids(진딧물)` · `1 Thrips(총채벌레)` · `2 Whiteflies(가루이)`
- 분할: train 1,493 / valid 442 / test 213장
- ⚠️ Aphids 라벨이 압도적으로 많은 **클래스 불균형** 있음 (학습 시 참고)

### 클래스 불균형 보정 (선택)

소수 클래스(Thrips·Whiteflies)가 포함된 train 이미지를 복제해 비율을 보정합니다.

```bash
python3 prepare_dataset.py --oversample          # 최대 4배 복제
python3 prepare_dataset.py --oversample --max-dup 3
```

| 클래스 | 보정 전 | 보정 후 |
|--------|--------:|--------:|
| Aphids     | 18,088 | 41,587 |
| Thrips     |  3,275 | 13,100 |
| Whiteflies |  4,709 | 18,836 |

> `--oversample` 없이 다시 실행하면 원본(base) 상태로 되돌아갑니다(복제본 자동 정리).

## 🛠 참고

- GPU 권장. NVIDIA(CUDA)면 그대로 동작, Colab T4 권장.
- AMD ROCm(예: Radeon 8060S/gfx1151)에서 `HIP error: invalid device function` 이 나면
  `HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 train.py` 로 실행하세요.
