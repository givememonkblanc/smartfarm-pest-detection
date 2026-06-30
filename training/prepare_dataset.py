"""
Roboflow COCO 포맷(Pests.v3i.coco) → YOLO 포맷 변환 스크립트.

Roboflow에서 'COCO'로 내려받은 데이터셋은 split별로
    <split>/_annotations.coco.json  +  <split>/*.jpg
구조입니다. ultralytics(YOLO) 학습은 다음 구조를 기대합니다.
    <split>/images/*.jpg
    <split>/labels/*.txt   (class cx cy w h, 0~1 정규화)

이 스크립트는 위 구조로 변환하고 data.yaml 까지 생성합니다.
한 번 더 실행해도 안전(idempotent)하도록 작성했습니다.

사용법:
    python3 prepare_dataset.py                     # 기본 폴더 ./Pests
    python3 prepare_dataset.py ./Pests
    python3 prepare_dataset.py --oversample        # 클래스 불균형 보정(train 오버샘플링)
"""
from __future__ import annotations

import argparse
import math
import json
import shutil
from collections import Counter
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = ["train", "valid", "test"]
OVERSAMPLE_TAG = ".os"  # 오버샘플 복제본 파일명 표식 (재실행 시 정리용)


def convert_split(split_dir: Path, id2cls: dict[int, int]) -> tuple[int, int]:
    """한 split을 YOLO 구조로 변환. (이미지 수, 라벨 수) 반환."""
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"  [skip] {ann_path} 없음")
        return 0, 0

    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    images = {im["id"]: im for im in coco["images"]}

    # image_id -> [YOLO 라벨 라인]
    labels: dict[int, list[str]] = {im_id: [] for im_id in images}
    for a in coco["annotations"]:
        cls = id2cls.get(a["category_id"])
        if cls is None:  # supercategory placeholder 등은 건너뜀
            continue
        im = images[a["image_id"]]
        w, h = im["width"], im["height"]
        x, y, bw, bh = a["bbox"]  # COCO: 좌상단 x,y + 폭,높이 (절대 px)
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h
        # 0~1 범위로 클램프 (경계 라벨 안전장치)
        cx, cy, nw, nh = (min(max(v, 0.0), 1.0) for v in (cx, cy, nw, nh))
        labels[a["image_id"]].append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    img_out = split_dir / "images"
    lbl_out = split_dir / "labels"
    img_out.mkdir(exist_ok=True)
    lbl_out.mkdir(exist_ok=True)

    # 이전 오버샘플 복제본 정리 -> 항상 base 상태에서 시작 (재현성)
    for d in (img_out, lbl_out):
        for p in list(d.glob(f"*{OVERSAMPLE_TAG}*")):
            p.unlink()

    # 이미지 이동 (split 폴더 직속 jpg -> images/)
    moved = 0
    for p in list(split_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            shutil.move(str(p), str(img_out / p.name))
            moved += 1

    # 라벨 작성 (이미지가 images/ 든 직속이든 모두 대응)
    name2id = {Path(im["file_name"]).name: im_id for im_id, im in images.items()}
    n_lbl = 0
    for img_path in img_out.iterdir():
        im_id = name2id.get(img_path.name)
        if im_id is None:
            continue
        lines = labels.get(im_id, [])
        (lbl_out / f"{img_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_lbl += 1

    n_img = sum(1 for p in img_out.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"  [{split_dir.name}] 이미지 {n_img}장 (이번에 이동 {moved}) · 라벨 {n_lbl}개")
    return n_img, n_lbl


def build_class_map(root: Path) -> tuple[dict[int, int], list[str]]:
    """모든 split의 annotation에서 실제 등장하는 category만 클래스로 채택."""
    used_ids: set[int] = set()
    id2name: dict[int, str] = {}
    for split in SPLITS:
        ann = root / split / "_annotations.coco.json"
        if not ann.exists():
            continue
        coco = json.loads(ann.read_text(encoding="utf-8"))
        for c in coco["categories"]:
            id2name[c["id"]] = c["name"]
        for a in coco["annotations"]:
            used_ids.add(a["category_id"])
    if not used_ids:
        raise SystemExit("annotation에서 사용된 카테고리를 찾지 못했습니다.")

    ordered = sorted(used_ids)  # coco id 오름차순 -> YOLO 0..n-1
    id2cls = {cid: i for i, cid in enumerate(ordered)}
    names = [id2name[cid] for cid in ordered]
    return id2cls, names


def class_instance_counts(split_dir: Path, nc: int) -> Counter:
    """split 라벨에서 클래스별 인스턴스(박스) 수를 셈."""
    cnt = Counter()
    lbl_dir = split_dir / "labels"
    if not lbl_dir.exists():
        return cnt
    for txt in lbl_dir.glob("*.txt"):
        for line in txt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cnt[int(line.split()[0])] += 1
    return cnt


def oversample_train(split_dir: Path, names: list[str], max_dup: int = 4) -> None:
    """소수 클래스가 포함된 train 이미지를 복제해 인스턴스 비율을 보정.

    각 클래스 c의 가중치 = target / count_c (target=최다 클래스 수).
    이미지의 복제 배수 = 그 이미지에 등장하는 클래스들의 가중치 최댓값(올림),
    단 max_dup 으로 상한. 표준적인 oversampling 기법으로, 데이터가 적은
    수업용에 적합합니다(완전한 균형은 아니며 소수 클래스 표현을 키웁니다).
    """
    img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
    nc = len(names)

    # 재실행 안전: 이전 오버샘플 복제본(*.osN.*) 먼저 제거
    removed = 0
    for d in (img_dir, lbl_dir):
        for p in list(d.glob(f"*{OVERSAMPLE_TAG}*")):
            p.unlink()
            removed += 1
    if removed:
        print(f"  [oversample] 기존 복제본 {removed}개 정리")

    before = class_instance_counts(split_dir, nc)
    if not before:
        print("  [oversample] 라벨이 없어 건너뜀")
        return
    target = max(before.values())
    weight = {c: target / before.get(c, 1) for c in range(nc)}

    created = 0
    for lbl in list(lbl_dir.glob("*.txt")):
        if OVERSAMPLE_TAG in lbl.name:
            continue
        classes = {int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()}
        if not classes:
            continue
        factor = min(max_dup, math.ceil(max(weight[c] for c in classes)))
        if factor <= 1:
            continue
        # 원본 이미지 찾기
        img = next((p for p in img_dir.glob(f"{lbl.stem}.*")
                    if p.suffix.lower() in IMG_EXTS), None)
        if img is None:
            continue
        for k in range(1, factor):  # factor-1 개의 추가 복제본
            stem = f"{lbl.stem}{OVERSAMPLE_TAG}{k}"
            shutil.copy(img, img_dir / f"{stem}{img.suffix}")
            shutil.copy(lbl, lbl_dir / f"{stem}.txt")
            created += 1

    after = class_instance_counts(split_dir, nc)
    print(f"  [oversample] 복제 이미지 {created}장 추가")
    for c in range(nc):
        print(f"    {names[c]:<12} {before.get(c,0):>6} -> {after.get(c,0):>6}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Roboflow COCO -> YOLO 변환")
    ap.add_argument("root", nargs="?", default="Pests", help="데이터셋 폴더 (기본 ./Pests)")
    ap.add_argument("--oversample", action="store_true",
                    help="train 세트 오버샘플링으로 클래스 불균형 보정")
    ap.add_argument("--max-dup", type=int, default=4,
                    help="이미지당 최대 복제 배수 상한 (기본 4)")
    cli = ap.parse_args()

    root = Path(cli.root).resolve()
    if not root.exists():
        raise SystemExit(f"데이터셋 폴더가 없습니다: {root}")

    print(f"데이터셋: {root}")
    id2cls, names = build_class_map(root)
    print(f"클래스 {len(names)}종: {names}")
    print(f"매핑(coco_id -> yolo_idx): {id2cls}")

    counts = Counter()
    for split in SPLITS:
        split_dir = root / split
        if split_dir.exists():
            n_img, _ = convert_split(split_dir, id2cls)
            counts[split] = n_img

    if cli.oversample:
        print("클래스 불균형 보정(train 오버샘플링):")
        oversample_train(root / "train", names, max_dup=cli.max_dup)
        counts["train"] = sum(1 for p in (root / "train" / "images").iterdir()
                              if p.suffix.lower() in IMG_EXTS)

    data_yaml = root / "data.yaml"
    yaml_lines = [
        f"path: {root}",
        f"train: train/images",
        f"val: valid/images",
        f"test: test/images",
        "",
        f"nc: {len(names)}",
        f"names: {names}",
        "",
    ]
    data_yaml.write_text("\n".join(yaml_lines), encoding="utf-8")
    print(f"\ndata.yaml 생성 완료 -> {data_yaml}")
    print(f"split별 이미지 수: {dict(counts)}")


if __name__ == "__main__":
    main()
