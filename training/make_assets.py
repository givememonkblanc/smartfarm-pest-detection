"""강의 HTML 삽입용 예시 이미지 생성.

- assets/preview_grid.jpg : 원본 데이터셋 미리보기(2x3 그리드)
- assets/detect_example.jpg : 여러 클래스가 섞인 탐지 예시(박스+라벨)
- assets/class_<name>.jpg : 클래스별 대표 예시(박스+라벨)

박스는 정답 라벨(ground-truth)로 그립니다(학습 전 시연용).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("Pests/train")
IMG_D, LBL_D = ROOT / "images", ROOT / "labels"
OUT = Path("assets")
OUT.mkdir(exist_ok=True)

NAMES = ["Aphids", "Thrips", "Whiteflies"]
COLORS = [(46, 160, 67), (255, 143, 0), (33, 118, 220)]  # 초록/주황/파랑
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_labels(stem: str) -> list[tuple[int, float, float, float, float]]:
    txt = LBL_D / f"{stem}.txt"
    out = []
    for ln in txt.read_text().splitlines():
        if ln.strip():
            c, cx, cy, w, h = ln.split()
            out.append((int(c), float(cx), float(cy), float(w), float(h)))
    return out


def stats():
    """각 이미지의 (stem, 클래스별 박스수, 총박스, 평균박스면적) 수집."""
    rows = []
    for lbl in LBL_D.glob("*.txt"):
        boxes = load_labels(lbl.stem)
        if not boxes:
            continue
        per = [0, 0, 0]
        area = 0.0
        for c, _, _, w, h in boxes:
            per[c] += 1
            area += w * h
        rows.append((lbl.stem, per, len(boxes), area / len(boxes)))
    return rows


def draw(stem: str, out_path: Path, width: int = 680, labels: bool = True):
    img = Image.open(IMG_D / f"{stem}.jpg").convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    fsz = max(14, W // 36)
    try:
        font = ImageFont.truetype(FONT_PATH, fsz)
    except Exception:
        font = ImageFont.load_default()

    for c, cx, cy, w, h in load_labels(stem):
        x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
        x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
        col = COLORS[c]
        draw.rectangle([x1, y1, x2, y2], outline=col, width=max(2, W // 280))
        if not labels:
            continue
        label = NAMES[c]
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y1 - th - 4)
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=col)
        draw.text((x1 + 3, ty + 2), label, fill="white", font=font)

    # 리사이즈
    scale = width / W
    img = img.resize((width, int(H * scale)))
    img.save(out_path, quality=88)
    print(f"  저장: {out_path} ({img.size[0]}x{img.size[1]})")


def make_grid(stems: list[str], out_path: Path, cell: int = 320, cols: int = 3):
    rows = (len(stems) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * cell), (245, 249, 245))
    for i, stem in enumerate(stems):
        im = Image.open(IMG_D / f"{stem}.jpg").convert("RGB")
        im = im.resize((cell, cell))
        r, c = divmod(i, cols)
        canvas.paste(im, (c * cell, r * cell))
    canvas.save(out_path, quality=85)
    print(f"  저장: {out_path} ({canvas.size[0]}x{canvas.size[1]})")


def main():
    rows = stats()

    # 클래스별 대표: 박스가 적고(3~8개) 큼직하며, 거의 해당 클래스만 있는 이미지
    # -> 텍스트 라벨 없이 박스만 그려 깔끔하게(클래스명은 HTML 캡션으로)
    for c in range(3):
        cand = [r for r in rows
                if 3 <= r[1][c] <= 8           # 해당 클래스 박스 3~8개
                and r[2] <= r[1][c] + 1        # 다른 클래스 거의 없음
                and r[3] > 0.004]              # 박스가 충분히 큼
        cand.sort(key=lambda r: r[3], reverse=True)  # 큰 박스 우선
        if cand:
            draw(cand[0][0], OUT / f"class_{NAMES[c].lower()}.jpg", labels=False)

    # 탐지 예시: 2개 이상 클래스가 섞이고 총 박스 4~14개, 박스가 보기 좋은 크기
    mixed = [r for r in rows
             if sum(1 for v in r[1] if v > 0) >= 2 and 4 <= r[2] <= 14 and r[3] > 0.0015]
    mixed.sort(key=lambda r: r[3], reverse=True)
    if mixed:
        draw(mixed[0][0], OUT / "detect_example.jpg")

    # 원본 미리보기 그리드: 박스 적당(2~10개)한 다양한 이미지 6장
    preview = [r for r in rows if 2 <= r[2] <= 10 and r[3] > 0.001]
    preview.sort(key=lambda r: r[3], reverse=True)
    stems = [r[0] for r in preview[:30][::5]][:6]
    if len(stems) >= 6:
        make_grid(stems, OUT / "preview_grid.jpg")


if __name__ == "__main__":
    main()
