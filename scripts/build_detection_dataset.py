"""
확장 1단계: 오버레이 → YOLO 검출 데이터셋 자동 생성.

make_map_overlays.py가 만든 data/images/overlay/의 이미지와 labels.csv를 읽어
YOLO 형식(정규화 바운딩박스) 라벨로 변환한다. 원의 (중심, 반경)을 알므로
라벨을 자동으로 붙일 수 있다(수작업 라벨링 불필요).

클래스: 0 = safe(현재 흰 원), 1 = next(다음 파란 원)
바운딩박스: 원을 감싸는 정사각형 [cx-r, cy-r, cx+r, cy+r] → 이미지 밖은 클램프.

출력:
  data/detect/
    images/{train,val}/*.png
    labels/{train,val}/*.txt   (YOLO: cls x_center y_center w h, 모두 0~1 정규화)
    data.yaml

매치 단위 train/val 분리(누수 방지).

실행: python scripts/build_detection_dataset.py
주의: data/images/overlay/ 를 먼저 make_map_overlays.py로 생성해 두어야 함.
"""
import csv
import os
import shutil
import sys

import cv2
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..")
OVL = os.path.join(BASE, "data", "images", "overlay")
OUT = os.path.join(BASE, "data", "detect")
VAL_RATIO = 0.2
SEED = 42


def bbox_yolo(cx, cy, r, w, h):
    """원(cx,cy,r) → 이미지(w,h) 기준 YOLO 정규화 박스. 이미지 밖은 클램프."""
    x0 = max(0.0, cx - r); y0 = max(0.0, cy - r)
    x1 = min(float(w), cx + r); y1 = min(float(h), cy + r)
    if x1 <= x0 or y1 <= y0:
        return None
    xc = (x0 + x1) / 2 / w
    yc = (y0 + y1) / 2 / h
    bw = (x1 - x0) / w
    bh = (y1 - y0) / h
    return xc, yc, bw, bh


def main():
    labels_csv = os.path.join(OVL, "labels.csv")
    if not os.path.exists(labels_csv):
        print("data/images/overlay/labels.csv 없음. make_map_overlays.py를 먼저 실행하세요.")
        return

    rows = list(csv.DictReader(open(labels_csv, encoding="utf-8")))
    if not rows:
        print("labels.csv 가 비어 있습니다.")
        return

    # 매치 단위 split
    rng = np.random.default_rng(SEED)
    match_ids = sorted({r["match_id"] for r in rows})
    rng.shuffle(match_ids)
    n_val = max(1, int(len(match_ids) * VAL_RATIO))
    val_ids = set(match_ids[:n_val])

    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)

    made, skipped = 0, 0
    for r in rows:
        img_path = os.path.join(OVL, r["filename"])
        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]
        split = "val" if r["match_id"] in val_ids else "train"

        lines = []
        for cls, (cxk, cyk, rk) in ((0, ("safe_cx", "safe_cy", "safe_r")),
                                    (1, ("next_cx", "next_cy", "next_r"))):
            r_ = float(r[rk])
            if r_ <= 0:
                continue
            box = bbox_yolo(float(r[cxk]), float(r[cyk]), r_, w, h)
            if box is None:
                continue
            lines.append(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")

        if not lines:
            skipped += 1
            continue

        stem = os.path.splitext(r["filename"])[0]
        shutil.copy(img_path, os.path.join(OUT, "images", split, r["filename"]))
        with open(os.path.join(OUT, "labels", split, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        made += 1

    # data.yaml
    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(
            f"path: {os.path.abspath(OUT)}\n"
            "train: images/train\n"
            "val: images/val\n"
            "nc: 2\n"
            "names: ['safe', 'next']\n"
        )

    print(f"YOLO 데이터셋 생성: {made}장 (스킵 {skipped}), val 매치 {n_val}/{len(match_ids)}")
    print(f"경로: {OUT}  (data.yaml 포함)")
    print("다음: 데스크탑(GPU)에서 python ml/train_yolo.py")


if __name__ == "__main__":
    main()
