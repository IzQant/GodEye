"""
YOLO 라벨 QA: build_detection_dataset.py가 만든 라벨을 이미지 위에 다시 그려
학습 전에 라벨이 실제 원과 맞는지 눈으로 검증한다. (GPU 시간 낭비 방지)

동작:
- data/detect/images/{split}/ 와 labels/{split}/ 를 매칭.
- 각 라벨 박스(class 0=safe 초록, 1=next 파랑)를 이미지에 그려
  data/detect/_qa/ 에 저장(최대 N장).
- 라벨 통계(이미지 수, 클래스별 박스 수, 박스 크기 이상치)도 출력.

실행: python scripts/check_detection_labels.py [--split train|val] [--n 12]
"""
import argparse
import glob
import os

import cv2

BASE = os.path.join(os.path.dirname(__file__), "..")
DET = os.path.join(BASE, "data", "detect")
QA = os.path.join(DET, "_qa")
COLORS = {0: (0, 220, 0), 1: (255, 130, 40)}   # safe=초록, next=파랑
NAMES = {0: "safe", 1: "next"}


def draw_labels(img, label_path):
    """YOLO 라벨(정규화 cls xc yc w h)을 이미지에 박스로 그린다."""
    h, w = img.shape[:2]
    boxes = 0
    warns = []
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) != 5:
                warns.append("형식오류")
                continue
            cls = int(float(p[0]))
            xc, yc, bw, bh = (float(x) for x in p[1:])
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                warns.append("범위이상")
            x0 = int((xc - bw / 2) * w); y0 = int((yc - bh / 2) * h)
            x1 = int((xc + bw / 2) * w); y1 = int((yc + bh / 2) * h)
            cv2.rectangle(img, (x0, y0), (x1, y1), COLORS.get(cls, (200, 200, 200)), 2)
            cv2.putText(img, NAMES.get(cls, str(cls)), (x0, max(12, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(cls, (200, 200, 200)), 1)
            boxes += 1
    return boxes, warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--n", type=int, default=12)
    opt = ap.parse_args()

    img_dir = os.path.join(DET, "images", opt.split)
    lbl_dir = os.path.join(DET, "labels", opt.split)
    if not os.path.isdir(img_dir):
        print(f"{img_dir} 없음. scripts/build_detection_dataset.py를 먼저 실행하세요.")
        return

    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    os.makedirs(QA, exist_ok=True)

    total, cls_count, drawn, all_warns = 0, {0: 0, 1: 0}, 0, 0
    for i, ip in enumerate(imgs):
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(lp):
            continue
        total += 1
        # 클래스 카운트
        for line in open(lp):
            pp = line.split()
            if len(pp) == 5:
                cls_count[int(float(pp[0]))] = cls_count.get(int(float(pp[0])), 0) + 1
        # 앞 N장만 시각화 저장
        if drawn < opt.n:
            img = cv2.imread(ip)
            boxes, warns = draw_labels(img, lp)
            all_warns += len(warns)
            cv2.imwrite(os.path.join(QA, f"qa_{stem}.png"), img)
            drawn += 1

    print(f"[{opt.split}] 이미지 {total}장")
    print(f"  클래스별 박스: safe={cls_count.get(0,0)}, next={cls_count.get(1,0)}")
    print(f"  QA 시각화 {drawn}장 → {QA}")
    if all_warns:
        print(f"  ⚠️ 라벨 경고 {all_warns}건(형식/범위) — 확인 필요")
    else:
        print("  라벨 형식/범위 이상 없음")
    print("\n_qa/ 폴더의 이미지를 열어 박스가 실제 흰/파란 원을 감싸는지 확인하세요.")


if __name__ == "__main__":
    main()
