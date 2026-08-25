"""
검출 방식 비교: 색상 기반(CircleDetector) vs YOLO(ONNX).

오버레이 데이터셋(data/images/overlay/labels.csv, 픽셀 정답)에 대해 두 방식을 돌려
- 검출 성공률(safe/next)
- 중심 오차(px), 반경 오차(px) 평균/중앙값
을 비교한다. YOLO ONNX가 없으면 색상 방식만 평가.

사용: python ml/compare_detectors.py
"""
import csv
import os

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(__file__))
OVL = os.path.join(BASE, "data", "images", "overlay")


def load_labels():
    p = os.path.join(OVL, "labels.csv")
    if not os.path.exists(p):
        return []
    return list(csv.DictReader(open(p, encoding="utf-8")))


def err(det, gt):
    """검출 원 dict{cx,cy,r} vs 정답(gx,gy,gr) → (중심오차, 반경오차) 또는 None."""
    if det is None:
        return None
    ce = ((det["cx"] - gt[0]) ** 2 + (det["cy"] - gt[1]) ** 2) ** 0.5
    re = abs(det["r"] - gt[2])
    return ce, re


def evaluate(name, detector, rows):
    hits_s = hits_n = total = 0
    ces, res = [], []
    for r in rows:
        img = cv2.imread(os.path.join(OVL, r["filename"]))
        if img is None:
            continue
        total += 1
        out = detector.detect_with_confidence(img)
        gs = (float(r["safe_cx"]), float(r["safe_cy"]), float(r["safe_r"]))
        es = err(out["safe"], gs)
        if es:
            hits_s += 1; ces.append(es[0]); res.append(es[1])
        if out.get("next") is not None and float(r["next_r"]) > 0:
            hits_n += 1
    if total == 0:
        print(f"[{name}] 평가할 이미지 없음"); return
    med = lambda a: float(np.median(a)) if a else float("nan")
    mean = lambda a: float(np.mean(a)) if a else float("nan")
    print(f"[{name}] 이미지 {total} | safe 성공 {hits_s}/{total} ({100*hits_s/total:.0f}%) "
          f"| next 성공 {hits_n}/{total}")
    print(f"        중심오차 px 평균 {mean(ces):.1f} / 중앙 {med(ces):.1f} | "
          f"반경오차 px 평균 {mean(res):.1f} / 중앙 {med(res):.1f}")


def main():
    rows = load_labels()
    if not rows:
        print("labels.csv 없음. make_map_overlays.py를 먼저 실행하세요."); return

    from app.services.circle_detector import CircleDetector
    evaluate("색상(HSV+Contour)", CircleDetector(), rows)

    from app.services.yolo_detector import get_yolo_detector
    y = get_yolo_detector()
    if y is not None:
        evaluate("YOLO(ONNX)", y, rows)
    else:
        print("[YOLO] ONNX 모델 없음 → 색상 방식만 평가. ml/export_onnx.py로 먼저 내보내세요.")


if __name__ == "__main__":
    import sys
    sys.path.append(BASE)
    main()
