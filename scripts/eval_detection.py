"""
Day 24 완료 기준 확인: 합성 이미지에서 원 검출 정확도 측정.

synthetic/labels.csv의 정답(중심/반경)과 detect_circles 결과를 비교한다.
- 중심 오차(px), 반경 오차(px)
- 성공 판정: 중심 오차와 반경 오차가 각각 이미지 한 변의 2% 이내
  (해상도가 다양하므로 절대 px가 아니라 크기 대비 상대 기준 사용)
- 해상도별 성공률도 함께 출력

실행: python scripts/eval_detection.py
"""
import csv
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.services.circle_detector import detect_circles

SYN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images", "synthetic")
TOL_RATIO = 0.02  # 이미지 한 변의 2%


def main():
    labels_path = os.path.join(SYN_DIR, "labels.csv")
    rows = list(csv.DictReader(open(labels_path, encoding="utf-8")))

    results = []       # (size, safe_ok, next_ok, safe_cerr, safe_rerr, next_cerr, next_rerr)
    per_size = {}      # size -> [성공 개수(원 2개 모두), 전체]

    for r in rows:
        size = int(r["size"])
        img = cv2.imread(os.path.join(SYN_DIR, r["filename"]))
        det = detect_circles(img)
        tol = size * TOL_RATIO

        def check(pred, cx, cy, rr):
            if pred is None:
                return False, None, None
            cerr = float(np.hypot(pred["cx"] - cx, pred["cy"] - cy))
            rerr = abs(pred["r"] - rr)
            return (cerr <= tol and rerr <= tol), cerr, rerr

        safe_ok, s_ce, s_re = check(det["safe"], int(r["safe_cx"]), int(r["safe_cy"]), int(r["safe_r"]))
        next_ok, n_ce, n_re = check(det["next"], int(r["next_cx"]), int(r["next_cy"]), int(r["next_r"]))

        results.append((size, safe_ok, next_ok, s_ce, s_re, n_ce, n_re))
        both = safe_ok and next_ok
        per_size.setdefault(size, [0, 0])
        per_size[size][0] += int(both)
        per_size[size][1] += 1

    n = len(results)
    safe_rate = sum(r[1] for r in results) / n
    next_rate = sum(r[2] for r in results) / n
    both_rate = sum(r[1] and r[2] for r in results) / n

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"총 {n}장 (허용오차 = 한 변의 {TOL_RATIO*100:.0f}%)")
    print(f"흰 원(safe) 성공률: {safe_rate*100:.1f}%  | 평균 중심오차 {mean([r[3] for r in results]):.1f}px, "
          f"반경오차 {mean([r[4] for r in results]):.1f}px")
    print(f"파란 원(next) 성공률: {next_rate*100:.1f}%  | 평균 중심오차 {mean([r[5] for r in results]):.1f}px, "
          f"반경오차 {mean([r[6] for r in results]):.1f}px")
    print(f"둘 다 성공: {both_rate*100:.1f}%")

    print("\n해상도별 (둘 다 성공/전체):")
    for size in sorted(per_size):
        ok, tot = per_size[size]
        print(f"  {size:>4}px: {ok}/{tot} ({ok/tot*100:.0f}%)")

    if both_rate >= 0.8:
        print(f"\n✅ Day 24 완료 기준 통과: 대부분 이미지에서 원 중심/반경 추출 성공 ({both_rate*100:.0f}%)")
    else:
        print(f"\n❌ 성공률 {both_rate*100:.0f}% — 검출/파라미터 점검 필요")


if __name__ == "__main__":
    main()
