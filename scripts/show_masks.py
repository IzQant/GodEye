"""
Day 23 완료 기준 확인: HSV 마스크가 원을 눈에 띄게 분리하는지 시각화.

합성 이미지 1장을 골라 원본 | 흰 마스크 | 파란 마스크를 가로로 붙여 저장하고,
각 마스크의 흰 픽셀 수를 출력한다(둘 다 0보다 크면 분리 성공).

실행: python scripts/show_masks.py [파일명]
      (파일명 생략 시 synthetic 폴더의 첫 이미지)
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.services.circle_detector import make_masks

SYN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images", "synthetic")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "figures", "hsv_masks.png")


def main():
    if len(sys.argv) > 1:
        path = os.path.join(SYN_DIR, sys.argv[1])
    else:
        path = sorted(glob.glob(os.path.join(SYN_DIR, "*.png")))[0]

    img = cv2.imread(path)
    if img is None:
        print(f"이미지를 열 수 없음: {path}")
        return

    white_mask, blue_mask = make_masks(img)
    print(f"파일: {os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})")
    print(f"흰 마스크 픽셀: {int((white_mask > 0).sum())}")
    print(f"파란 마스크 픽셀: {int((blue_mask > 0).sum())}")

    # 마스크를 3채널로 바꿔 원본과 가로로 붙인다
    w3 = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
    b3 = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
    montage = np.hstack([img, w3, b3])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, montage)
    print(f"시각화 저장: {OUT_PATH} (원본 | 흰 마스크 | 파란 마스크)")

    if (white_mask > 0).sum() > 0 and (blue_mask > 0).sum() > 0:
        print("✅ Day 23 완료 기준 통과: 색상 마스크로 원 영역 분리 확인")
    else:
        print("❌ 한쪽 마스크가 비어 있음 — HSV 범위 조정 필요")


if __name__ == "__main__":
    main()
