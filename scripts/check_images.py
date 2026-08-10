"""
Day 22 완료 기준 점검: 테스트 이미지 확보 현황 확인.

data/images/real/ 와 synthetic/ 의 이미지를 세고, 각 파일이 실제로 열리는지
(손상 여부) 검증한다. real 이미지가 20장 이상이면 완료 기준 통과.

실행: python scripts/check_images.py
"""
import glob
import os

import cv2

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "images")
EXTS = ("*.png", "*.jpg", "*.jpeg")


def scan(folder: str):
    paths = []
    for ext in EXTS:
        paths += glob.glob(os.path.join(folder, ext))
    ok, broken, sizes = 0, [], []
    for p in sorted(paths):
        img = cv2.imread(p)
        if img is None:
            broken.append(os.path.basename(p))
        else:
            ok += 1
            sizes.append(f"{img.shape[1]}x{img.shape[0]}")
    return ok, broken, sizes


def main():
    for name in ("real", "synthetic"):
        folder = os.path.join(BASE, name)
        ok, broken, sizes = scan(folder)
        uniq = sorted(set(sizes))
        print(f"[{name}] 정상 {ok}장" + (f", 손상 {len(broken)}장 {broken}" if broken else ""))
        if uniq:
            print(f"    해상도 종류: {uniq}")

    real_ok, _, _ = scan(os.path.join(BASE, "real"))
    print()
    if real_ok >= 20:
        print(f"✅ Day 22 완료 기준 통과: real 이미지 {real_ok}장 (>=20)")
    else:
        print(f"⏳ real 이미지 {real_ok}장 — 20장 이상 필요. "
              "그동안 synthetic으로 Day 23~24 검출 개발 가능.")


if __name__ == "__main__":
    main()
