"""
Day 22 보조: 합성 미니맵 생성기.

실제 스크린샷을 모으는 동안, Day 23~24의 원 검출(HSV 필터 + Contour) 코드를
개발·검증할 수 있도록 '미니맵 비슷한' 합성 이미지를 만든다.
- 어두운/노이즈 배경 위에 흰 원(현재 안전지대)과 파란 원(다음 자기장)을 그린다.
- 각 이미지의 정답 좌표(중심/반경)를 synthetic/labels.csv에 기록한다.
  → Day 24 검출 결과와 이 정답을 비교해 정확도를 잴 수 있다.

주의: 실제 게임 UI가 아니므로 검출 파라미터 튜닝의 '출발점'일 뿐,
      최종 검증은 real/ 스크린샷으로 해야 한다.

실행: python scripts/make_synthetic_minimaps.py [개수]   (기본 24)
"""
import csv
import os
import sys

import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images", "synthetic")

# 검출 대상 색 (OpenCV는 BGR 순서)
WHITE_BGR = (255, 255, 255)   # 현재 안전지대(흰 원)
BLUE_BGR = (255, 130, 40)     # 다음 자기장(파란 원) — 파랑 계열

# 실제 스크린샷의 다양한 미니맵 크기를 흉내내기 위한 해상도 후보.
# 검출 코드가 특정 크기에 과적합되지 않도록 여러 크기를 섞는다.
SIZES = [256, 384, 512, 640, 768, 896, 1024]


def make_one(rng, size=512):
    """합성 미니맵 1장 생성 → (이미지, 정답 dict). size는 정사각 한 변(px)."""
    # 원 테두리 두께는 해상도에 비례시켜 큰 이미지에서 너무 얇아 보이지 않게
    thickness = max(2, size // 256)
    # 배경: 어두운 회녹색 + 약한 노이즈 (지형 느낌, 검출 난이도 부여)
    bg = rng.integers(40, 70, size=(size, size, 3), dtype=np.uint8)
    img = bg.copy()

    # 현재 안전지대(흰 원): 화면 중앙 근처, 큰 반경
    safe_r = int(rng.integers(size * 0.28, size * 0.42))
    safe_c = (int(rng.integers(safe_r, size - safe_r)),
              int(rng.integers(safe_r, size - safe_r)))

    # 다음 자기장(파란 원): 흰 원 안쪽에 '완전히 포함'되어야 한다 (PUBG 규칙).
    # 즉 두 중심 거리 + next_r <= safe_r  ⟺  중심 오프셋 벡터 크기 <= (safe_r - next_r).
    # 축별로 따로 뽑으면 대각선 합이 이 한계를 넘을 수 있으므로,
    # 반지름 max_off인 '원판' 안에서 (각도, 거리)로 오프셋을 뽑아 포함을 보장한다.
    next_r = int(safe_r * rng.uniform(0.5, 0.85))
    max_off = safe_r - next_r
    angle = rng.uniform(0, 2 * np.pi)
    dist = max_off * np.sqrt(rng.uniform(0, 1))  # sqrt: 원판 내 균등 분포
    next_c = (int(round(safe_c[0] + dist * np.cos(angle))),
              int(round(safe_c[1] + dist * np.sin(angle))))

    # 원은 '테두리'로 그린다 (게임 미니맵처럼)
    cv2.circle(img, safe_c, safe_r, WHITE_BGR, thickness=thickness)
    cv2.circle(img, next_c, next_r, BLUE_BGR, thickness=thickness)

    label = {
        "size": size,
        "safe_cx": safe_c[0], "safe_cy": safe_c[1], "safe_r": safe_r,
        "next_cx": next_c[0], "next_cy": next_c[1], "next_r": next_r,
    }
    return img, label


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    rows = []
    for i in range(n):
        size = int(rng.choice(SIZES))       # 이미지마다 해상도를 다르게
        img, label = make_one(rng, size=size)
        fname = f"synthetic_{i:03d}.png"
        cv2.imwrite(os.path.join(OUT_DIR, fname), img)
        rows.append({"filename": fname, **label})

    with open(os.path.join(OUT_DIR, "labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "size", "safe_cx", "safe_cy", "safe_r",
                                          "next_cx", "next_cy", "next_r"])
        w.writeheader()
        w.writerows(rows)

    sizes_used = sorted(set(r["size"] for r in rows))
    print(f"합성 이미지 {n}장 + labels.csv 생성 → {OUT_DIR}")
    print(f"해상도 종류: {sizes_used}")


if __name__ == "__main__":
    main()
