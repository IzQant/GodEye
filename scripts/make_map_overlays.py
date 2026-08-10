"""
텔레메트리 + 실제 맵 이미지 오버레이 생성기.

이미 만든 zones_dataset.csv(단계별 자기장 좌표, cm)를 실제 맵 이미지 위에
그려서, 실제 미니맵과 유사하면서 정답 라벨이 완벽한 학습/평가 이미지를 만든다.

동작:
- data/maps/<map>.png 가 있는 맵의 행만 처리.
- 맵 좌표(cm) → 픽셀: pixel = coord / MAP_SIZE_CM * 이미지_한변.
- 흰 원(safety) + 파란 원(next=poison)을 테두리로 그린다.
- data/images/overlay/ 에 이미지 저장 + labels.csv(픽셀 정답) 기록.

실행: python scripts/make_map_overlays.py
"""
import csv
import os
import sys

import cv2
import numpy as np
import pandas as pd
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.services.coordinate_transform import MAP_SIZES_CM

BASE = os.path.join(os.path.dirname(__file__), "..")
CSV_PATH = os.path.join(BASE, "data", "processed", "zones_dataset.csv")
MAPS_DIR = os.path.join(BASE, "data", "maps")
OUT_DIR = os.path.join(BASE, "data", "images", "overlay")

WHITE_BGR = (255, 255, 255)   # 현재 안전지대
BLUE_BGR = (255, 130, 40)     # 다음 자기장
FLIP_Y = False                # 맵 이미지의 y축이 반대면 True


def map_to_px(coord_cm, size_cm, img_side, flip=False):
    p = coord_cm / size_cm * img_side
    return (img_side - p) if flip else p


def load_maps():
    """data/maps/의 맵 이미지를 {맵이름: (이미지, 한변px)}로 로드."""
    maps = {}
    for name in MAP_SIZES_CM:
        path = os.path.join(MAPS_DIR, f"{name.lower()}.png")
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                maps[name] = img
    return maps


def main():
    df = pd.read_csv(CSV_PATH)
    maps = load_maps()

    if not maps:
        print("data/maps/에 맵 이미지가 없습니다. README를 참고해 <map>.png를 넣어주세요.")
        needed = sorted(df["map"].unique())
        print(f"현재 데이터셋에 필요한 맵: {needed}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    # 재실행 시 이전 결과물(overlay_*.png)을 먼저 정리해 orphan 파일이 쌓이지 않게 한다.
    # (파일명이 match_id+phase 기반으로 고정이라 대부분 덮어써지지만, 맵 구성이 바뀐
    #  경우를 대비해 시작 시 한 번 비운다. 권한 문제로 못 지우면 그냥 건너뜀.)
    removed = 0
    for old in glob.glob(os.path.join(OUT_DIR, "overlay_*.png")):
        try:
            os.remove(old)
            removed += 1
        except OSError:
            pass
    if removed:
        print(f"이전 오버레이 {removed}장 정리")

    rows_out = []
    for _, r in df.iterrows():
        mp = r["map"]
        if mp not in maps:
            continue
        size_cm = MAP_SIZES_CM[mp]
        base = maps[mp]
        h, w = base.shape[:2]
        img = base.copy()
        thickness = max(2, w // 400)

        def to_px(x, y):
            return (int(map_to_px(x, size_cm, w)),
                    int(map_to_px(y, size_cm, h, FLIP_Y)))

        sc = to_px(r["safety_x"], r["safety_y"])
        nc = to_px(r["poison_x"], r["poison_y"])
        sr = int(r["safety_radius"] / size_cm * w)
        nr = int(r["poison_radius"] / size_cm * w)

        cv2.circle(img, sc, sr, WHITE_BGR, thickness)
        cv2.circle(img, nc, nr, BLUE_BGR, thickness)

        # 고정 파일명(match_id+phase 기반) → 재실행 시 같은 파일을 덮어써 orphan 방지
        fname = f"overlay_{mp.lower()}_{r['match_id']}_p{int(r['phase'])}.png"
        cv2.imwrite(os.path.join(OUT_DIR, fname), img)
        rows_out.append({
            "filename": fname, "map": mp, "phase": int(r["phase"]),
            "match_id": r["match_id"],
            "safe_cx": sc[0], "safe_cy": sc[1], "safe_r": sr,
            "next_cx": nc[0], "next_cy": nc[1], "next_r": nr,
        })

    with open(os.path.join(OUT_DIR, "labels.csv"), "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=["filename", "map", "phase", "match_id",
                                             "safe_cx", "safe_cy", "safe_r",
                                             "next_cx", "next_cy", "next_r"])
        wcsv.writeheader()
        wcsv.writerows(rows_out)

    used = sorted(set(r["map"] for r in rows_out))
    print(f"오버레이 {len(rows_out)}장 생성 → {OUT_DIR}")
    print(f"사용한 맵: {used}")
    missing = sorted(set(df['map'].unique()) - set(maps.keys()))
    if missing:
        print(f"이미지 없어 건너뛴 맵: {missing}")


if __name__ == "__main__":
    main()
