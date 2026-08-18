"""
특징 공학(feature engineering): 전환쌍에 '지오메트리 특징'을 추가한다.

왜 이 컬럼들인가 (자기장 이동에 실제 신호가 되는 것):
- pos_x_norm, pos_y_norm : 현재 원 중심의 맵 내 상대 위치(0~1). 가장자리/중앙 구분.
- dist_center_norm       : 맵 중앙에서의 거리(반경 대비). 가장자리 원일수록 안쪽으로 당겨짐.
- angle_to_center        : 현재 중심 → 맵 중앙 방향(라디안). 이동이 중앙 쪽으로 치우치는 경향.
- prev_dx_n, prev_dy_n   : 직전 이동(모멘텀)을 맵 크기로 정규화. 이전 방향이 이어지는 경향.
- radius_norm            : 현재 반경/맵크기. 단계 진행도(원이 작을수록 후반).

추가 과정(개념):
1) 원본 좌표(cm)는 스케일이 커서 그대로 쓰기보다 맵 크기로 정규화해 0~1로 만든다.
2) '위치' 뿐 아니라 '중앙과의 관계(거리·방향)'를 명시적 컬럼으로 넣어 모델이
   지형/가장자리 효과를 학습하기 쉽게 돕는다.
3) 시퀀스 정보(직전 이동)를 컬럼으로 펼쳐 트리 모델도 모멘텀을 쓰게 한다.
"""
import numpy as np

from app.services.coordinate_transform import MAP_SIZES_CM

# 추가되는 특징 컬럼 이름
ENGINEERED = ["pos_x_norm", "pos_y_norm", "dist_center_norm",
              "angle_to_center", "prev_dx_n", "prev_dy_n", "radius_norm"]


def add_engineered(pairs):
    """전환쌍 DataFrame에 지오메트리/모멘텀 특징 컬럼을 추가해 반환."""
    df = pairs.copy()
    size = df["map"].map(MAP_SIZES_CM).astype(float)
    cx = df["safety_x"] / size
    cy = df["safety_y"] / size
    df["pos_x_norm"] = cx
    df["pos_y_norm"] = cy
    # 맵 중앙(0.5, 0.5)과의 거리
    dxc, dyc = 0.5 - cx, 0.5 - cy
    df["dist_center_norm"] = np.sqrt(dxc ** 2 + dyc ** 2)
    df["angle_to_center"] = np.arctan2(dyc, dxc)
    df["prev_dx_n"] = df["prev_dx"] / size
    df["prev_dy_n"] = df["prev_dy"] / size
    df["radius_norm"] = df["safety_radius"] / size
    return df
