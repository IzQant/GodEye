"""
Day 26 작업: 미니맵 픽셀 좌표 → 맵 기준 좌표(텔레메트리 cm) 변환.

CV로 검출한 원의 픽셀 좌표를 예측 모델이 쓰는 맵 좌표계로 바꿔야
검출 결과를 그대로 /api/predict에 넣을 수 있다.

두 가지 변환:
1) AffineTransform (스케일+오프셋)
   미니맵이 맵을 '똑바로' 축소해 보여주는 경우(가장 흔함).
   map_x = ox + sx * px,  map_y = oy + sy * py
   기준점 2개 이상(픽셀↔맵)이 있으면 축별 최소제곱으로 sx,ox,sy,oy를 구한다.
2) HomographyTransform
   미니맵이 회전/기울어져 원근 왜곡이 있을 때. 대응점 4개 이상으로 3x3 행렬 추정.

가장 신뢰할 수 있는 방법은 '알려진 기준점'으로 보정(calibration)하는 것이므로,
맵 크기 상수에 의존하지 않고 기준점으로 변환을 맞춘다.
(MAP_SIZES는 참고용 근사값 — 전체 맵 미니맵일 때의 편의 함수에만 사용)
"""
import numpy as np

try:
    import cv2
except ImportError:  # 어파인만 쓰면 cv2 없이도 동작
    cv2 = None

# 참고용: 맵 좌표계 한 변의 대략적 크기(cm). 정확 보정은 기준점으로 하는 것을 권장.
MAP_SIZES_CM = {
    "Erangel": 816000, "Miramar": 816000, "Taego": 816000,
    "Deston": 816000, "Rondo": 816000,
    "Sanhok": 408000, "Vikendi": 612000,
    "Karakin": 204000, "Paramo": 306000,
}


class AffineTransform:
    """map = offset + scale * pixel (x, y 각각 독립)."""

    def __init__(self, sx, ox, sy, oy):
        self.sx, self.ox, self.sy, self.oy = sx, ox, sy, oy

    def apply(self, px, py):
        return self.ox + self.sx * px, self.oy + self.sy * py

    def apply_circle(self, cx, cy, r):
        """중심은 변환하고, 반경은 스케일 크기(평균)로 환산."""
        mx, my = self.apply(cx, cy)
        mr = r * (abs(self.sx) + abs(self.sy)) / 2
        return {"x": mx, "y": my, "radius": mr}


def fit_affine(pixel_pts, map_pts) -> AffineTransform:
    """
    대응점들로 축별 최소제곱 직선 적합 → 스케일/오프셋 추정.
    pixel_pts, map_pts: [(x, y), ...] 같은 길이(>=2).
    """
    px = np.array(pixel_pts, dtype=float)
    mp = np.array(map_pts, dtype=float)
    if len(px) < 2:
        raise ValueError("기준점이 2개 이상 필요합니다.")

    # x축: map_x = sx*px_x + ox  (1차 최소제곱)
    sx, ox = np.polyfit(px[:, 0], mp[:, 0], 1)
    sy, oy = np.polyfit(px[:, 1], mp[:, 1], 1)
    return AffineTransform(float(sx), float(ox), float(sy), float(oy))


def full_map_affine(image_w, image_h, map_name) -> AffineTransform:
    """전체 맵 미니맵(축 정렬)일 때, 맵 크기 상수로 간단히 변환 생성(편의 함수)."""
    size = MAP_SIZES_CM.get(map_name)
    if size is None:
        raise ValueError(f"알 수 없는 맵: {map_name}")
    return AffineTransform(size / image_w, 0.0, size / image_h, 0.0)


class HomographyTransform:
    """원근 왜곡까지 처리하는 3x3 호모그래피 변환."""

    def __init__(self, H):
        self.H = H

    def apply(self, px, py):
        v = self.H @ np.array([px, py, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])


def fit_homography(pixel_pts, map_pts) -> HomographyTransform:
    """대응점 4개 이상으로 호모그래피 추정 (cv2 필요)."""
    if cv2 is None:
        raise RuntimeError("호모그래피에는 OpenCV(cv2)가 필요합니다.")
    src = np.array(pixel_pts, dtype=np.float32)
    dst = np.array(map_pts, dtype=np.float32)
    if len(src) < 4:
        raise ValueError("호모그래피는 대응점 4개 이상 필요합니다.")
    H, _ = cv2.findHomography(src, dst)
    return HomographyTransform(H)
