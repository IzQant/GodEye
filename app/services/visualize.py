"""
Day 32 작업: 예측 결과를 지도 이미지 위에 오버레이해 PNG로 렌더.

- 현재 원(흰), 예측 다음 원(파랑), 예측 불확실성 반경(옅은 파랑 점선 대용 얇은 원)을 그린다.
- 맵 좌표(cm) → 픽셀 변환: pixel = map_coord / 맵크기(cm) * 이미지_한변.
  (full_map_affine의 역방향. 전체 지도 이미지를 base로 가정.)
"""
import cv2
import numpy as np

from app.services.coordinate_transform import MAP_SIZES_CM

WHITE = (255, 255, 255)
BLUE = (255, 130, 40)
CYAN = (230, 200, 120)


def draw_overlay(base_bgr, current_map, predicted_map, confidence_radius, map_name):
    """
    base_bgr: 배경 이미지(전체 지도 또는 업로드 스크린샷)
    current_map/predicted_map: {x,y,radius} 맵 좌표(cm). None이면 생략.
    반환: PNG 바이트.
    """
    size = MAP_SIZES_CM.get(map_name)
    img = base_bgr.copy()
    h, w = img.shape[:2]
    t = max(2, w // 400)

    def to_px(mx, my):
        return (int(mx / size * w), int(my / size * h))

    def r_px(mr):
        return max(1, int(mr / size * w))

    if size is not None:
        if current_map:
            cv2.circle(img, to_px(current_map["x"], current_map["y"]),
                       r_px(current_map["radius"]), WHITE, t)
        if predicted_map:
            pc = to_px(predicted_map["x"], predicted_map["y"])
            cv2.circle(img, pc, r_px(predicted_map["radius"]), BLUE, t)
            # 예측 중심 표시(작은 십자)
            cv2.drawMarker(img, pc, BLUE, cv2.MARKER_CROSS, max(8, w // 60), t)
            # 불확실성 반경(예측 중심 주변, 얇게)
            if confidence_radius:
                cv2.circle(img, pc, r_px(confidence_radius), CYAN, max(1, t // 2))

    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def blank_base(side=1024):
    """맵 이미지가 없을 때 쓰는 중립 배경(회색)."""
    return np.full((side, side, 3), 60, np.uint8)
