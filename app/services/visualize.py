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


def draw_heatmap(base_bgr, current_map, phase, map_name, hmodel):
    """
    다음 원 중심의 확률 히트맵을 지도 위에 얹어 PNG로 반환 (임시 연결).
    현재 원(청록)도 함께 그린다.
    """
    size = MAP_SIZES_CM[map_name]
    img = base_bgr.copy()
    h, w = img.shape[:2]
    cx, cy, cr = current_map["x"], current_map["y"], current_map["radius"]

    X, Y, D = hmodel.predict_grid(cx, cy, cr, int(phase), res=120)
    Dn = D / (D.max() + 1e-9)

    # 히트맵 색상화(JET) + 픽셀 bbox 계산
    heat = cv2.applyColorMap((Dn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    x0 = max(0, int((cx - 1.2 * cr) / size * w)); x1 = min(w, int((cx + 1.2 * cr) / size * w))
    y0 = max(0, int((cy - 1.2 * cr) / size * h)); y1 = min(h, int((cy + 1.2 * cr) / size * h))
    if x1 > x0 and y1 > y0:
        heat_r = cv2.resize(heat, (x1 - x0, y1 - y0))
        alpha = cv2.resize((Dn * 0.6).astype(np.float32), (x1 - x0, y1 - y0))[..., None]
        region = img[y0:y1, x0:x1].astype(np.float32)
        img[y0:y1, x0:x1] = (region * (1 - alpha) + heat_r.astype(np.float32) * alpha).astype(np.uint8)

    # 현재 원 테두리
    t = max(2, w // 400)
    cv2.circle(img, (int(cx / size * w), int(cy / size * h)), int(cr / size * w), CYAN, t)

    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()
