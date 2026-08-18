"""
사진에서 지도 영역(사각형)을 자동 추정 — '자동 감지' 버튼용 (best-effort).

사진 속 지도는 대체로 가장 큰 사각형 영역이다. 에지 검출 → 윤곽선 근사로
4점 사각형을 찾는다. 못 찾으면 안쪽 여백 기본값을 돌려주고, 사용자가 보정한다.
정확한 검출은 보장하지 않는다(수동 조정 폴백 전제).
"""
import cv2
import numpy as np


def _order(pts):
    """4점을 TL, TR, BR, BL 순서로 정렬."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return [tuple(map(float, p)) for p in (tl, tr, br, bl)]


def detect_map_corners(image_bgr):
    """
    BGR 이미지 → 지도로 추정되는 4점 [(x,y)*4] (TL,TR,BR,BL), 원본 픽셀 좌표.
    실패 시 안쪽 12% 여백의 기본 사각형 반환.
    """
    h, w = image_bgr.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        area = cv2.contourArea(c)
        if area < 0.15 * img_area:
            break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            best = approx.reshape(4, 2)
            break

    if best is not None:
        return _order(best)

    # 폴백: 안쪽 12% 여백 기본 사각형
    mx, my = w * 0.12, h * 0.12
    return [(mx, my), (w - mx, my), (w - mx, h - my), (mx, h - my)]
