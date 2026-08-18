"""
사진 보정: 사용자가 찍은 지도 네 모서리 → 정사각형(전체 지도)으로 원근 변환.

사진(모니터 촬영/여백 있음/기울어짐)에서는 '이미지 전체 = 전체 지도' 가정이 깨진다.
사용자가 지도의 네 모서리(TL, TR, BR, BL)를 지정하면, 그 사각형을
정사각 이미지로 원근 보정(warp)해서 '깨끗한 전체 지도'를 만든다.
이후 좌표 변환(full_map_affine)이 정상 동작한다.
"""
import cv2
import numpy as np


def warp_map(image_bgr, corners, out_size: int = 1024):
    """
    corners: [(x,y) x4] 원본 이미지 픽셀 좌표, 순서 TL, TR, BR, BL.
    반환: (정사각 보정 이미지, 3x3 변환행렬 M)  — M은 원본→보정 좌표 매핑.
    """
    src = np.array(corners, dtype=np.float32)
    dst = np.array([[0, 0], [out_size, 0], [out_size, out_size], [0, out_size]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image_bgr, M, (out_size, out_size))
    return warped, M


def warp_point(M, x, y):
    """원본 픽셀 (x,y) → 보정 이미지 픽셀."""
    v = M @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def warp_circle(M, cx, cy, r):
    """
    원본에서 찍은 원(중심 cx,cy, 반경 r) → 보정 이미지 좌표.
    원근 보정 후엔 완전한 원이 아닐 수 있어, 중심과 상/우 가장자리를 변환해
    반경을 평균으로 근사한다.
    """
    wcx, wcy = warp_point(M, cx, cy)
    ex, ey = warp_point(M, cx + r, cy)   # 오른쪽 가장자리
    tx, ty = warp_point(M, cx, cy + r)   # 아래쪽 가장자리
    r1 = ((ex - wcx) ** 2 + (ey - wcy) ** 2) ** 0.5
    r2 = ((tx - wcx) ** 2 + (ty - wcy) ** 2) ** 0.5
    return {"cx": wcx, "cy": wcy, "r": (r1 + r2) / 2}
