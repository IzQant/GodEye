"""
특징점 매칭 기반 자동 지도 정렬 (사진 → 전체지도 정사각 보정).

사용자가 모니터를 촬영한 사진에는 베젤·책상·기울어짐이 섞여 있어
'이미지 전체 = 전체 지도' 가정이 깨진다. 이 모듈은 참조 맵(data/maps/<map>.png)과
사진 사이의 SIFT 특징점을 매칭하고 RANSAC으로 호모그래피를 추정해,
사진을 참조 맵 좌표계(정사각 전체지도)로 자동 보정한다.

- 지형 텍스처(도로·해안선)는 특징점이 풍부해 원근·조명 변화에 강건.
- 4점을 '찾는' 게 아니라 수백 개 대응점으로 변환을 '추정' → 부분 가림에도 견딤.
- 성공 판정: RANSAC 인라이어 수/비율 임계값. 실패 시 None 반환(호출부 폴백).
"""
import os

import cv2
import numpy as np

MAPS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "maps")

OUT_SIZE = 1024          # 보정 결과(정사각 전체지도) 한 변
PHOTO_MAX_SIDE = 1600    # 사진은 이 크기로 축소해 매칭(속도), H는 원본 기준으로 환산
MIN_GOOD_MATCHES = 30    # ratio test 통과 최소 매칭 수
MIN_INLIERS = 25         # RANSAC 인라이어 최소 수
MIN_INLIER_RATIO = 0.25  # 인라이어/good 비율 최소

_ref_cache: dict = {}    # map_name -> (gray, kp, des)  참조 맵 특징점 캐시
_sift = None


def _get_sift():
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=4000)
    return _sift


def _load_ref(map_name: str):
    """참조 맵을 OUT_SIZE 정사각으로 로드하고 특징점을 1회 계산해 캐시."""
    key = map_name.lower()
    if key in _ref_cache:
        return _ref_cache[key]
    path = os.path.join(MAPS_DIR, f"{key}.png")
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(cv2.resize(img, (OUT_SIZE, OUT_SIZE)), cv2.COLOR_BGR2GRAY)
    kp, des = _get_sift().detectAndCompute(gray, None)
    _ref_cache[key] = (gray, kp, des)
    return _ref_cache[key]


def align_photo_to_map(photo_bgr: np.ndarray, map_name: str):
    """
    사진을 map_name 전체지도 좌표계(OUT_SIZE 정사각)로 보정.

    반환: (warped_bgr, H, info)
      - warped_bgr: 보정된 정사각 지도 이미지 (실패 시 None)
      - H: 원본 사진 픽셀 → 보정 이미지 픽셀 3x3 호모그래피 (실패 시 None)
      - info: {"ok", "reason", "good", "inliers"}
    """
    ref = _load_ref(map_name)
    if ref is None:
        return None, None, {"ok": False, "reason": f"참조 맵 없음: {map_name}", "good": 0, "inliers": 0}
    ref_gray, ref_kp, ref_des = ref
    if ref_des is None or len(ref_kp) < MIN_GOOD_MATCHES:
        return None, None, {"ok": False, "reason": "참조 맵 특징점 부족", "good": 0, "inliers": 0}

    # 사진 축소(속도) — 배율 s 기록, H 계산 뒤 원본 기준으로 환산
    h, w = photo_bgr.shape[:2]
    s = min(1.0, PHOTO_MAX_SIDE / max(h, w))
    small = cv2.resize(photo_bgr, (int(w * s), int(h * s))) if s < 1.0 else photo_bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    kp, des = _get_sift().detectAndCompute(gray, None)
    if des is None or len(kp) < MIN_GOOD_MATCHES:
        return None, None, {"ok": False, "reason": "사진 특징점 부족", "good": 0, "inliers": 0}

    # KNN 매칭 + Lowe ratio test
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(des, ref_des, k=2)
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]
    if len(good) < MIN_GOOD_MATCHES:
        return None, None, {"ok": False, "reason": f"매칭 부족({len(good)})", "good": len(good), "inliers": 0}

    src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)     # 사진(축소) 좌표
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  # 참조 맵 좌표

    H_small, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if H_small is None or inliers < MIN_INLIERS or inliers / len(good) < MIN_INLIER_RATIO:
        return None, None, {"ok": False, "reason": f"호모그래피 불안정(inliers {inliers}/{len(good)})",
                            "good": len(good), "inliers": inliers}

    # 축소 좌표계 → 원본 좌표계 보정: H = H_small @ S (S: 원본→축소 스케일)
    S = np.diag([s, s, 1.0])
    H = H_small @ S

    warped = cv2.warpPerspective(photo_bgr, H, (OUT_SIZE, OUT_SIZE))
    return warped, H, {"ok": True, "reason": "ok", "good": len(good), "inliers": inliers}
