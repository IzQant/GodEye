"""
특징점 매칭 기반 자동 지도 정렬 (사진 → 전체지도 정사각 보정).

사용자가 모니터를 촬영한 사진에는 베젤·책상·기울어짐이 섞여 있어
'이미지 전체 = 전체 지도' 가정이 깨진다. 이 모듈은 참조 맵(data/maps/<map>.png)과
사진 사이의 SIFT 특징점을 매칭하고 RANSAC으로 호모그래피를 추정해,
사진을 참조 맵 좌표계(정사각 전체지도)로 자동 보정한다.

강건화(실사진 대응):
- 참조 맵은 '원본 해상도'로 특징점 계산(업스케일 보간 뭉갬 방지), H는 출력 크기로 후환산.
- 1차 시도 실패 시 재시도 사다리: CLAHE 대비강화 → (작은 사진이면) 2x 업스케일+CLAHE.
- 성공 판정: good/inlier 임계값. 실패 시 (None, None, info) — 호출부에서 폴백·경고.
"""
import os

import cv2
import numpy as np

MAPS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "maps")

OUT_SIZE = 1024          # 보정 결과(정사각 전체지도) 한 변
PHOTO_MAX_SIDE = 1600    # 사진 매칭용 최대 크기(속도)
UPSCALE_IF_BELOW = 900   # 사진이 이보다 작으면 업스케일 재시도 후보
MIN_GOOD_MATCHES = 30
MIN_INLIERS = 25
MIN_INLIER_RATIO = 0.25

_ref_cache: dict = {}
_sift = None


def _get_sift():
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=5000)
    return _sift


def _clahe(gray):
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)


def _load_ref(map_name: str):
    """참조 맵을 '원본 해상도'로 로드해 특징점 1회 계산·캐시. (side, kp, des) 반환."""
    key = map_name.lower()
    if key in _ref_cache:
        return _ref_cache[key]
    path = os.path.join(MAPS_DIR, f"{key}.png")
    img = cv2.imread(path)
    if img is None:
        return None
    side = min(img.shape[:2])
    img = cv2.resize(img, (side, side)) if img.shape[0] != img.shape[1] else img
    gray = _clahe(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    kp, des = _get_sift().detectAndCompute(gray, None)
    _ref_cache[key] = (side, kp, des)
    return _ref_cache[key]


def _try_match(gray, ref_kp, ref_des, ref_side):
    """사진(gray) → 참조 원본 좌표계 호모그래피 시도. (H_ref, good, inliers)"""
    kp, des = _get_sift().detectAndCompute(gray, None)
    if des is None or len(kp) < MIN_GOOD_MATCHES:
        return None, 0, 0
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des, ref_des, k=2)
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]
    if len(good) < MIN_GOOD_MATCHES:
        return None, len(good), 0
    src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if H is None or inliers < MIN_INLIERS or inliers / len(good) < MIN_INLIER_RATIO:
        return None, len(good), inliers
    return H, len(good), inliers


def align_photo_to_map(photo_bgr: np.ndarray, map_name: str):
    """
    사진을 map_name 전체지도 좌표계(OUT_SIZE 정사각)로 보정.
    반환: (warped_bgr|None, H|None, info)  — H는 원본 사진 픽셀 → 보정 픽셀.
    """
    ref = _load_ref(map_name)
    if ref is None:
        return None, None, {"ok": False, "reason": f"참조 맵 없음: {map_name}", "good": 0, "inliers": 0}
    ref_side, ref_kp, ref_des = ref
    if ref_des is None or len(ref_kp) < MIN_GOOD_MATCHES:
        return None, None, {"ok": False, "reason": "참조 맵 특징점 부족", "good": 0, "inliers": 0}

    h, w = photo_bgr.shape[:2]
    base_scale = min(1.0, PHOTO_MAX_SIDE / max(h, w))

    # 재시도 사다리: (배율, CLAHE 여부)
    attempts = [(base_scale, False), (base_scale, True)]
    if max(h, w) < UPSCALE_IF_BELOW:
        attempts.append((2.0, True))   # 작은 사진: 2x 업스케일 + CLAHE

    best = {"good": 0, "inliers": 0}
    for s, use_clahe in attempts:
        img = photo_bgr if s == 1.0 else cv2.resize(
            photo_bgr, (int(w * s), int(h * s)),
            interpolation=cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if use_clahe:
            gray = _clahe(gray)
        H_ref, good, inliers = _try_match(gray, ref_kp, ref_des, ref_side)
        best = max(best, {"good": good, "inliers": inliers}, key=lambda d: d["inliers"])
        if H_ref is not None:
            # (원본→시도배율) → (참조원본→출력) 순으로 합성
            S_photo = np.diag([s, s, 1.0])
            S_out = np.diag([OUT_SIZE / ref_side, OUT_SIZE / ref_side, 1.0])
            H = S_out @ H_ref @ S_photo
            warped = cv2.warpPerspective(photo_bgr, H, (OUT_SIZE, OUT_SIZE))
            return warped, H, {"ok": True, "reason": "ok", "good": good, "inliers": inliers}

    return None, None, {"ok": False,
                        "reason": f"매칭 부족(good {best['good']}, inliers {best['inliers']})",
                        "good": best["good"], "inliers": best["inliers"]}
