"""
검출 모듈(CircleDetector) 단위 테스트 (Day 27).
샘플 이미지 기반: 합성 이미지(정답 labels.csv 보유)를 사용.
"""
import csv
import glob
import os

import cv2
import numpy as np
import pytest

from app.services.circle_detector import CircleDetector

SYN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images", "synthetic")


def _first_synthetic():
    files = sorted(glob.glob(os.path.join(SYN_DIR, "*.png")))
    return files[0] if files else None


@pytest.fixture(scope="module")
def detector():
    return CircleDetector()


def test_make_masks_nonempty(detector):
    path = _first_synthetic()
    if path is None:
        pytest.skip("synthetic 이미지 없음")
    img = cv2.imread(path)
    white, blue = detector.make_masks(img)
    assert (white > 0).sum() > 0
    assert (blue > 0).sum() > 0


def test_detect_matches_label(detector):
    """합성 이미지 1장에서 검출 중심이 정답과 가까운지(관대한 허용오차)."""
    labels_path = os.path.join(SYN_DIR, "labels.csv")
    if not os.path.exists(labels_path):
        pytest.skip("labels.csv 없음")
    row = next(csv.DictReader(open(labels_path, encoding="utf-8")))
    img = cv2.imread(os.path.join(SYN_DIR, row["filename"]))
    size = int(row["size"])
    det = detector.detect_circles(img)

    assert det["safe"] is not None and det["next"] is not None
    tol = size * 0.03
    assert abs(det["safe"]["cx"] - int(row["safe_cx"])) <= tol
    assert abs(det["safe"]["cy"] - int(row["safe_cy"])) <= tol
    assert abs(det["next"]["cx"] - int(row["next_cx"])) <= tol
    assert abs(det["next"]["cy"] - int(row["next_cy"])) <= tol


def test_confidence_high_on_clean(detector):
    path = _first_synthetic()
    if path is None:
        pytest.skip("synthetic 이미지 없음")
    res = detector.detect_with_confidence(cv2.imread(path))
    # 깨끗한 합성 이미지는 대체로 수동입력 불필요
    assert res["needs_manual"] is False


def test_fallback_on_garbage(detector):
    """원이 없는 잡음 이미지 → 검출 실패로 needs_manual=True."""
    noise = np.random.default_rng(0).integers(0, 60, (300, 300, 3), dtype=np.uint8)
    res = detector.detect_with_confidence(noise)
    assert res["needs_manual"] is True
    assert len(res["reasons"]) > 0


def test_custom_hsv_injection():
    """설정 주입: 말도 안 되는 HSV 범위면 흰 원도 못 잡음(클래스 파라미터 반영 확인)."""
    path = _first_synthetic()
    if path is None:
        pytest.skip("synthetic 이미지 없음")
    # 흰색 범위를 빈 범위로 → 흰 마스크 비어야 함
    d = CircleDetector(white_hsv=((0, 254, 0), (0, 255, 0)))
    white, _ = d.make_masks(cv2.imread(path))
    assert (white > 0).sum() == 0
