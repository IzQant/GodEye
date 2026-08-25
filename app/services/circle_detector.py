"""
미니맵에서 흰 원(현재 안전지대)/파란 원(다음 자기장)을 검출하는 모듈.

Day 23: HSV 색상 필터링
Day 24: Contour + minEnclosingCircle로 중심/반경 추출
Day 25: 신뢰도(circularity) + 수동입력 폴백 분기
Day 27: 위 기능을 CircleDetector 클래스로 정리 (설정 주입/재사용 용이)

동작 단계 요약:
1) BGR → HSV 변환 (밝기 변화에 강건)
2) 흰/파란 각각 cv2.inRange로 색 마스크 + 모폴로지로 잡티 제거
3) 마스크에서 가장 큰 윤곽선 → minEnclosingCircle로 중심/반경
4) circularity로 신뢰도 산출, 낮으면 수동 입력 필요로 표시
"""
import cv2
import numpy as np

# 기본 HSV 범위(OpenCV: H 0~179, S 0~255, V 0~255)
DEFAULT_WHITE_HSV = ((0, 0, 180), (179, 40, 255))
DEFAULT_BLUE_HSV = ((95, 80, 80), (135, 255, 255))
CONFIDENCE_THRESHOLD = 0.70


class CircleDetector:
    def __init__(self, white_hsv=DEFAULT_WHITE_HSV, blue_hsv=DEFAULT_BLUE_HSV,
                 min_confidence=CONFIDENCE_THRESHOLD, min_radius=5.0):
        self.white_hsv = white_hsv
        self.blue_hsv = blue_hsv
        self.min_confidence = min_confidence
        self.min_radius = min_radius
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # ---- 1~2단계: 색 마스크 ----
    def _mask(self, hsv, hsv_range):
        lo = np.array(hsv_range[0], dtype=np.uint8)
        hi = np.array(hsv_range[1], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

    def make_masks(self, image):
        """BGR 이미지 → (흰 마스크, 파란 마스크)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return self._mask(hsv, self.white_hsv), self._mask(hsv, self.blue_hsv)

    # ---- 3~4단계: 원 추출 + 신뢰도 ----
    def detect_circle(self, mask):
        """이진 마스크 → {cx, cy, r, confidence} 또는 None."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(largest)
        if r < self.min_radius:
            return None
        area = cv2.contourArea(largest)
        perim = cv2.arcLength(largest, True)
        circularity = (4 * np.pi * area / (perim ** 2)) if perim > 0 else 0.0
        return {"cx": float(cx), "cy": float(cy), "r": float(r),
                "confidence": float(min(circularity, 1.0))}

    def detect_circles(self, image):
        """BGR 이미지 → {'safe': {...}|None, 'next': {...}|None}."""
        white_mask, blue_mask = self.make_masks(image)
        return {"safe": self.detect_circle(white_mask),
                "next": self.detect_circle(blue_mask)}

    def detect_with_confidence(self, image, min_radius_frac=None):
        """
        검출 + 신뢰도 판정 + 폴백 분기.
        흰/파란 중 하나라도 검출 실패거나 신뢰도가 임계값 미만이면 needs_manual=True.
        반환: {'safe', 'next', 'needs_manual', 'reasons'}
        (min_radius_frac: YOLO 검출기와 인터페이스 호환용. 색상 검출기는 사용하지 않음)
        """
        det = self.detect_circles(image)
        reasons = []
        for key, label in (("safe", "흰 원"), ("next", "파란 원")):
            c = det[key]
            if c is None:
                reasons.append(f"{label} 검출 실패")
            elif c["confidence"] < self.min_confidence:
                reasons.append(f"{label} 신뢰도 낮음({c['confidence']:.2f})")
        return {"safe": det["safe"], "next": det["next"],
                "needs_manual": len(reasons) > 0, "reasons": reasons}


# ---- 하위 호환: 기존 스크립트가 쓰던 모듈 레벨 함수/상수 유지 ----
WHITE_HSV_RANGE = DEFAULT_WHITE_HSV
BLUE_HSV_RANGE = DEFAULT_BLUE_HSV
_DEFAULT = CircleDetector()


def make_masks(image):
    return _DEFAULT.make_masks(image)


def detect_circle(mask, min_radius: float = 5.0):
    d = CircleDetector(min_radius=min_radius)
    return d.detect_circle(mask)


def detect_circles(image):
    return _DEFAULT.detect_circles(image)


def detect_with_confidence(image, min_conf: float = CONFIDENCE_THRESHOLD):
    return CircleDetector(min_confidence=min_conf).detect_with_confidence(image)
