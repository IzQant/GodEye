"""
YOLO(ONNX) 기반 자기장 원 검출기 — 확장 1단계 서빙용.

데스크탑에서 학습한 YOLOv8(best.pt)을 ONNX로 내보낸 파일(ml/models/zone_detect.onnx)을
OpenCV DNN으로 추론한다. torch/ultralytics 의존 없이(배포 경량 유지) 동작.

- 전처리: letterbox(비율 유지 패딩) → 1x3xSxS, 0~1 정규화
- 후처리: YOLOv8 출력 (1, 4+nc, N) 디코드 → NMS → 클래스별 최고점 박스
- 박스 → 원: 중심 = 박스 중심, 반경 = (w+h)/4 (원의 외접 정사각형 가정)
- 인터페이스: CircleDetector.detect_with_confidence와 동일한 dict 반환
  {"safe", "next", "needs_manual", "reasons"}  (safe=cls0, next=cls1)
"""
import os

import cv2
import numpy as np

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..",
                                  "ml", "models", "zone_detect.onnx")
INPUT_SIZE = 768          # 학습 imgsz와 동일해야 함
CONF_THRESHOLD = 0.35     # 검출 신뢰도 임계값(이하면 수동 폴백)
NMS_THRESHOLD = 0.45
CLASS_NAMES = {0: "safe", 1: "next"}


def letterbox(img, size):
    """비율 유지 리사이즈 + 회색 패딩. (blob, scale, pad_x, pad_y) 반환."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, s, px, py


def decode_yolov8(out, conf_thr):
    """
    YOLOv8 ONNX 출력 (1, 4+nc, N) → [(cx, cy, w, h, score, cls), ...]
    좌표는 letterbox 입력 기준.
    """
    pred = out[0]                    # (4+nc, N)
    if pred.shape[0] > pred.shape[1]:
        pred = pred.T                # 안전장치: (N, 4+nc)로 온 경우
    boxes_cxcywh = pred[:4].T        # (N, 4)
    scores_all = pred[4:].T          # (N, nc)
    cls_ids = scores_all.argmax(1)
    scores = scores_all.max(1)
    keep = scores >= conf_thr
    result = []
    for (cx, cy, w, h), sc, ci in zip(boxes_cxcywh[keep], scores[keep], cls_ids[keep]):
        result.append((float(cx), float(cy), float(w), float(h), float(sc), int(ci)))
    return result


class YoloCircleDetector:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 conf_threshold: float = CONF_THRESHOLD):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX 모델 없음: {model_path}")
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.conf_threshold = conf_threshold

    def _infer(self, image):
        blob_img, s, px, py = letterbox(image, INPUT_SIZE)
        blob = cv2.dnn.blobFromImage(blob_img, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        dets = decode_yolov8(out, self.conf_threshold)

        # NMS (클래스 무관 일괄 → 클래스별 최고점만 쓸 것이므로 충분)
        if dets:
            boxes = [[d[0] - d[2] / 2, d[1] - d[3] / 2, d[2], d[3]] for d in dets]
            scores = [d[4] for d in dets]
            idx = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, NMS_THRESHOLD)
            idx = np.array(idx).ravel().tolist() if len(idx) else []
            dets = [dets[i] for i in idx]

        # 클래스별 최고 점수 박스 → 원(원본 좌표계로 역변환)
        best = {}
        for cx, cy, w, h, sc, ci in dets:
            if ci not in best or sc > best[ci]["confidence"]:
                ox = (cx - px) / s
                oy = (cy - py) / s
                r = ((w / s) + (h / s)) / 4      # 외접 정사각형 → 반경
                best[ci] = {"cx": float(ox), "cy": float(oy), "r": float(r),
                            "confidence": float(sc)}
        return best

    def detect_circles(self, image):
        best = self._infer(image)
        return {"safe": best.get(0), "next": best.get(1)}

    def detect_with_confidence(self, image):
        """CircleDetector와 동일 인터페이스. safe 미검출/저신뢰 시 needs_manual."""
        det = self.detect_circles(image)
        reasons = []
        for key, label in (("safe", "흰 원"), ("next", "파란 원")):
            c = det[key]
            if c is None:
                reasons.append(f"{label} 검출 실패(YOLO)")
        # 예측 파이프라인에는 safe(현재 원)만 필수 — next 실패는 경고로만
        needs_manual = det["safe"] is None
        return {"safe": det["safe"], "next": det["next"],
                "needs_manual": needs_manual, "reasons": reasons}


# ---- 파이프라인용 로더(1회 로드 캐시, 실패 시 None) ----
_YOLO = None
_YOLO_TRIED = False


def get_yolo_detector():
    """ONNX 모델이 있으면 로드해 재사용, 없으면 None (호출부에서 색상 방식 폴백)."""
    global _YOLO, _YOLO_TRIED
    if not _YOLO_TRIED:
        _YOLO_TRIED = True
        try:
            _YOLO = YoloCircleDetector()
            print("[yolo_detector] ONNX 모델 로드 완료")
        except Exception as e:
            print(f"[yolo_detector] 미사용(로드 실패/파일 없음): {e}")
            _YOLO = None
    return _YOLO
