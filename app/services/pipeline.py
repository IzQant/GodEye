"""
Day 29 작업: 통합 파이프라인.

두 입력 경로를 하나의 결과 스키마로 묶는다.
1) matchId 경로 : 텔레메트리 조회 → 현재 원(맵 좌표) → 예측
2) 이미지 경로  : 전체 지도 스크린샷 + phase + map + (선택)수동 원좌표
                  → 현재 원 검출(픽셀) → 맵 좌표 변환 → 예측
   (설계: friday/raw/memories/design_image_phase_input.md)

공통 결과 스키마(build_result):
{
  input_type: "match_id" | "image",
  map, phase,
  current:   {x, y, radius}  # 맵 좌표(cm)
  predicted: {x, y, radius} | None,
  confidence_radius: float | None,
  needs_manual: bool,
  reasons: [str],
  pixel: {...} | None,        # 이미지 경로일 때 오버레이용 픽셀 정보
}
"""
from app.services.circle_detector import CircleDetector
from app.services.coordinate_transform import full_map_affine
from app.services.match_service import get_current_circle
from app.services.model_service import get_predictor
from app.services.yolo_detector import get_yolo_detector

_detector = CircleDetector()


def _get_detector():
    """YOLO(ONNX) 모델이 있으면 우선 사용, 없으면 색상 기반으로 폴백.
    둘 다 detect_with_confidence(image) 인터페이스가 동일하다."""
    return get_yolo_detector() or _detector


def build_result(input_type, map_name, phase, current, predicted,
                 confidence_radius, needs_manual=False, reasons=None, pixel=None):
    """모든 경로가 동일한 형태로 반환하도록 결과를 조립."""
    return {
        "input_type": input_type,
        "map": map_name,
        "phase": phase,
        "current": current,
        "predicted": predicted,
        "confidence_radius": confidence_radius,
        "needs_manual": needs_manual,
        "reasons": reasons or [],
        "pixel": pixel,
    }


def _predict(predictor, x, y, r, phase, map_name):
    out = predictor.predict(x, y, r, phase=phase, map_name=map_name)
    return ({"x": out["x"], "y": out["y"], "radius": out["radius"]},
            out["confidence_radius"])


def analyze_by_match_id(match_id, predictor=None):
    """matchId → 현재 원(맵 좌표) → 예측. 항상 자동(수동 폴백 없음)."""
    predictor = predictor or get_predictor()
    cur = get_current_circle(match_id)   # {safety_x/y/radius, phase, map_name}
    current = {"x": cur["safety_x"], "y": cur["safety_y"], "radius": cur["safety_radius"]}
    predicted, conf = _predict(predictor, cur["safety_x"], cur["safety_y"],
                               cur["safety_radius"], cur["phase"], cur["map_name"])
    return build_result("match_id", cur["map_name"], cur["phase"],
                        current, predicted, conf)


def analyze_by_image(image, phase, map_name, manual_pixel=None, predictor=None):
    """
    전체 지도 이미지 + phase + map (+ 선택 수동 원좌표) → 예측.

    manual_pixel: {"cx","cy","r"} 사용자가 직접 찍은 현재 원(픽셀). 있으면 검출 대신 사용.
    검출 신뢰도가 낮고 수동 입력도 없으면 needs_manual=True로 예측 없이 반환.
    """
    h, w = image.shape[:2]
    transform = full_map_affine(w, h, map_name)  # 픽셀→맵(전체지도 스케일)

    # 현재 원(픽셀) 확보: 수동 입력 우선, 없으면 자동 검출
    if manual_pixel is not None:
        pcx, pcy, pr = manual_pixel["cx"], manual_pixel["cy"], manual_pixel["r"]
    else:
        det = _get_detector()
        print(f"[detector] 사용 검출기: {type(det).__name__}", flush=True)
        try:
            res = det.detect_with_confidence(image)
        except Exception as e:
            # YOLO 추론 실패(메모리/런타임 등) → 색상 방식으로 폴백해 결과가 끊기지 않게
            print(f"[detector] {type(det).__name__} 실패 → 색상 폴백: {e}", flush=True)
            res = _detector.detect_with_confidence(image)
        safe = res["safe"]
        if res["needs_manual"] or safe is None:
            # 검출 실패/저신뢰 → 예측 없이 수동 입력 요청 (모델 로드 불필요)
            return build_result("image", map_name, phase, None, None, None,
                                needs_manual=True, reasons=res["reasons"])
        pcx, pcy, pr = safe["cx"], safe["cy"], safe["r"]

    # 예측 직전에만 모델 로드(검출 실패 시엔 로드하지 않음)
    predictor = predictor or get_predictor()

    # 픽셀 → 맵 좌표
    cur_map = transform.apply_circle(pcx, pcy, pr)   # {x, y, radius}
    predicted, conf = _predict(predictor, cur_map["x"], cur_map["y"],
                               cur_map["radius"], phase, map_name)
    try:
        from app.services.coordinate_transform import MAP_SIZES_CM
        s = MAP_SIZES_CM.get(map_name, 1)
        print(f"[predict] 현재({cur_map['x']/s*100:.0f}%,{cur_map['y']/s*100:.0f}%) "
              f"→ 예측({predicted['x']/s*100:.0f}%,{predicted['y']/s*100:.0f}%) "
              f"r {predicted['radius']/s*100:.0f}%", flush=True)
    except Exception:
        pass
    return build_result("image", map_name, phase, cur_map, predicted, conf,
                        pixel={"cx": pcx, "cy": pcy, "r": pr})
