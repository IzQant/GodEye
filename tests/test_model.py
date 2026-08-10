"""
모델/서비스 단위 테스트 (Day 21).
- ZonePredictor.predict가 스키마대로 값을 내는지
- phase fallback이 동작하는지
- match_service가 캐시 매치에서 현재 원을 뽑는지
"""
import os

import pytest

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "models", "predictor.joblib")
_model_missing = not os.path.exists(MODEL_PATH)


@pytest.mark.skipif(_model_missing, reason="predictor.joblib 없음")
def test_predictor_output_schema():
    from app.services.model_service import get_predictor
    p = get_predictor()
    out = p.predict(350000, 550000, 100000, phase=5, map_name="Erangel")
    for key in ("x", "y", "radius", "confidence_radius", "used_phase"):
        assert key in out
    assert out["radius"] > 0
    assert out["confidence_radius"] >= 0


@pytest.mark.skipif(_model_missing, reason="predictor.joblib 없음")
def test_predictor_phase_fallback():
    # 존재하지 않는 큰 phase → 가장 가까운 하위 단계로 폴백(예외 없이 동작)
    from app.services.model_service import get_predictor
    p = get_predictor()
    out = p.predict(300000, 300000, 20000, phase=999, map_name="Erangel")
    assert out["used_phase"] in p.known_phases


def test_match_service_current_circle(cached_match_id):
    from app.services.match_service import get_current_circle
    cur = get_current_circle(cached_match_id)
    for key in ("safety_x", "safety_y", "safety_radius", "phase", "map_name"):
        assert key in cur
    assert cur["safety_radius"] > 0
