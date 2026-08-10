"""
/health, /api/predict 통합 테스트 (Day 21).
모델 파일(predictor.joblib)이 없으면 예측 테스트는 skip한다.
"""
import os

import pytest

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "models", "predictor.joblib")
_model_missing = not os.path.exists(MODEL_PATH)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.skipif(_model_missing, reason="predictor.joblib 없음 (ml/train_final.py 먼저 실행)")
def test_predict_success(client, cached_match_id):
    r = client.post("/api/predict", json={"match_id": cached_match_id})
    assert r.status_code == 200
    body = r.json()
    # 응답 스키마 확인
    assert body["match_id"] == cached_match_id
    assert "predicted" in body and {"x", "y", "radius"} <= set(body["predicted"])
    assert isinstance(body["phase"], int)
    # 반경은 양수, 신뢰반경은 0 이상
    assert body["predicted"]["radius"] > 0
    assert body["confidence_radius"] >= 0


def test_predict_invalid_match(client):
    # 캐시에 없는 매치 → 서버가 죽지 않고 404로 응답
    r = client.post("/api/predict", json={"match_id": "definitely-not-a-real-match"})
    assert r.status_code == 404
    assert "detail" in r.json()


def test_predict_missing_field(client):
    # match_id 누락 → Pydantic 검증 실패 422
    r = client.post("/api/predict", json={})
    assert r.status_code == 422
