"""
E2E 시나리오 테스트 (Day 34) — 주요 흐름 3가지.
1) matchId 성공
2) 이미지 성공 (검출 → 예측)
3) 이미지 실패 (검출 실패 → needs_manual, 예측 없음)

모델 로드가 안 되는 환경(sklearn 버전 등)을 위해, 예측이 필요한 경로는
pipeline.get_predictor를 스텁으로 교체(monkeypatch)해 흐름 자체를 검증한다.
"""
import cv2
import numpy as np
import pytest


class _StubPredictor:
    def predict(self, x, y, r, phase, map_name):
        return {"x": x + 1000, "y": y - 500, "radius": r * 0.7,
                "confidence_radius": 8000.0}


def _white_blue_png(size=512):
    img = np.full((size, size, 3), 50, np.uint8)
    cv2.circle(img, (size // 2, size // 2), int(size * 0.35), (255, 255, 255), 2)
    cv2.circle(img, (size // 2, size // 2), int(size * 0.22), (255, 130, 40), 2)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _noise_png(size=300):
    noise = np.random.default_rng(0).integers(0, 60, (size, size, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", noise)
    return buf.tobytes()


# --- 1) matchId 성공 (모델 없으면 503 → skip) ---
def test_e2e_match_id(client, cached_match_id):
    r = client.post("/api/analyze", data={"match_id": cached_match_id})
    if r.status_code == 503:
        pytest.skip("모델 로드 불가 환경")
    assert r.status_code == 200
    body = r.json()
    assert body["input_type"] == "match_id"
    assert body["predicted"] is not None


# --- 2) 이미지 성공 (스텁 예측기로 흐름 검증) ---
def test_e2e_image_success(client, monkeypatch):
    monkeypatch.setattr("app.services.pipeline.get_predictor", lambda: _StubPredictor())
    r = client.post("/api/analyze",
                    files={"image": ("m.png", _white_blue_png(), "image/png")},
                    data={"phase": "5", "map_name": "Erangel"})
    assert r.status_code == 200
    body = r.json()
    assert body["input_type"] == "image"
    assert body["needs_manual"] is False
    assert body["predicted"] is not None
    assert body["current"] is not None


# --- 3) 이미지 실패 → 수동 입력 유도(예측 없음, 모델 로드도 안 함) ---
def test_e2e_image_fail(client):
    r = client.post("/api/analyze",
                    files={"image": ("m.png", _noise_png(), "image/png")},
                    data={"phase": "5", "map_name": "Erangel"})
    assert r.status_code == 200
    body = r.json()
    assert body["needs_manual"] is True
    assert body["predicted"] is None
    assert len(body["reasons"]) > 0


# --- 보너스: 시각화 PNG 반환 (스텁) ---
def test_e2e_visualize_png(client, monkeypatch):
    monkeypatch.setattr("app.services.pipeline.get_predictor", lambda: _StubPredictor())
    r = client.post("/api/visualize",
                    files={"image": ("m.png", _white_blue_png(), "image/png")},
                    data={"phase": "5", "map_name": "Erangel"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 500
