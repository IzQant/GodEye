"""
/api/analyze 입력 검증 테스트 (Day 30).
모델/이미지 파일 없이도 도는 검증·에러 케이스 위주.
"""
import cv2
import numpy as np


def _png_bytes():
    img = np.full((200, 200, 3), 50, np.uint8)
    cv2.circle(img, (100, 100), 70, (255, 255, 255), 2)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_analyze_no_input(client):
    r = client.post("/api/analyze", data={})
    assert r.status_code == 422


def test_analyze_image_missing_phase(client):
    r = client.post("/api/analyze",
                    files={"image": ("m.png", _png_bytes(), "image/png")},
                    data={"map_name": "Erangel"})
    assert r.status_code == 422


def test_analyze_unknown_map(client):
    r = client.post("/api/analyze",
                    files={"image": ("m.png", _png_bytes(), "image/png")},
                    data={"phase": "5", "map_name": "NotARealMap"})
    assert r.status_code == 400


def test_analyze_broken_image(client):
    r = client.post("/api/analyze",
                    files={"image": ("m.png", b"not-an-image", "image/png")},
                    data={"phase": "5", "map_name": "Erangel"})
    assert r.status_code == 400


def test_analyze_match_id_path(client, cached_match_id):
    # 모델 로드 가능하면 200, 버전 불일치 환경이면 503 — 둘 다 '서버 안 죽음' 확인
    r = client.post("/api/analyze", data={"match_id": cached_match_id})
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        body = r.json()
        assert body["input_type"] == "match_id"
        assert body["predicted"] is not None
