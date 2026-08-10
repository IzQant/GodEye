"""
pytest 공용 픽스처.
- client: FastAPI 테스트 클라이언트
- cached_match_id: data/raw에 있는 실제 매치 ID 하나 (없으면 관련 테스트 skip)
"""
import glob
import os

import pytest
from fastapi.testclient import TestClient

from app.main import app

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture(scope="session")
def cached_match_id():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")),
                   key=os.path.getsize, reverse=True)  # 큰(=실제 경기) 파일 우선
    if not files:
        pytest.skip("data/raw에 캐시된 매치가 없어 예측 테스트를 건너뜁니다.")
    return os.path.splitext(os.path.basename(files[0]))[0]
