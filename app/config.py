"""
환경변수 로딩 모듈.
.env 파일의 값을 읽어 앱 전역에서 쓸 설정값으로 제공한다.
(Day 7 뼈대 단계 — DB/모델 관련 값은 이후 주차에서 실제로 사용)
"""
import os

from dotenv import load_dotenv

# .env를 한 번 읽어 os.environ에 올린다. 앱 어디서 import하든 최초 1회만 로드된다.
load_dotenv()


class Settings:
    # PUBG API 관련
    PUBG_API_KEY: str = os.getenv("PUBG_API_KEY", "")
    PUBG_SHARD: str = os.getenv("PUBG_SHARD", "steam")
    PUBG_PLAYER_NAME: str = os.getenv("PUBG_PLAYER_NAME", "")

    # DB (Week 2에서 실제 사용)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # 세션/보안 (Week 5~6에서 실제 사용)
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")


# 앱 전역에서 settings 하나만 import해서 쓰도록 인스턴스로 노출
settings = Settings()
