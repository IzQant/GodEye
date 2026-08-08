"""
SQLAlchemy 엔진/세션 설정 (Day 8).

- engine: DB에 실제로 연결하는 객체. DATABASE_URL(.env)을 사용.
- SessionLocal: 요청마다 짧게 쓰고 닫는 DB 세션 팩토리.
- Base: 모든 ORM 모델이 상속하는 기반 클래스 (models.py에서 사용).
- get_db(): FastAPI 라우터에서 의존성 주입으로 세션을 받아 쓰고,
            끝나면 자동으로 닫아주는 제너레이터.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

# DATABASE_URL이 비어 있으면 로컬 compose 기본값으로 대체.
DATABASE_URL = settings.DATABASE_URL or "postgresql://pubg:pubg@localhost:5432/pubg_zone"

# pool_pre_ping: 끊긴 커넥션을 미리 감지해 재연결 (무료 호스팅에서 유용)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI 의존성: 세션을 열어 넘겨주고, 요청이 끝나면 반드시 닫는다."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
