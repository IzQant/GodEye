"""
Alembic 마이그레이션 실행 환경 (Day 9).

- app.database.Base.metadata를 target으로 삼아, 모델 변경 시
  `alembic revision --autogenerate`가 차이를 감지할 수 있게 한다.
- 접속 URL은 app.config의 DATABASE_URL(.env)에서 가져온다.
"""
import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# 프로젝트 루트를 import 경로에 추가 (app 패키지를 찾기 위함)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.config import settings
from app.database import Base
from app import models  # noqa: F401  (모델을 import해야 메타데이터에 테이블이 등록됨)

config = context.config

# .env의 DATABASE_URL을 우선 사용, 없으면 로컬 compose 기본값
db_url = settings.DATABASE_URL or "postgresql://pubg:pubg@localhost:5432/pubg_zone"
config.set_main_option("sqlalchemy.url", db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """DB에 직접 연결하지 않고 SQL 스크립트만 생성하는 모드."""
    context.configure(
        url=db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """DB에 실제 연결해 마이그레이션을 적용하는 모드."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
