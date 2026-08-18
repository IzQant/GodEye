#!/bin/sh
# 컨테이너 시작 스크립트 (배포용).
# 1) DB 마이그레이션(있으면) 적용 — 실패해도 서비스는 계속 뜬다(best-effort).
# 2) 플랫폼이 주는 $PORT(없으면 8000)에 바인딩해 gunicorn 실행.
#    무료 티어 메모리 절약을 위해 워커 기본 1개(WEB_CONCURRENCY로 조정).

alembic upgrade head || echo "[start] 마이그레이션 건너뜀/실패(계속 진행)"

exec gunicorn -k uvicorn.workers.UvicornWorker \
    -w "${WEB_CONCURRENCY:-1}" \
    -b "0.0.0.0:${PORT:-8000}" \
    app.main:app
