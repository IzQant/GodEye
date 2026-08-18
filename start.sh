#!/bin/sh
# 컨테이너 시작 스크립트 (배포용).
# 핵심: gunicorn이 $PORT에 반드시 붙도록 하고, 마이그레이션이 서버 기동을 막지 않게 한다.

echo "[start] PORT=${PORT:-8000} WEB_CONCURRENCY=${WEB_CONCURRENCY:-1}"

# DB 마이그레이션은 best-effort. DB가 늦게 뜨거나 안 붙어도 서버는 떠야 하므로
# 타임아웃(25s)을 걸어 절대 여기서 멈추지 않게 한다.
( timeout 25 alembic upgrade head && echo "[start] migration done" ) \
    || echo "[start] migration skipped/failed (continuing)"

# 플랫폼이 주는 $PORT(없으면 8000)에 바인딩. 무료 티어 메모리 절약 위해 워커 기본 1개.
exec gunicorn -k uvicorn.workers.UvicornWorker \
    -w "${WEB_CONCURRENCY:-1}" \
    -b "0.0.0.0:${PORT:-8000}" \
    --timeout 120 \
    app.main:app
