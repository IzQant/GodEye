# 프로덕션 이미지 (Day 36)
FROM python:3.11-slim

# OpenCV(libgl1, libglib2.0-0) 및 scikit-learn/scipy(OpenMP=libgomp1) 실행 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 런타임 의존성만 설치(가벼운 이미지)
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# 앱/모델 코드 + 학습에 필요한 정제 데이터셋 복사
COPY app/ ./app/
COPY ml/ ./ml/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY data/processed/ ./data/processed/

# 컨테이너의 sklearn 버전으로 예측 모델을 빌드 시 생성(버전 불일치 방지)
RUN python ml/train_final.py

EXPOSE 8000
# 프로덕션: gunicorn(worker manager) + uvicorn worker
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "2", "-b", "0.0.0.0:8000", "app.main:app"]
