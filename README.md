# GodEye
배틀그라운드 실시간 자기장 예측 서비스

PUBG 매치 텔레메트리를 기반으로 다음 자기장(파란 원)의 위치·반경을 예측하는 웹 서비스.
6주 구현 로드맵 기준 진행 중 (2026-07-21 시작).

## 아키텍처 (목표)
FastAPI 백엔드 + PostgreSQL + scikit-learn 예측 모델 + OpenCV 원 검출 → Docker 배포.
사용자가 matchId 또는 미니맵 스크린샷을 입력하면 다음 자기장 예측 위치를 반환한다.

## 프로젝트 구조
```
app/          FastAPI 앱 (main.py, config.py, routers/, services/, templates/)
ml/           모델 학습·평가·EDA (eda.ipynb 등)
scripts/      데이터 수집·파싱·통합 스크립트
data/         raw/ (원본 JSON), processed/ (zones_dataset.csv), images/ (CV용)
tests/        pytest
```

## 로컬 실행
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env 설정 (PUBG_API_KEY 등)

# 데이터 수집 → 파싱 → 통합
python scripts/collect_batch.py 50
python scripts/build_dataset.py

# 전체 파이프라인 한 번에 (통합 → 통계 → 평가)
python scripts/run_pipeline.py

# 서버 실행
uvicorn app.main:app --reload   # http://127.0.0.1:8000/health
```

## 데이터셋 & 베이스라인 결과 (v0.2 시점)
- 데이터: 고유 매치 20개 → 정제 후 138 phase 행
- 맵별 분포: Sanhok 45, Erangel 39, Taego 30, Miramar 13, Rondo 6, Karakin 5
- 베이스라인 오차(다음 원 중심, 매치 단위 train16/test4): 평균 5.4m / 중앙값 0.5m / p90 10.9m
- 단계별: phase 1~4 ≈0.2~0.5m, phase 6~7 ≈14~28m
- 주의: 오차가 작은 것은 예측 대상(다음 원)이 이미 발표된 값이고 초반이 동심원이기 때문.
  상세·한계는 `friday/raw/memories/baseline_eval.md` 참고. Week 3 모델의 비교 기준선.

## 진행 로그
### Week 1 (07-21 ~ 07-27) — 셋업 + 텔레메트리 수집 [v0.1]
- Day 1 (07-21): 프로젝트 셋업, requirements.txt, 폴더 구조
- Day 2 (07-22): pubg_client.py (매치 리스트 조회 + 429 backoff)
- Day 3 (07-23): download_telemetry.py (텔레메트리 원본 JSON 다운로드, data/raw/)
- Day 4 (07-26): telemetry_parser.py (LogGameStatePeriodic 파싱 + phase별 요약)
- Day 5 (07-26): collect_batch.py (배치 다운로드·파싱 자동화, 캐시·레이트리밋 대응)
- Day 6 (07-26): build_dataset.py (결측치/이상치 처리 + 맵명 정규화 → zones_dataset.csv)
- Day 7 (07-26): FastAPI 뼈대(main.py, /health), config.py, EDA 노트북, 주간 회고

### Week 2 (07-28 ~ 08-03) — DB 설계 + 베이스라인 모델
- Day 8 (07-28): docker-compose.yml(로컬 Postgres), database.py(엔진/세션), models.py(matches/circles/predictions/request_logs)
- Day 9 (07-29): Alembic 마이그레이션(0001_init), load_dataset.py (CSV → DB 적재)
- Day 10 (07-30): schemas.py(Pydantic), routers/predict.py·detect.py 스켈레톤, main.py 등록 (/docs 확인)
- Day 11 (07-31): analyze_patterns.py (단계별 이동벡터·축소비율 분석 → figures/ + phase_stats.json)
- Day 12 (08-01): baseline_model.py (규칙 기반 다음 원 추정기, phase_stats.json 사용)
- Day 13 (08-02): evaluate.py (매치 단위 train/test, 오차(m) 측정 → baseline_eval.md)
- Day 14 (08-03): run_pipeline.py (통합→통계→평가 한 번에 실행), README 결과 정리, 주간 회고 [v0.2]
