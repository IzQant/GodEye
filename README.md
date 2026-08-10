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

### Week 3 (08-04 ~ 08-10) — 모델 고도화 + 예측 API
- Day 15 (08-04): train_regression.py (RandomForest 다음 원 예측, 56매치). 발견: 절대좌표 목표에선 copy-baseline이 더 강함 → Day 18에서 이동량(delta) 목표로 재구성 예정
- Day 16 (08-05): gaussian_model.py (이동량 2D 가우시안 근사 + 신뢰구간 반경, delta 프레이밍 적용)
- Day 17 (08-06, 선택): train_mlp.py (PyTorch MLP, 이동량 목표, 오차 평균 10.8m). Day 18에서 전 모델 동일 프레이밍 공정 비교 예정
- Day 18 (08-07): compare_models.py (4모델 공정 비교 + 지도 시각화). 최종 모델 RF(delta) 선정 (Copy 대비 마진 작음). → model_comparison.md
- Day 19 (08-08): train_final.py (최종 모델 joblib 직렬화), model_service.py (1회 로드 캐시 + 맵별 폴백 확장 여지)
- Day 20 (08-09): /api/predict 실구현 (match_service로 현재 원 추출 → 모델 추론 → JSON). 실 matchId 200 / 오류 matchId 404 확인
- Day 21 (08-10): pytest 7개(단위4+통합3) 통과, 예외처리(404/422/503), 주간 회고 [v0.3]

### Week 4 (08-11 ~ 08-17) — 컴퓨터 비전 원 검출
- Day 22 (08-11): data/images 구조(real/synthetic)+README, check_images.py(검증), make_synthetic_minimaps.py(합성 100장+정답, 다양한 해상도). real 20장은 본인 캡처 필요
- Day 23 (08-12): circle_detector.py (HSV 색상 필터링: BGR→HSV→inRange→모폴로지). show_masks.py 시각화, 합성 100장 마스크 분리 성공
- Day 24 (08-13): detect_circle/detect_circles (Contour+minEnclosingCircle), eval_detection.py. 합성 100장 검출 성공률 100%(중심0px/반경~2px). 실 스크린샷은 Day 25 폴백 대비
- Day 25 (08-14): 신뢰도(circularity)+detect_with_confidence(수동입력 폴백 분기), analyze_failures.py. clean 95%검출 / 어려운 변형 100% 폴백 트리거 확인
- Day 26 (08-15): coordinate_transform.py (픽셀→맵 좌표: 어파인 스케일/오프셋 + 호모그래피), verify_transform.py로 기준점 보정 후 오차 ~0 검증
- (데이터) make_map_overlays.py: 텔레메트리+실제 맵 이미지 오버레이로 준실사 라벨 미니맵 대량 생성 (data/maps/에 맵 이미지 필요)
- Day 27 (08-16): circle_detector를 CircleDetector 클래스로 정리, test_detector.py 5개 추가(전체 통과). 예측 라우터 모델오류 503 처리
- Day 28 (08-17): /api/detect 실구현(이미지 업로드→흰/파란 원 좌표+needs_manual), 주간 회고 [v0.4]
