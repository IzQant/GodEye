# GodEye
배틀그라운드 실시간 자기장 예측 서비스

PUBG 매치 텔레메트리를 기반으로 **다음 자기장(안전지대)의 위치·반경을 예측**하는 웹 서비스.
FastAPI + scikit-learn + OpenCV로 만들었고, Docker로 Railway에 배포되어 공개 URL(HTTPS)로 동작한다.

## 기능
- 웹 UI(`/`): matchId 탭 / 이미지+단계 탭. 결과를 지도 위에 오버레이(또는 확률 히트맵)로 표시.
- 두 입력 경로:
  - **matchId**: PUBG API 텔레메트리 조회 → 현재 원 → 예측.
  - **이미지**: 전체 지도 화면 업로드 + 현재 phase·맵 입력 → (자동 검출/수동 지정) 현재 원 → 예측.
    사진 보정(지도 네 꼭짓점 4클릭 → 원근 변환)으로 모니터 촬영 사진도 처리.
- 예측 모델: **단계 전환(phase N→N+1)** 회귀(RandomForest) + 단계별 축소비율 + 불확실성.
  확률 히트맵(KDE)로 "다음 원이 있을 확률 분포"도 제공.
- API: `/health`, `/api/predict`, `/api/detect`, `/api/analyze`(통합), `/api/visualize`(오버레이 PNG),
  `/api/heatmap`(확률 히트맵 PNG), `/api/detect-corners`(사진 보정 보조).
- 안전장치: 레이트리밋(IP당 분당 30, 429), 요청 로깅(request_logs), 친화적 에러(422/400/404/503),
  `/health` DB 핑.

## 예측에 대한 정직한 설명
PUBG는 다음 원 중심을 현재 원 안에서 상당히 무작위로 정한다. 따라서 **중심 점 예측은 원리적 한계**가
있고(RF ≈ copy 기준선), 이 서비스의 가치는 (1) **반경 축소 예측**, (2) **확률 범위(히트맵)** 제시에 있다.
자세한 실험·근거는 `friday/raw/memories/` 문서들 참고(model_reframe_transition.md 등).

## 프로젝트 구조
```
app/        FastAPI 앱: main.py, config.py, database.py, models.py, schemas.py,
            rate_limit.py, request_log.py, routers/(predict,detect,analyze), services/, templates/
ml/         모델·분석: dataset_pairs.py, train_final.py, evaluate.py, heatmap_model.py,
            features.py, compare_algos.py, analyze_patterns.py ...
scripts/    데이터 수집·파싱·통합·부하테스트: collect_batch, build_dataset, load_dataset, run_pipeline ...
data/       raw/(원본 JSON, gitignore), processed/(zones_dataset.csv, 커밋됨), images/, maps/
alembic/    DB 마이그레이션
tests/      pytest
```

## 로컬 실행 (빠른 시작)
`data/processed/zones_dataset.csv`가 저장소에 포함되어 있어, **PUBG 키 없이도** 모델 학습 후 바로
띄울 수 있다(이미지·히트맵 경로 동작). matchId 예측만 PUBG API 키가 필요하다.
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # 전체 개발 의존성

python ml/train_final.py                 # 번들 CSV로 예측 모델(ml/models/predictor.joblib) 생성
uvicorn app.main:app --reload            # http://127.0.0.1:8000  (웹 UI / /docs)
```

### matchId 예측까지 쓰려면 (PUBG API)
```bash
cp .env.example .env    # 파일이 없으면 아래 값을 담아 .env 생성
#   PUBG_API_KEY=...     PUBG_SHARD=steam(또는 kakao)     PUBG_PLAYER_NAME=(선택)
```

### 데이터를 직접 더 모으려면
```bash
python scripts/collect_batch.py 300      # 텔레메트리 수집(중복 자동 스킵)
python scripts/build_dataset.py          # data/raw → data/processed/zones_dataset.csv
python ml/train_final.py                 # 모델 재학습(반드시 build 후)
```

### DB(요청 로깅)까지 쓰려면 (선택, Docker 필요)
```bash
docker compose up -d db                  # 로컬 Postgres
alembic upgrade head                     # 테이블 생성
# .env의 DATABASE_URL=postgresql://pubg:pubg@localhost:5432/pubg_zone
```

### 컨테이너로 실행
```bash
docker build -t godeye . && docker run --rm -p 8000:8000 --env-file .env godeye
# 또는 앱+DB 함께: docker compose up --build
```

## 배포
Railway(Docker) + Postgres. 단계별 안내는 `DEPLOY.md`. Render 대안은 `render.yaml`.
이미지 빌드 시 `train_final.py`로 모델을 생성해 라이브러리 버전 불일치를 방지한다.

## 데이터셋 현황 (2026-09-03 기준)
- `zones_dataset.csv`: **369매치, 2,807개 단계 기록, 2,438개 단계 전환쌍**.
- 맵별 매치: Erangel 131, Taego 126, Sanhok 36, Miramar 21, Vikendi 18,
  Rondo 17, Karakin 11, Paramo 9.
- 단계별 전환쌍: phase 1~3 각 369, phase 4 368, phase 5 351, phase 6 293,
  phase 7 210, phase 8 108, phase 9 1. 후반 단계는 여전히 표본이 부족하다.
- 현재 데이터 재평가 히트맵: coverage 50/80/90% 영역에서 실제 포함률 54.8/89.9/97.3%,
  균등분포 대비 로그가능도 **+2.137**(잘 보정됐지만 다소 보수적).
- 기존 단계 전환 중심 오차는 copy 159m / baseline 160m / RF 167m였으며,
  369매치 전체를 반영한 점 예측 모델 재학습·재평가는 아직 필요하다.
- 필요 매치 수 가이드: 후반 단계 신뢰도 확보는 전체 500~800매치,
  맵별 전용 모델은 맵당 100~150매치 이상. 현재는 Erangel/Taego만 최소 기준에 진입했다.

## 테스트
```bash
pytest -q     # 환경에 따라 모델/이미지 의존 테스트는 skip
```

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

### Week 5 (08-18 ~ 08-24) — 파이프라인 통합 + 프론트엔드
- Day 29 (08-18): pipeline.py (matchId/이미지 두 경로 → 동일 결과 스키마). 이미지 경로=전체지도+phase+맵+수동폴백. 예측기 주입형
- Day 30 (08-19): /api/analyze 통합 엔드포인트(입력검증+친화적 에러 422/400/404/503), test_analyze.py 5개 통과
- Day 31 (08-20): Jinja2+Bootstrap 프론트엔드(matchId/이미지 탭, 맵 선택, 결과 카드)
- Day 32 (08-21): 결과 지도 오버레이(visualize.py, /api/visualize PNG), 프론트 표시
- (개선) 수동 입력을 클릭+드래그 원 지정 + 사진 보정(4모서리 원근변환, perspective.py)
- (모델 재구성) 예측 목표를 단계 전환(phase N→N+1)으로 변경 → 초반 단계도 의미있는 축소·이동 예측.
- Day 33 (08-22): 레이트리밋(rate_limit.py, 데코레이터) + 요청 로깅(request_log.py, request_logs)
- Day 34 (08-23): E2E 3시나리오 테스트(test_e2e.py), 검출 실패 시 모델 로드 지연 수정
- Day 35 (08-24): UI 다듬기(로딩/버튼/429), README·발표개요, 주간 회고 [v0.5]
- (실험) 히트맵 예측(heatmap_model.py, /api/heatmap, 프론트 토글), 알고리즘 비교(RF/LGBM/XGB),
  특징 공학(features.py). 사진 보정 4클릭·자동감지(map_detect.py). model_algo_feature_experiment.md 등 기록

### Week 6 (08-25 ~ 08-31) — 배포·모니터링·문서화
- Day 36 (08-25): 프로덕션 Dockerfile(경량 런타임 + 빌드 시 모델 학습), requirements-runtime.txt, .dockerignore, compose app 서비스. docker build/run 성공
- Day 37 (08-26): Railway 배포(start.sh $PORT, DEPLOY.md, render.yaml). 배포 URL /health 200
- Day 38 (08-27): DB 연결(DATABASE_URL 참조) + 마이그레이션 + request_logs 적재. postgres:// 보정
- Day 39 (08-28): UptimeRobot /health 모니터, scripts/load_test.py 부하 테스트
- Day 40 (08-29): 보안/안정성(시크릿 점검, 레이트리밋·에러 재확인, /health DB핑+connect_timeout)
- Day 41 (08-30): 문서화 — README 완성. (발표자료·회고는 로드맵 확장 이후로 보류)
