# Week 2 회고 (Day 8~14, 07-28 ~ 08-03)

주차 목표: DB 설계 + 베이스라인 예측 모델 + 평가
종료 산출물: 베이스라인 모델 + 평가 리포트 + v0.2

## 완료한 것
- Day 8: docker-compose.yml(로컬 Postgres), database.py(엔진/세션), models.py(4개 테이블)
- Day 9: Alembic 마이그레이션(0001_init), load_dataset.py (CSV → DB 적재)
- Day 10: schemas.py(Pydantic), predict/detect 스켈레톤 라우터, /docs 등록
- Day 11: analyze_patterns.py (이동벡터·축소비율 → figures/ + phase_stats.json)
- Day 12: baseline_model.py (규칙 기반 다음 원 추정기)
- Day 13: evaluate.py (매치 단위 train/test, 오차(m) 측정)
- Day 14: run_pipeline.py (전체 파이프라인 한 번에 실행), README 정리, 회고

## 베이스라인 성능 (v0.2)
- 매치 단위 train 16 / test 4 (test 행 26)
- 오차: 평균 5.4m / 중앙값 0.5m / p90 10.9m
- 단계별: phase 1~4 ≈0.2~0.5m, phase 6~7 ≈14~28m

## 중요한 인식 (정직한 한계)
- 오차가 작은 것은 모델이 뛰어나서가 아님. 예측 대상(다음 원 poison)이 같은
  스냅샷에서 이미 게임이 발표한 값이고, 초반 단계는 동심원(중심 고정, 반경만 축소)
  이라 "다음=현재"로 찍어도 오차가 0에 가깝기 때문.
- 실질적 난이도는 중후반(phase 5~7)에 집중.
- Week 3에서 회귀/확률분포 모델로 고도화하되, "한 단계 더 앞 예측" 같은
  더 유용한 목표로 재구성할지 검토 여지 있음.

## 이슈 및 해결
- (Day 9) Postgres FK 엄격 검사로 적재 실패 → matches를 flush 후 circles 삽입으로 해결.
  교훈: DB 검증은 FK 검사를 켜고 해야 함(SQLite 기본값은 FK 미검사).

## 다음 주(Week 3)로
- Day 15~: scikit-learn 회귀 모델, 확률분포/신뢰구간, 모델 비교 리포트,
  최종 모델 직렬화, /api/predict 실제 구현
- 데이터 20매치로 여전히 적음 — 회귀 모델이 데이터 병목이면 특정 맵 집중 수집 고려

## 버전 태그
- v0.2 (본인이 직접 git tag)
