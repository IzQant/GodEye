# 진행 기록 — Day 7 (07-27, 월) [Week 1 버퍼/정리일]

작성일: 2026-07-27

## 오늘 한 것
- **FastAPI 뼈대** (`app/main.py`): `/health` 엔드포인트 작성.
  실제 부팅시켜 200 + `{"status":"ok"}` 반환 검증 완료.
  (Week 6 배포 시 DB 연결 확인까지 확장 예정)
- **환경설정 모듈** (`app/config.py`): `.env` 값을 읽어 전역 `settings`로 노출.
  DB/SECRET_KEY는 자리만 잡아두고 이후 주차에서 실제 사용.
- **EDA 노트북** (`ml/eda.ipynb`): 4개 섹션 — 맵별 phase 분포,
  phase별 반경 축소, 축소 비율 분포, 중심 이동 거리. 로직은 실제 데이터로 검증.
- **주간 회고** (`friday/raw/memories/week1_retrospective.md`) 작성.
- **README 갱신**: 아키텍처/구조/실행법/Week 1 진행 로그 정리.

## EDA 핵심 발견
- 축소 비율(다음 반경/현재 반경) 평균 0.932, 표준편차 0.132
  → 단계마다 비교적 일정하게 축소. 베이스라인 모델 가정과 부합.
- 중심 이동 거리: 초반 단계 거의 0, 중반(phase 5~7)에 최대 ~2100m
  → 중반 단계 예측이 더 어려움.

## 완료 기준
- uvicorn으로 /health 200 응답 ✅ (검증 완료)
- git tag v0.1 → 본인이 직접 남길 것 (git은 본인 관리)

## 남긴 것 / 다음 주로
- 데이터 20매치로 Week 1 목표(30+)에 약간 못 미침.
  Week 2 시작 전 `python scripts/collect_batch.py 50`로 보강 권장.
- 다음: Day 8 (07-28) — PostgreSQL 스키마 + Docker Compose (Week 2 시작)
