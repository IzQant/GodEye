# PUBG 자기장 예측 서비스 — 진행 기록 (Day 4~7)

기준 문서: 6주 구현 로드맵
기간: 2026-07-24(Day 4) ~ 2026-07-27(Day 7) · Week 1 후반부
작성일: 2026-07-27

## Day 4 (07-24) — 텔레메트리 파싱
- `app/services/telemetry_parser.py` 작성
  - `parse_zone_events()`: `_T == "LogGameStatePeriodic"` 이벤트만 필터링,
    현재 안전지대(safety)·다음 자기장(poison)의 좌표/반경을 이벤트 단위 DataFrame으로 변환
  - `summarize_phases()`: poison_radius가 바뀌는 지점을 기준으로 phase 번호를 매기고,
    phase별로 시작/종료 시간·좌표·반경을 요약
- `scripts/parse_match.py`로 실제 매치 검증 (Desert_Main, 7개 phase, 반경 축소 패턴 확인)
- **완료 기준 통과**: 매치 1개에서 단계별 원 정보 DataFrame 생성 확인

## Day 5 (07-25) — 배치 수집 자동화
- `scripts/collect_batch.py` 작성
  - 본인 매치 + `/samples` 조합으로 대상 ID 확보(중복 제거)
  - 매치 순회하며 다운로드 + 파싱, 원본 캐시 재사용, 요청 간 6초 대기(레이트리밋 대응)
  - 이번에 실제 API 호출한 경우에만 대기 → 캐시된 건 빠르게 스킵
- **완료 기준 통과**: 매치 30개 이상 수집·파싱 성공

## Day 6 (07-26) — 데이터셋 통합
- `scripts/build_dataset.py` 작성
  - `data/raw/`의 모든 매치 파싱 → 정제 → `data/processed/zones_dataset.csv` 통합
  - 정제: 맵명 정규화(코드명→읽기 쉬운 이름), 훈련장 제외, phase 0 제거,
    결측치·좌표 이상치(0~900000cm 범위) 제거
  - 맵명 매핑 실제 데이터로 검증 (Tiger→Taego, Summerland→Karakin, Neon→Rondo 등)
- 결과: 20매치 → 정제 후 138 phase 행 (187행 → 138행, 49행 제거)
- 맵별 분포: Sanhok 45, Erangel 39, Taego 30, Miramar 13, Rondo 6, Karakin 5
- **완료 기준 통과**: 단일 CSV로 통합 완료

## Day 7 (07-27, 월) — Week 1 버퍼/정리일
- `app/main.py`: FastAPI 뼈대, `/health` 엔드포인트 (부팅 후 200 검증 완료)
- `app/config.py`: `.env` 값을 전역 `settings`로 노출
- `ml/eda.ipynb`: EDA 노트북 4개 섹션 (맵별 분포, 반경 축소, 축소 비율, 이동 거리)
- `friday/raw/memories/week1_retrospective.md`: 주간 회고
- README 갱신 (아키텍처/구조/실행법/진행 로그)
- **완료 기준**: uvicorn `/health` 200 응답 ✅ / git tag v0.1은 본인이 직접

### EDA 핵심 발견
- 축소 비율(다음 반경/현재 반경) 평균 0.932, 표준편차 0.132
  → 단계마다 비교적 일정. 베이스라인(평균 비율 적용) 가정과 부합
- 중심 이동 거리: 초반 단계 거의 0, 중반(phase 5~7)에 최대 ~2100m
  → 중반 단계 예측이 더 어려움

## 이슈 및 해결
- (Day 4~5) `/samples` 대체 조회로 본인 매치 부재 문제 계속 우회
- (Day 7) git lock 관련 환경 제약 → git 작업은 본인이 직접 관리로 정리

## 다음 작업
- 데이터 20매치로 Week 1 목표(30+)에 약간 못 미침 →
  Week 2 시작 전 `python scripts/collect_batch.py 50`로 보강 권장
- Day 8 (07-28): PostgreSQL 스키마 + Docker Compose (Week 2 시작)
