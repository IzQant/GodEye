# Week 1 회고 (Day 1~7, 07-21 ~ 07-27)

주차 목표: 프로젝트 셋업 + 텔레메트리 수집·파싱 + FastAPI 뼈대
종료 산출물: 자기장 데이터셋(CSV) + v0.1
※ 원본 기록이 유실되어 대화 기록 기반으로 재작성 (2026-08-10).

## 완료한 것
- Day 1 (07-21): 프로젝트 셋업. PUBG API 키 발급, 부록 A 폴더 구조 생성,
  requirements.txt, .gitignore, scripts/test_api_key.py → 인증 200 OK.
- Day 2 (07-22): app/services/pubg_client.py. 매치 리스트 조회, 429 backoff,
  /samples 대체 조회(본인 계정 최근 14일 매치 없을 때).
- Day 3 (07-23): scripts/download_telemetry.py. 텔레메트리 원본 JSON 다운로드
  → data/raw/{matchId}.json. 매치 3개 확보.
- Day 4 (07-24): app/services/telemetry_parser.py. LogGameStatePeriodic 파싱,
  poison_radius 변화로 phase 구분 → summarize_phases로 단계별 요약.
  scripts/parse_match.py로 실제 매치 검증.
- Day 5 (07-25): scripts/collect_batch.py. 배치 다운로드+파싱 자동화,
  원본 캐시 재사용, 요청 간 6초 대기(레이트리밋 대응).
- Day 6 (07-26): scripts/build_dataset.py. 정제(맵명 정규화, 훈련장 제외,
  phase 0 제거, 결측/좌표 이상치 제거) → data/processed/zones_dataset.csv 통합.
- Day 7 (07-27, 버퍼일): app/main.py(/health 200), app/config.py,
  ml/eda.ipynb, README 초안, 주간 회고.

## 데이터셋 현황 (v0.1 시점)
- 고유 매치 20개 → 정제 후 138 phase 행
- 맵 분포: Sanhok 45, Erangel 39, Taego 30, Miramar 13, Rondo 6, Karakin 5

## EDA에서 확인한 것
- 축소 비율(다음 반경/현재 반경) 평균 0.932, 표준편차 0.132
  → 단계마다 비교적 일정하게 축소. 베이스라인(평균 비율 적용) 가정과 부합.
- 중심 이동 거리: 초반 단계(1~4)엔 거의 0(다음 원이 현재 원과 겹침, 동심원),
  중반(phase 5~7)에 커짐.
  ※ 단위 주의: 좌표/이동량은 cm. 미터는 ÷100 (unit_correction.md 참고).

## 이슈 및 해결
- /status는 shard 없이 호출해야 함(초기 404 원인, 수정 완료).
- 본인 계정 최근 14일 매치 없음 → /samples 대체 조회로 해결(API 스펙상 정상).
- 이 환경의 파일시스템 제약으로 git lock 정리 불가 → git 작업은 본인이 직접 관리로 정리.

## 주요 결정
- 작업 폴더: ~/Desktop/workspace/GodEye 고정.
- 기술 스택: 로드맵 확정(FastAPI, PostgreSQL, SQLAlchemy, scikit-learn, OpenCV, Docker).
- 진행 원칙: "작동하는 것 우선", 각 주 끝 git tag(v0.1~).

## 다음 주(Week 2)로
- 데이터 20매치로 목표(30+)에 약간 못 미침 → collect_batch.py로 보강 권장.
- Day 8~: PostgreSQL 스키마 + Docker Compose, 베이스라인 모델, 평가.

## 버전 태그
- v0.1 (본인이 직접 git tag)
