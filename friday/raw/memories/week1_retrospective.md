# Week 1 회고 (Day 1~7, 07-21 ~ 07-27)

주차 목표: 프로젝트 셋업 + 텔레메트리 수집·파싱 + FastAPI 뼈대
종료 산출물: 자기장 데이터셋(CSV) + v0.1

## 완료한 것
- Day 1: 프로젝트 셋업 (API 키, 폴더 구조, requirements.txt, 인증 테스트 200 OK)
- Day 2: pubg_client.py (매치 리스트 조회, 429 backoff, /samples 대체 조회)
- Day 3: download_telemetry.py (텔레메트리 원본 JSON 다운로드)
- Day 4: telemetry_parser.py (LogGameStatePeriodic 파싱 → phase별 요약)
- Day 5: collect_batch.py (배치 수집 자동화, 캐시·레이트리밋 대응)
- Day 6: build_dataset.py (정제 + zones_dataset.csv 통합)
- Day 7: FastAPI 뼈대(main.py, /health 200), config.py, EDA 노트북, 회고

## 데이터셋 현황 (v0.1 시점)
- 고유 매치 20개 → 정제 후 138 phase 행
- 맵별 분포: Sanhok 45, Erangel 39, Taego 30, Miramar 13, Rondo 6, Karakin 5

## EDA에서 확인한 것
- 축소 비율(다음 반경/현재 반경) 평균 0.932, 표준편차 0.132
  → 단계마다 대체로 일정하게 줄어듦. 베이스라인(평균 비율 적용) 가정과 부합.
- 중심 이동 거리는 초반 단계(1~4)엔 거의 0(다음 원이 현재 원과 겹침),
  중반(phase 5~7)에 최대 ~2100m로 커짐 → 중반 단계 예측이 더 어려움.

## 이슈 및 해결
- 본인 계정 최근 14일 매치 없음 → /samples 엔드포인트로 대체 (API 스펙상 정상 동작)
- git lock 파일 관련 환경 제약 → git 작업은 본인이 직접 관리하기로 정리
- /status는 shard 없이 호출해야 함 (초기 404 원인, 수정 완료)

## 다음 주(Week 2)로 넘길 것
- 데이터가 20매치라 목표(30+)에 약간 못 미침. Week 2 초반 또는 여유 시
  collect_batch.py 재실행으로 50매치까지 보강 권장.
- Day 8~: PostgreSQL 스키마 + Docker Compose, 베이스라인 모델, 평가 스크립트

## 버전 태그
- v0.1 (본인이 직접 git tag)
