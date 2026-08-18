# Week 6 배포 기록 (Day 36~40)

주차 목표: 실제 인터넷 배포 + 모니터링 + 보안/안정성
플랫폼: Railway (Docker 빌드), Postgres addon

## 완료한 것
- Day 36: 프로덕션 Dockerfile(경량 런타임 requirements-runtime.txt, 빌드 시 train_final로
  모델 생성→sklearn 버전 불일치 원천 차단), .dockerignore, docker-compose app 서비스 활성화.
  → 로컬 docker build+run 성공.
- Day 37: Railway 배포. start.sh($PORT 바인딩 + 마이그레이션 타임아웃), render.yaml, DEPLOY.md.
  → 배포 URL /health 200.
- Day 38: DB 연결(DATABASE_URL=${{Postgres.DATABASE_URL}}), 마이그레이션 성공, request_logs 적재.
  CORS/정적경로는 same-origin+CDN이라 이슈 없음.
- Day 39: UptimeRobot /health 5분 모니터, scripts/load_test.py 동시요청 부하 테스트.
- Day 40: 보안/안정성 — 시크릿 노출 없음(.env gitignore/dockerignore, env로만 로드),
  레이트리밋 429 재확인, 잘못된 입력 500 없음(422/400/404/503), /health에 DB핑 추가
  (connect_timeout 3s, DB죽어도 200 유지).

## 배포 트러블슈팅 (겪은 것 → 해결)
- "application failed to respond"(502): 앱이 8000 고정 바인딩 → Railway는 $PORT(8080)로 라우팅.
  해결: start.sh가 $PORT에 바인딩, Dockerfile EXPOSE 8080/ENV PORT=8080으로 통일 +
  Railway 도메인 타겟 포트 8080 지정.
- 마이그레이션 localhost 연결 실패: DATABASE_URL이 Postgres 미참조.
  해결: 앱 Variables에 DATABASE_URL=${{Postgres.DATABASE_URL}} 추가.
- postgres:// 접두사: SQLAlchemy2.0은 postgresql://만 인식 → database.py/env.py에서 보정.
- curl TLS 에러: 로컬은 http://, 배포는 포트 없는 https://도메인.
- Pydantic model_name 경고: schemas에 protected_namespaces=() 설정으로 제거.

## 배포 환경 변수 (Railway)
- DATABASE_URL=${{Postgres.DATABASE_URL}} (참조)
- PUBG_API_KEY, PUBG_SHARD=steam, SECRET_KEY, WEB_CONCURRENCY=1

## 현재 상태
- 공개 URL(HTTPS)로 서비스 동작, /health 200(db:ok), 요청 로그 DB 적재, 레이트리밋/모니터링 적용.
- 로드맵 Day 41(문서화) 중 README만 완성. 발표자료·회고는 확장(Phase 5~7) 이후로 보류.

## 파일
- Dockerfile, requirements-runtime.txt, .dockerignore, start.sh, render.yaml, DEPLOY.md
- app/main.py(/health DB핑), app/database.py(connect_timeout, postgres:// 보정)
- scripts/load_test.py
