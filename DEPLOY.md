# 배포 가이드 (Day 37~)

Docker 이미지를 클라우드(무료 티어)에 올려 공개 URL로 서비스한다.
두 플랫폼 중 하나를 고른다. 둘 다 이 저장소의 Dockerfile을 그대로 사용한다.

사전 준비
- 저장소가 GitHub에 올라가 있어야 한다(대용량 데이터는 gitignore된 상태).
- data/processed/zones_dataset.csv 는 커밋되어 있어야 한다(빌드 시 모델 학습에 사용).

환경변수(둘 다 공통)
- PUBG_API_KEY : PUBG 개발자 API 키 (시크릿)
- PUBG_SHARD   : steam (카카오면 kakao)
- SECRET_KEY   : 임의 문자열(자동 생성 가능)
- WEB_CONCURRENCY : 1 (무료 티어 메모리 절약)
- DATABASE_URL : 플랫폼의 Postgres가 제공(직접 입력 X)

------------------------------------------------------------------
## A. Railway (권장)
1. https://railway.app 로그인 → New Project → Deploy from GitHub repo → GodEye 선택.
   (Railway가 Dockerfile을 자동 감지해 빌드)
2. New → Database → PostgreSQL 추가.
3. 앱 서비스 → Variables 에서:
   - DATABASE_URL = ${{Postgres.DATABASE_URL}}  (Postgres 서비스 참조)
   - PUBG_API_KEY, PUBG_SHARD, SECRET_KEY, WEB_CONCURRENCY=1 입력.
4. 앱 서비스 → Settings → Networking → Generate Domain 으로 공개 URL 발급.
5. 배포 완료 후 https://<도메인>/health → {"status":"ok"} 확인.

## B. Render (대안)
1. https://render.com → New → Blueprint → 이 저장소 선택.
   (render.yaml이 웹 서비스 + Postgres를 함께 생성)
2. PUBG_API_KEY 는 대시보드에서 직접 입력(sync:false).
3. 배포 후 https://<서비스>.onrender.com/health 확인.

------------------------------------------------------------------
동작 방식(참고)
- 컨테이너 시작 시 start.sh 가 `alembic upgrade head`(마이그레이션)를 먼저 시도하고,
  gunicorn을 $PORT에 바인딩해 실행한다. 마이그레이션 실패해도 서비스는 뜬다.
- 예측 모델(predictor.joblib)은 이미지 빌드 중 train_final.py로 생성되므로
  별도 업로드가 필요 없다.

주의
- 무료 티어는 유휴 시 슬립할 수 있다(첫 요청이 느림). Day 39 UptimeRobot으로 완화.
- CORS/도메인/정적경로 이슈는 Day 38에서 점검.
