"""
FastAPI 엔트리포인트.

- /health : 헬스체크
- /api/predict : matchId → 다음 자기장 예측
- /api/detect  : 스크린샷 → 원 검출
- /api/analyze : matchId 또는 (이미지+phase+맵) → 통합 예측

로컬 실행:
    uvicorn app.main:app --reload
확인:
    http://127.0.0.1:8000/health  → {"status": "ok"} (200)
    http://127.0.0.1:8000/docs    → Swagger UI (등록된 엔드포인트 확인)
"""
import os

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.rate_limit import limiter
from app.request_log import log_request_middleware
from app.routers import analyze, detect, predict
from app.services.coordinate_transform import MAP_SIZES_CM

app = FastAPI(title="PUBG Zone Predictor")

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# 레이트리밋: 각 엔드포인트에 @limiter.limit 데코레이터로 적용(라우터 참고).
# (SlowAPIMiddleware는 BaseHTTPMiddleware와 충돌하므로 데코레이터 방식 사용)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 요청 로깅 미들웨어(/api/* → request_logs, best-effort)
app.middleware("http")(log_request_middleware)

# 라우터를 /api 접두사로 등록.
app.include_router(predict.router, prefix="/api")
app.include_router(detect.router, prefix="/api")
app.include_router(analyze.router, prefix="/api")


@app.get("/")
def index(request: Request):
    """프론트엔드 페이지(matchId/이미지 입력 폼 + 결과 표시)."""
    # 최신 Starlette 시그니처: (request, 템플릿명, context)
    return templates.TemplateResponse(
        request, "index.html", {"maps": list(MAP_SIZES_CM)})


@app.get("/health")
def health():
    """
    헬스체크 엔드포인트.
    지금은 단순히 200 OK만 반환한다.
    (Week 6 배포 시 DB 연결 확인까지 넣도록 확장 예정 — 부록 배포 체크리스트 참고)
    """
    return {"status": "ok"}
