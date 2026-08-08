"""
FastAPI 엔트리포인트.

- /health : 헬스체크
- /api/predict : matchId → 다음 자기장 예측 (Day 10 스켈레톤)
- /api/detect  : 스크린샷 → 원 검출 (Day 10 스켈레톤)

로컬 실행:
    uvicorn app.main:app --reload
확인:
    http://127.0.0.1:8000/health  → {"status": "ok"} (200)
    http://127.0.0.1:8000/docs    → Swagger UI (등록된 엔드포인트 확인)
"""
from fastapi import FastAPI

from app.routers import detect, predict

app = FastAPI(title="PUBG Zone Predictor")

# 예측/검출 라우터를 /api 접두사로 등록.
# analyze(통합) 라우터는 Week 5에서 추가한다.
app.include_router(predict.router, prefix="/api")
app.include_router(detect.router, prefix="/api")


@app.get("/health")
def health():
    """
    헬스체크 엔드포인트.
    지금은 단순히 200 OK만 반환한다.
    (Week 6 배포 시 DB 연결 확인까지 넣도록 확장 예정 — 부록 배포 체크리스트 참고)
    """
    return {"status": "ok"}
