"""
FastAPI 엔트리포인트 (Day 7 뼈대 단계).

지금은 서비스가 살아있는지 확인하는 /health 엔드포인트만 있다.
예측/검출 라우터(predict, detect, analyze)는 이후 주차에서 추가한다.

로컬 실행:
    uvicorn app.main:app --reload
확인:
    http://127.0.0.1:8000/health  → {"status": "ok"} (200)
    http://127.0.0.1:8000/docs    → Swagger UI
"""
from fastapi import FastAPI

app = FastAPI(title="PUBG Zone Predictor")


@app.get("/health")
def health():
    """
    헬스체크 엔드포인트.
    지금은 단순히 200 OK만 반환한다.
    (Week 6 배포 시 DB 연결 확인까지 넣도록 확장 예정 — 부록 배포 체크리스트 참고)
    """
    return {"status": "ok"}
