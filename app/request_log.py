"""
요청 로깅 (Day 33).

모든 /api/* 요청과 그 결과(성공/에러)를 request_logs 테이블에 기록한다.
DB가 없거나 실패해도 요청 처리를 막지 않도록 best-effort로 동작한다(에러는 삼킴).
"""
from starlette.concurrency import run_in_threadpool


def _write(endpoint: str, input_type: str, status: str, error_message: str | None):
    """동기 DB 기록. 실패 시 조용히 무시(서비스 중단 방지)."""
    try:
        from app.database import SessionLocal
        from app.models import RequestLog
        db = SessionLocal()
        try:
            db.add(RequestLog(endpoint=endpoint, input_type=input_type,
                              status=status, error_message=error_message))
            db.commit()
        finally:
            db.close()
    except Exception as e:
        print(f"[request_log] DB 기록 실패(무시): {e}")


async def log_request_middleware(request, call_next):
    """/api/* 요청을 처리하고 결과를 로그로 남기는 미들웨어."""
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/api/"):
        # 입력 유형 추정: 쿼리/폼을 다시 읽지 않고 헤더의 content-type 정도로만 구분
        ctype = request.headers.get("content-type", "")
        input_type = "image" if "multipart" in ctype else "match_id"
        status = "success" if response.status_code < 400 else "error"
        err = None if status == "success" else f"HTTP {response.status_code}"
        await run_in_threadpool(_write, path, input_type, status, err)
    return response
