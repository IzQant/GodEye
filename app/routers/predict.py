"""
/api/predict 라우터 (Day 20: 실구현).

흐름: matchId → 현재 원 추출(match_service) → 모델 추론(model_service)
      → 다음 원 중심·반경·신뢰구간 JSON 응답.
"""
from fastapi import APIRouter, HTTPException, Request

from app.rate_limit import limiter
from app.schemas import Circle, PredictRequest, PredictResponse
from app.services.match_service import MatchNotFoundError, get_current_circle
from app.services.model_service import get_predictor

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
@limiter.limit("30/minute")
def predict_next_circle(request: Request, req: PredictRequest):
    """matchId를 받아 다음 자기장(중심·반경·신뢰구간)을 예측한다."""
    # 1) matchId → 현재 원 특징
    try:
        cur = get_current_circle(req.match_id)
    except MatchNotFoundError as e:
        # 사용자가 이해할 수 있는 404 메시지 (서버는 죽지 않는다)
        raise HTTPException(status_code=404, detail=str(e))

    # 2) 모델 로드(최초 1회) 및 추론
    #    파일 없음/역직렬화 실패(예: 라이브러리 버전 불일치) 등 모든 로드 오류를
    #    503으로 처리해, 모델 문제로 서버가 500을 내지 않도록 한다.
    try:
        predictor = get_predictor()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"모델이 아직 준비되지 않았습니다: {e}")

    out = predictor.predict(
        cur["safety_x"], cur["safety_y"], cur["safety_radius"],
        phase=cur["phase"], map_name=cur["map_name"], conf=0.95,
    )

    # 3) 응답 구성
    return PredictResponse(
        match_id=req.match_id,
        phase=cur["phase"],
        predicted=Circle(x=out["x"], y=out["y"], radius=out["radius"]),
        confidence_radius=out["confidence_radius"],
        model_name="RF(delta)+gaussian",
    )
