"""
/api/predict 라우터 스켈레톤 (Day 10).

지금은 실제 예측 로직이 없다. 입력을 받아 형태만 맞춘 더미 응답을 돌려주고,
Swagger(/docs)에 엔드포인트가 등록되는 것까지가 오늘 목표.
실제 모델 연결은 Week 3(Day 20)에서 채운다.
"""
from fastapi import APIRouter

from app.schemas import Circle, PredictRequest, PredictResponse

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict_next_circle(req: PredictRequest):
    """matchId를 받아 다음 자기장을 예측한다. (현재는 더미 응답)"""
    # TODO(Week 3): matchId로 텔레메트리/DB 조회 → 특징 추출 → 모델 추론
    return PredictResponse(
        match_id=req.match_id,
        phase=0,
        predicted=Circle(x=0.0, y=0.0, radius=0.0),
        confidence_radius=None,
        model_name="baseline(stub)",
    )
