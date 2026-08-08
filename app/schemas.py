"""
Pydantic 요청/응답 스키마 (Day 10).

FastAPI는 이 스키마로 (1) 입력 검증, (2) 응답 형태 고정, (3) Swagger 문서 자동 생성을 한다.
아직 실제 로직은 없고, 예측/검출 엔드포인트가 주고받을 데이터의 "형태"만 정의한다.
"""
from pydantic import BaseModel, Field


# ---------- 공통 ----------
class Circle(BaseModel):
    """자기장 원 하나: 중심좌표(x, y)와 반경. 좌표 단위는 텔레메트리 기준(cm)."""
    x: float
    y: float
    radius: float


# ---------- /api/predict ----------
class PredictRequest(BaseModel):
    """matchId를 받아 그 매치의 최신 상태로 다음 자기장을 예측한다."""
    match_id: str = Field(..., description="PUBG 매치 ID")


class PredictResponse(BaseModel):
    match_id: str
    phase: int = Field(..., description="예측 기준이 된 현재 단계")
    predicted: Circle = Field(..., description="예측된 다음 자기장(중심·반경)")
    # 확률분포/신뢰구간은 Week 3에서 채운다. 지금은 선택 필드로만 자리 확보.
    confidence_radius: float | None = Field(
        None, description="예측 중심의 불확실성 반경(신뢰구간). 미구현 시 null"
    )
    model_name: str = Field("baseline", description="사용한 예측 모델 이름")


# ---------- /api/detect ----------
class DetectResponse(BaseModel):
    """스크린샷에서 검출한 원. 픽셀 좌표와 (변환 가능하면) 맵 좌표를 함께 반환."""
    pixel: Circle = Field(..., description="화면 픽셀 기준 좌표")
    map_coord: Circle | None = Field(
        None, description="맵 좌표계로 변환한 값. 변환 실패 시 null"
    )
    detected: bool = Field(..., description="검출 성공 여부")
    message: str | None = Field(None, description="실패 사유 등 부가 메시지")
