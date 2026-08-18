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


# ---------- /api/analyze (통합) ----------
class AnalyzeResponse(BaseModel):
    """matchId/이미지 두 경로가 공유하는 통합 응답."""
    input_type: str = Field(..., description='"match_id" 또는 "image"')
    map: str | None = Field(None, description="맵 이름")
    phase: int | None = Field(None, description="단계")
    current: Circle | None = Field(None, description="현재 안전지대(맵 좌표)")
    predicted: Circle | None = Field(None, description="예측된 다음 자기장(맵 좌표)")
    confidence_radius: float | None = Field(None, description="예측 불확실성 반경")
    needs_manual: bool = Field(False, description="수동 좌표 입력이 필요한지")
    reasons: list[str] = Field(default_factory=list, description="needs_manual 사유")


# ---------- /api/detect ----------
class DetectResponse(BaseModel):
    """스크린샷에서 검출한 원(픽셀 좌표). 흰 원=현재, 파란 원=다음 자기장."""
    safe: Circle | None = Field(None, description="현재 안전지대(흰 원) 픽셀 좌표")
    next_zone: Circle | None = Field(None, description="다음 자기장(파란 원) 픽셀 좌표")
    needs_manual: bool = Field(
        ..., description="검출 신뢰도가 낮아 수동 좌표 입력이 필요한지"
    )
    reasons: list[str] = Field(
        default_factory=list, description="needs_manual인 경우 사유 목록"
    )
