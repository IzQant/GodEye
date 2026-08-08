"""
/api/detect 라우터 스켈레톤 (Day 10).

미니맵 스크린샷을 업로드받아 원(흰/파란)을 검출하는 엔드포인트의 뼈대.
실제 OpenCV 검출 로직은 Week 4(Day 28)에서 채운다.
지금은 파일을 받되 검출은 미구현 상태의 더미 응답을 돌려준다.
"""
from fastapi import APIRouter, File, UploadFile

from app.schemas import Circle, DetectResponse

router = APIRouter(tags=["detect"])


@router.post("/detect", response_model=DetectResponse)
async def detect_circle(image: UploadFile = File(..., description="미니맵 스크린샷")):
    """업로드된 이미지에서 원을 검출한다. (현재는 더미 응답)"""
    # TODO(Week 4): OpenCV 색상 필터 + Contour로 원 검출 → 픽셀→맵 좌표 변환
    return DetectResponse(
        pixel=Circle(x=0.0, y=0.0, radius=0.0),
        map_coord=None,
        detected=False,
        message=f"검출 미구현(stub). 수신 파일: {image.filename}",
    )
