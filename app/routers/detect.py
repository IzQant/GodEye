"""
/api/detect 라우터 (Day 28: 실구현).

미니맵 스크린샷 업로드 → CircleDetector로 흰/파란 원 검출 →
픽셀 좌표/반경 + 신뢰도 기반 수동입력 필요 여부(needs_manual) 반환.
"""
import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.rate_limit import limiter
from app.schemas import Circle, DetectResponse
from app.services.circle_detector import CircleDetector

router = APIRouter(tags=["detect"])
_detector = CircleDetector()  # 검출기 1회 생성 후 재사용


@router.post("/detect", response_model=DetectResponse)
@limiter.limit("30/minute")
async def detect_circle(request: Request, image: UploadFile = File(..., description="미니맵 스크린샷")):
    """업로드된 이미지에서 흰/파란 원을 검출한다."""
    # 업로드 바이트 → OpenCV 이미지로 디코드
    data = await image.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다(형식 확인).")

    res = _detector.detect_with_confidence(img)

    def to_circle(c):
        return Circle(x=c["cx"], y=c["cy"], radius=c["r"]) if c else None

    return DetectResponse(
        safe=to_circle(res["safe"]),
        next_zone=to_circle(res["next"]),
        needs_manual=res["needs_manual"],
        reasons=res["reasons"],
    )
