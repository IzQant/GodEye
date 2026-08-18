"""
/api/analyze 통합 엔드포인트 (Day 30).

한 엔드포인트로 두 입력을 받는다(multipart/form-data):
- matchId 경로 : match_id 필드
- 이미지 경로  : image(전체 지도 스크린샷) + phase + map_name (+ 선택 수동 원좌표)

입력 검증과 사용자 친화적 에러:
- 아무 입력 없음 → 422
- 이미지인데 phase/map 누락 → 422
- 못 읽는 이미지 → 400
- 알 수 없는 맵 → 400
- 매치/텔레메트리 없음 → 404
- 모델 미준비(로드 실패) → 503
"""
import os

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from app.rate_limit import limiter
from app.schemas import AnalyzeResponse, Circle
from app.services import perspective, pipeline, visualize
from app.services.coordinate_transform import MAP_SIZES_CM
from app.services.match_service import MatchNotFoundError

router = APIRouter(tags=["analyze"])

MAPS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "maps")


def _parse_corners(corners: str | None):
    """"x1,y1;x2,y2;x3,y3;x4,y4"(TL,TR,BR,BL) → [(x,y)*4] 또는 None."""
    if not corners:
        return None
    try:
        pts = [tuple(float(v) for v in pair.split(",")) for pair in corners.split(";")]
        if len(pts) == 4:
            return pts
    except Exception:
        pass
    raise HTTPException(status_code=400, detail="corners 형식 오류(x1,y1;x2,y2;x3,y3;x4,y4)")


def _decode_and_warp(data, corners, manual):
    """
    업로드 바이트 → 이미지. corners가 있으면 원근 보정(정사각 전체지도)하고
    manual 원좌표도 함께 보정 좌표로 변환. 반환: (image, manual_pixel|None)
    """
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다(형식 확인).")
    pts = _parse_corners(corners)
    if pts is not None:
        img, M = perspective.warp_map(img, pts)
        if manual is not None:
            manual = perspective.warp_circle(M, manual["cx"], manual["cy"], manual["r"])
    return img, manual


def _to_response(res: dict) -> AnalyzeResponse:
    def circle(c):
        return Circle(x=c["x"], y=c["y"], radius=c["radius"]) if c else None
    return AnalyzeResponse(
        input_type=res["input_type"], map=res["map"], phase=res["phase"],
        current=circle(res["current"]), predicted=circle(res["predicted"]),
        confidence_radius=res["confidence_radius"],
        needs_manual=res["needs_manual"], reasons=res["reasons"],
    )


@router.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit("30/minute")
async def analyze(
    request: Request,
    match_id: str | None = Form(None),
    image: UploadFile | None = File(None),
    phase: int | None = Form(None),
    map_name: str | None = Form(None),
    manual_cx: float | None = Form(None),
    manual_cy: float | None = Form(None),
    manual_r: float | None = Form(None),
    corners: str | None = Form(None),
):
    try:
        # --- matchId 경로 ---
        if match_id:
            return _to_response(pipeline.analyze_by_match_id(match_id))

        # --- 이미지 경로 ---
        if image is not None:
            if phase is None or not map_name:
                raise HTTPException(
                    status_code=422,
                    detail="이미지 예측에는 phase와 map_name이 필요합니다.")
            if map_name not in MAP_SIZES_CM:
                raise HTTPException(
                    status_code=400,
                    detail=f"알 수 없는 맵입니다: {map_name}. 지원: {list(MAP_SIZES_CM)}")

            manual = None
            if manual_cx is not None and manual_cy is not None and manual_r is not None:
                manual = {"cx": manual_cx, "cy": manual_cy, "r": manual_r}

            img, manual = _decode_and_warp(await image.read(), corners, manual)
            res = pipeline.analyze_by_image(img, phase=phase, map_name=map_name,
                                            manual_pixel=manual)
            return _to_response(res)

        # --- 아무 입력 없음 ---
        raise HTTPException(status_code=422,
                            detail="match_id 또는 image(+phase,map_name)를 제공하세요.")

    except HTTPException:
        raise
    except MatchNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"모델이 준비되지 않았습니다: {e}")
    except Exception as e:
        # joblib 버전 불일치 등 모델 로드 실패도 503으로(서버가 500 내지 않도록)
        if "predict" in str(e).lower() or "joblib" in str(e).lower() or "attribute" in str(e).lower():
            raise HTTPException(status_code=503, detail=f"모델 로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"분석 중 오류: {e}")


@router.post("/visualize")
@limiter.limit("30/minute")
async def analyze_visual(
    request: Request,
    match_id: str | None = Form(None),
    image: UploadFile | None = File(None),
    phase: int | None = Form(None),
    map_name: str | None = Form(None),
    manual_cx: float | None = Form(None),
    manual_cy: float | None = Form(None),
    manual_r: float | None = Form(None),
    corners: str | None = Form(None),
):
    """analyze와 동일 입력. 결과를 지도 위에 오버레이한 PNG로 반환."""
    try:
        base = None
        if match_id:
            res = pipeline.analyze_by_match_id(match_id)
        elif image is not None:
            if phase is None or not map_name:
                raise HTTPException(status_code=422, detail="phase와 map_name이 필요합니다.")
            manual = None
            if manual_cx is not None and manual_cy is not None and manual_r is not None:
                manual = {"cx": manual_cx, "cy": manual_cy, "r": manual_r}
            base, manual = _decode_and_warp(await image.read(), corners, manual)
            res = pipeline.analyze_by_image(base, phase=phase, map_name=map_name,
                                            manual_pixel=manual)
        else:
            raise HTTPException(status_code=422, detail="match_id 또는 image를 제공하세요.")

        if res["predicted"] is None:
            raise HTTPException(status_code=422,
                                detail="예측 불가(수동 좌표 입력 필요). /api/analyze로 상태 확인.")

        # 배경 확보: 이미지 경로는 업로드 이미지, matchId 경로는 맵 이미지(없으면 회색 배경)
        if base is None:
            map_path = os.path.join(MAPS_DIR, f"{res['map'].lower()}.png")
            base = cv2.imread(map_path)
            if base is None:
                base = visualize.blank_base()

        png = visualize.draw_overlay(base, res["current"], res["predicted"],
                                     res["confidence_radius"], res["map"])
        return Response(content=png, media_type="image/png")

    except HTTPException:
        raise
    except MatchNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        if any(k in str(e).lower() for k in ("predict", "joblib", "attribute")):
            raise HTTPException(status_code=503, detail=f"모델 로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"시각화 오류: {e}")
