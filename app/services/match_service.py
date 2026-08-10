"""
Day 20 작업: matchId → '현재 원' 특징 추출 서비스.

/api/predict가 쓰는 중간 단계.
matchId를 받아 텔레메트리를 확보(로컬 캐시 우선, 없으면 PUBG API)하고,
가장 마지막(=가장 최근) 단계의 안전지대(safety)를 '현재 원'으로 뽑아
모델 입력 특징으로 돌려준다.

레이트리밋 절약을 위해 data/raw/{matchId}.json 캐시가 있으면 API를 부르지 않는다.
"""
import json
import os

from app.config import settings
from app.services.pubg_client import PubgClient
from app.services.telemetry_parser import parse_zone_events, summarize_phases

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

# 맵 코드명 → 정규화 이름 (build_dataset.py와 동일해야 모델 입력이 일치)
MAP_NAME_KO = {
    "Baltic_Main": "Erangel",
    "Desert_Main": "Miramar",
    "Savage_Main": "Sanhok",
    "DihorOtok_Main": "Vikendi",
    "Range_Main": "TrainingRange",
    "Summerland_Main": "Karakin",
    "Chimera_Main": "Paramo",
    "Tiger_Main": "Taego",
    "Kiki_Main": "Deston",
    "Neon_Main": "Rondo",
}


class MatchNotFoundError(Exception):
    """텔레메트리를 확보하지 못했거나 자기장 정보가 없을 때."""


def _load_telemetry(match_id: str) -> list[dict]:
    """캐시(data/raw) 우선, 없으면 PUBG API로 텔레메트리 확보."""
    cache_path = os.path.join(RAW_DIR, f"{match_id}.json")
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    if not settings.PUBG_API_KEY:
        raise MatchNotFoundError("캐시에 없고 PUBG_API_KEY도 없어 조회할 수 없습니다.")

    client = PubgClient(settings.PUBG_API_KEY, settings.PUBG_SHARD)
    try:
        url = client.get_telemetry_url(match_id)
        telemetry = client.download_telemetry(url)
    except Exception as e:
        raise MatchNotFoundError(f"텔레메트리 조회 실패: {e}")

    # 다음 요청을 위해 캐시에 저장
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(telemetry, f)
    return telemetry


def _extract_map(telemetry: list[dict]) -> str:
    for event in telemetry:
        if event.get("_T") == "LogMatchStart":
            code = event.get("mapName", "unknown")
            return MAP_NAME_KO.get(code, code)
    return "unknown"


def get_current_circle(match_id: str) -> dict:
    """
    matchId → 가장 최근 단계의 현재 원(safety) 특징.
    반환: {safety_x, safety_y, safety_radius, phase, map_name}
    """
    telemetry = _load_telemetry(match_id)
    map_name = _extract_map(telemetry)

    events = parse_zone_events(telemetry, match_id, map_name)
    phases = summarize_phases(events)
    if phases.empty:
        raise MatchNotFoundError("자기장 정보를 찾을 수 없습니다.")

    # 가장 마지막(최근) 단계를 '현재'로 사용
    last = phases.sort_values("phase").iloc[-1]
    return {
        "safety_x": float(last["safety_x"]),
        "safety_y": float(last["safety_y"]),
        "safety_radius": float(last["safety_radius"]),
        "phase": int(last["phase"]),
        "map_name": map_name,
    }
