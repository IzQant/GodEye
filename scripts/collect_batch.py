"""
Day 5 작업: 배치 수집 자동화.
matchId 리스트를 순회하며 요청 간 대기(기본 6초)를 두고
텔레메트리 다운로드 + 파싱(phase 요약)을 한 번에 수행한다.

- 원본 JSON: data/raw/{matchId}.json (이미 있으면 재사용, 재다운로드 안 함)
- 파싱 결과: 매치별 phase 요약을 하나로 합쳐 반환 (Day 6에서 CSV로 통합)

완료 기준(Day 5): 매치 30개 이상 원본 다운로드 및 파싱 성공.

실행: python scripts/collect_batch.py [목표개수]   (기본 30)
"""
import glob
import json
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from dotenv import load_dotenv

from app.services.pubg_client import PubgClient
from app.services.telemetry_parser import parse_zone_events, summarize_phases

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# 매치 상세/텔레메트리 조회는 레이트리밋(분당 약 10회) 대상이므로
# 요청 사이에 넉넉히 6초를 둔다. 30개면 대략 3분 정도 걸린다.
REQUEST_DELAY = 6


def collect_match_ids(client: PubgClient, target: int) -> list[str]:
    """
    수집 대상 매치 ID를 모은다.
    1) 본인 계정 매치(있으면)
    2) /samples 무작위 매치 (대량 확보용, 보통 여기서 대부분 채워짐)
    중복은 제거하고 target 개수만큼 잘라 반환한다.
    """
    player_name = os.getenv("PUBG_PLAYER_NAME")
    match_ids: list[str] = []

    if player_name:
        try:
            match_ids += client.get_player_matches(player_name, limit=target)
        except Exception as e:
            print(f"[경고] 본인 매치 조회 실패(무시하고 진행): {e}")

    # /samples는 한 번에 수십~수백 개를 주므로 target을 넉넉히 채울 수 있다.
    try:
        match_ids += client.get_sample_matches(limit=target * 2)
    except Exception as e:
        print(f"[경고] /samples 조회 실패: {e}")

    # 순서 유지하며 중복 제거
    unique_ids = list(dict.fromkeys(match_ids))
    return unique_ids[:target]


def extract_map_name(telemetry: list[dict]) -> str:
    """LogMatchStart 이벤트에서 맵 이름 추출 (없으면 unknown)."""
    for event in telemetry:
        if event.get("_T") == "LogMatchStart":
            return event.get("mapName", "unknown")
    return "unknown"


def download_and_parse(client: PubgClient, match_id: str) -> pd.DataFrame | None:
    """
    매치 1개를 다운로드(캐시 있으면 재사용)하고 phase 요약 DataFrame으로 파싱한다.
    실패하면 None을 반환하고 호출부에서 실패로 집계한다.
    """
    out_path = os.path.join(RAW_DIR, f"{match_id}.json")

    # 이미 받은 원본이 있으면 API를 다시 부르지 않고 파일에서 읽는다(레이트리밋 절약).
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            telemetry = json.load(f)
        downloaded = False
    else:
        telemetry_url = client.get_telemetry_url(match_id)
        telemetry = client.download_telemetry(telemetry_url)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(telemetry, f)
        downloaded = True

    map_name = extract_map_name(telemetry)
    events_df = parse_zone_events(telemetry, match_id, map_name)
    phases_df = summarize_phases(events_df)

    tag = "다운로드" if downloaded else "캐시"
    print(f"[OK] {match_id} ({tag}) — {map_name}, phase {len(phases_df)}개")
    return phases_df


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    load_dotenv()
    api_key = os.getenv("PUBG_API_KEY")
    shard = os.getenv("PUBG_SHARD", "steam")
    if not api_key:
        print("PUBG_API_KEY가 .env에 설정되지 않았습니다.")
        return

    os.makedirs(RAW_DIR, exist_ok=True)
    client = PubgClient(api_key, shard)

    print(f"목표: 매치 {target}개 수집")
    match_ids = collect_match_ids(client, target)
    print(f"대상 매치 ID {len(match_ids)}개 확보\n")

    all_phases = []
    success = 0
    for i, match_id in enumerate(match_ids, 1):
        already_cached = os.path.exists(os.path.join(RAW_DIR, f"{match_id}.json"))
        try:
            phases_df = download_and_parse(client, match_id)
            if phases_df is not None and not phases_df.empty:
                all_phases.append(phases_df)
                success += 1
        except Exception as e:
            print(f"[FAIL] {match_id} — {e}")

        # 마지막 항목이 아니고, 이번에 실제로 API를 호출(다운로드)한 경우에만 대기
        if i < len(match_ids) and not already_cached:
            time.sleep(REQUEST_DELAY)

    print(f"\n총 {success}/{len(match_ids)}개 매치 파싱 성공")

    if all_phases:
        combined = pd.concat(all_phases, ignore_index=True)
        print(f"통합 phase 행 수: {len(combined)}")

    if success >= 30:
        print("✅ Day 5 완료 기준 통과: 매치 30개 이상 수집·파싱 성공")
    else:
        print(f"❌ 30개 미달({success}개) — 재실행하면 캐시 덕분에 이어서 채워진다")


if __name__ == "__main__":
    main()
