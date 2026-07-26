"""
Day 4 완료 기준 확인용 스크립트.
data/raw/에 있는 매치 JSON 1개를 읽어 telemetry_parser로 파싱하고,
phase별로 정리된 DataFrame을 출력한다.

실행: python scripts/parse_match.py
"""
import glob
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.telemetry_parser import parse_zone_events, summarize_phases

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def load_first_match():
    """data/raw/*.json 중 첫 번째 파일을 읽어 (match_id, telemetry) 반환."""
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    if not files:
        return None, None

    path = files[0]
    match_id = os.path.splitext(os.path.basename(path))[0]
    with open(path, encoding="utf-8") as f:
        telemetry = json.load(f)
    return match_id, telemetry


def extract_map_name(telemetry: list[dict]) -> str:
    """LogMatchStart 이벤트에 맵 이름이 들어있다. 못 찾으면 unknown 반환."""
    for event in telemetry:
        if event.get("_T") == "LogMatchStart":
            return event.get("mapName", "unknown")
    return "unknown"


def main():
    match_id, telemetry = load_first_match()
    if not match_id:
        print("data/raw/에 매치 JSON이 없습니다. Day 3 스크립트를 먼저 실행하세요.")
        return

    print(f"매치 ID: {match_id}, 이벤트 수: {len(telemetry)}")

    map_name = extract_map_name(telemetry)
    print(f"맵: {map_name}")

    events_df = parse_zone_events(telemetry, match_id, map_name)
    print(f"\nLogGameStatePeriodic 이벤트 {len(events_df)}개 추출")

    phases_df = summarize_phases(events_df)
    print(f"\n=== 단계별 원 정보 DataFrame ({len(phases_df)}개 phase) ===")
    print(phases_df.to_string(index=False))

    if not phases_df.empty:
        print("\n✅ Day 4 완료 기준 통과: 단계별 원 정보 DataFrame 생성 확인")
    else:
        print("\n❌ phase가 비어 있음 — 텔레메트리 파일이나 파싱 로직 확인 필요")


if __name__ == "__main__":
    main()
