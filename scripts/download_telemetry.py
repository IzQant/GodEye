"""
Day 3 작업: 매치 상세 → 텔레메트리 URL 획득 → JSON 다운로드.
완료 기준: 매치 3개 텔레메트리 원본 다운로드 완료 (data/raw/{matchId}.json).

실행: python scripts/download_telemetry.py
"""
import json
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

from app.services.pubg_client import PubgClient

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def get_match_ids(client: PubgClient, player_name: str | None, limit: int = 3) -> list[str]:
    """본인 매치가 있으면 그걸 쓰고, 없으면 /samples로 대체."""
    match_ids: list[str] = []
    if player_name:
        match_ids = client.get_player_matches(player_name, limit=limit)
    if not match_ids:
        print("본인 계정 최근 매치 없음(또는 미설정) → /samples로 대체 조회")
        match_ids = client.get_sample_matches(limit=limit)
    return match_ids


def download_matches(client: PubgClient, match_ids: list[str]) -> int:
    os.makedirs(RAW_DIR, exist_ok=True)
    success_count = 0

    for match_id in match_ids:
        out_path = os.path.join(RAW_DIR, f"{match_id}.json")
        if os.path.exists(out_path):
            print(f"[SKIP] {match_id} — 이미 다운로드됨")
            success_count += 1
            continue

        try:
            telemetry_url = client.get_telemetry_url(match_id)
            telemetry = client.download_telemetry(telemetry_url)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(telemetry, f)
            print(f"[OK] {match_id} — {len(telemetry)}개 이벤트 저장 완료")
            success_count += 1
        except Exception as e:
            print(f"[FAIL] {match_id} — {e}")

        time.sleep(1)  # 매치 상세 조회는 레이트리밋 대상이므로 최소한의 간격을 둔다

    return success_count


def main():
    load_dotenv()
    api_key = os.getenv("PUBG_API_KEY")
    player_name = os.getenv("PUBG_PLAYER_NAME")
    shard = os.getenv("PUBG_SHARD", "steam")

    if not api_key:
        print("PUBG_API_KEY가 .env에 설정되지 않았습니다.")
        return

    client = PubgClient(api_key, shard)
    match_ids = get_match_ids(client, player_name, limit=3)

    if not match_ids:
        print("다운로드할 매치를 찾지 못했습니다.")
        return

    print(f"대상 매치 {len(match_ids)}개: {match_ids}")
    success_count = download_matches(client, match_ids)

    print(f"\n총 {success_count}/{len(match_ids)}개 매치 다운로드 완료")
    if success_count >= 3:
        print("✅ Day 3 완료 기준 통과")
    else:
        print("❌ 3개 미달 — 위 [FAIL] 로그 확인 필요")


if __name__ == "__main__":
    main()
