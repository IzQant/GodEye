"""
PUBG Open API 클라이언트.
Day 2 작업: 시즌/매치 리스트 조회 + 요청 헤더 처리 + 429 backoff.
"""
import time
import requests

BASE_URL = "https://api.pubg.com/shards/{shard}"


class PubgClient:
    def __init__(self, api_key: str, shard: str = "steam"):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/vnd.api+json",
        })
        self.shard = shard

    def _get(self, url: str, retries: int = 3) -> dict:
        """429(레이트리밋) 응답 시 backoff 후 재시도."""
        for attempt in range(retries):
            resp = self.session.get(url)
            if resp.status_code == 429:
                wait = 8 * (attempt + 1)
                print(f"[429] rate limited, {wait}초 대기 후 재시도 ({attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("PUBG API rate limit exceeded after retries")

    def get_seasons(self) -> list[dict]:
        """현재 shard의 시즌 목록 조회."""
        url = f"{BASE_URL.format(shard=self.shard)}/seasons"
        data = self._get(url)
        return data["data"]

    def get_current_season_id(self) -> str:
        for season in self.get_seasons():
            if season["attributes"].get("isCurrentSeason"):
                return season["id"]
        raise ValueError("현재 시즌을 찾을 수 없음")

    def get_player_matches(self, player_name: str, limit: int = 5) -> list[str]:
        """플레이어 닉네임으로 최근 매치 ID 리스트 조회 (기본 최근 5개)."""
        url = f"{BASE_URL.format(shard=self.shard)}/players?filter[playerNames]={player_name}"
        data = self._get(url)
        matches = data["data"][0]["relationships"]["matches"]["data"]
        match_ids = [m["id"] for m in matches]
        return match_ids[:limit]

    def get_sample_matches(self, limit: int = 5) -> list[str]:
        """플레이어 기록과 무관하게 최근 24시간 내 무작위 매치 ID 조회.
        본인 계정에 최근 14일 내 플레이 기록이 없을 때 대체 데이터 소스로 사용."""
        url = f"{BASE_URL.format(shard=self.shard)}/samples"
        data = self._get(url)
        matches = data["data"]["relationships"]["matches"]["data"]
        match_ids = [m["id"] for m in matches]
        return match_ids[:limit]

    def get_telemetry_url(self, match_id: str) -> str:
        url = f"{BASE_URL.format(shard=self.shard)}/matches/{match_id}"
        data = self._get(url)
        for asset in data["included"]:
            if asset["type"] == "asset":
                return asset["attributes"]["URL"]
        raise ValueError("telemetry asset not found")

    def download_telemetry(self, telemetry_url: str) -> list[dict]:
        resp = self.session.get(telemetry_url)
        resp.raise_for_status()
        return resp.json()


if __name__ == "__main__":
    # Day 2 완료 기준 확인: 본인 계정 최근 매치 5개 리스트 조회 성공
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("PUBG_API_KEY")
    player_name = os.getenv("PUBG_PLAYER_NAME")
    shard = os.getenv("PUBG_SHARD", "steam")

    if not api_key or not player_name:
        print("PUBG_API_KEY / PUBG_PLAYER_NAME을 .env에 설정하세요.")
    else:
        client = PubgClient(api_key, shard)
        try:
            match_ids = client.get_player_matches(player_name)
            if not match_ids:
                print("본인 계정 최근 14일 내 매치 없음 → /samples로 대체 조회")
                match_ids = client.get_sample_matches()
            print(f"매치 {len(match_ids)}개 조회 성공:")
            for mid in match_ids:
                print(f" - {mid}")
        except Exception as e:
            print(f"조회 실패: {e}")
