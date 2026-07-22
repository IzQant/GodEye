"""
Day 1 완료 기준 확인용 스크립트.
PUBG_API_KEY 발급 후 .env에 넣고 실행하면 /status 호출 결과를 출력한다.

실행: python scripts/test_api_key.py
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("PUBG_API_KEY")
SHARD = os.getenv("PUBG_SHARD", "steam")


def main():
    if not API_KEY:
        print("PUBG_API_KEY가 .env에 설정되지 않았습니다.")
        return

    # /status는 shard 없이 호출하는 엔드포인트
    url = "https://api.pubg.com/status"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/vnd.api+json",
    }
    resp = requests.get(url, headers=headers)
    print(f"status_code: {resp.status_code}")
    print(resp.text[:500])

    if resp.status_code == 200:
        print("\n✅ Day 1 완료 기준 통과: 인증 테스트 성공")
    else:
        print("\n❌ 인증 실패 — API 키 또는 shard 값을 확인하세요.")


if __name__ == "__main__":
    main()