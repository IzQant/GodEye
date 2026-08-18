"""
간단한 동시 요청 부하 테스트 (Day 39).

배포된 서비스의 /health에 동시 요청을 보내 서버가 죽지 않는지, 응답이 안정적인지 확인.
(/health는 레이트리밋 대상이 아니므로 부하 확인에 적합. /api/* 는 분당 30회 제한이 걸림.)

사용:
  python scripts/load_test.py https://<도메인>/health --n 100 --concurrency 20

옵션:
  --n            총 요청 수 (기본 100)
  --concurrency  동시 실행 수 (기본 20)
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def one(url):
    t0 = time.perf_counter()
    try:
        r = requests.get(url, timeout=15)
        return r.status_code, (time.perf_counter() - t0) * 1000
    except Exception as e:
        return f"ERR:{type(e).__name__}", (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=20)
    opt = ap.parse_args()

    print(f"대상: {opt.url} | 총 {opt.n}요청 / 동시 {opt.concurrency}")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=opt.concurrency) as ex:
        results = list(ex.map(lambda _: one(opt.url), range(opt.n)))
    elapsed = time.perf_counter() - t0

    codes = [c for c, _ in results]
    lat = sorted(l for _, l in results)
    ok = sum(1 for c in codes if c == 200)
    from collections import Counter
    dist = dict(Counter(codes))

    def pct(p):
        return lat[min(len(lat) - 1, int(len(lat) * p))]

    print(f"\n소요: {elapsed:.1f}s | 처리율: {opt.n/elapsed:.1f} req/s")
    print(f"성공(200): {ok}/{opt.n} ({ok/opt.n*100:.0f}%)")
    print(f"상태 분포: {dist}")
    print(f"지연(ms) p50={pct(0.5):.0f}  p95={pct(0.95):.0f}  max={lat[-1]:.0f}")
    if ok == opt.n:
        print("\n✅ 전부 성공 — 부하 중 서버 다운 없음")
    else:
        print("\n⚠️ 일부 실패 — 상태 분포 확인(무료 티어 슬립/콜드스타트 가능)")


if __name__ == "__main__":
    main()
