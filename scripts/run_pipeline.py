"""
Day 14 작업: Week 1~2 전체 파이프라인 통합 실행기.

개별 스크립트를 정해진 순서로 한 번에 실행해 "수집 → 정제 → 통계 → 평가"가
끊김 없이 돌아가는지 점검한다.

기본 실행(오프라인, 네트워크/DB 불필요) — 반복 가능한 통합 점검용:
    python scripts/run_pipeline.py
    (build_dataset → analyze_patterns → evaluate)

옵션:
    --collect N   앞단에 텔레메트리 배치 수집을 붙인다 (PUBG API 필요, 느림)
    --load-db     정제 후 CSV를 DB에 적재하는 단계를 포함한다 (Postgres 필요)

각 단계는 별도 프로세스로 실행하며, 하나라도 실패하면 즉시 멈춘다.
"""
import argparse
import os
import subprocess
import sys

BASE = os.path.dirname(__file__)
ROOT = os.path.join(BASE, "..")
PY = sys.executable  # 현재 venv의 파이썬을 그대로 사용


def run(step_name: str, args: list[str]):
    """한 단계를 하위 프로세스로 실행하고, 실패하면 파이프라인을 중단."""
    print(f"\n{'=' * 60}\n▶ {step_name}\n{'=' * 60}")
    result = subprocess.run([PY] + args, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n❌ 단계 실패: {step_name} (exit {result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", type=int, metavar="N",
                        help="앞단에 매치 N개 배치 수집 추가 (PUBG API 필요)")
    parser.add_argument("--load-db", action="store_true",
                        help="정제 후 CSV를 DB에 적재 (Postgres 필요)")
    opts = parser.parse_args()

    # 1) (선택) 텔레메트리 배치 수집
    if opts.collect:
        run(f"배치 수집 ({opts.collect}개)",
            ["scripts/collect_batch.py", str(opts.collect)])

    # 2) 데이터셋 통합 (raw JSON → 정제 CSV)
    run("데이터셋 통합 (build_dataset)", ["scripts/build_dataset.py"])

    # 3) (선택) DB 적재
    if opts.load_db:
        run("DB 적재 (load_dataset)", ["scripts/load_dataset.py"])

    # 4) 단계별 통계 생성 (phase_stats.json + 그래프)
    run("패턴 분석 (analyze_patterns)", ["ml/analyze_patterns.py"])

    # 5) 베이스라인 평가 (오차 리포트)
    run("베이스라인 평가 (evaluate)", ["ml/evaluate.py"])

    print(f"\n{'=' * 60}")
    print("✅ 전체 파이프라인 실행 성공 (Day 14 통합 점검 통과)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
