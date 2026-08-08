"""
Day 13 작업: 베이스라인 모델 평가.

정직한 평가를 위해:
1. 매치 단위로 train/test 분리 (행 단위로 나누면 같은 매치가 양쪽에 섞여 누수 발생)
2. train 매치들로만 phase_stats(단계별 평균 통계)를 계산
3. test 매치의 각 행에 대해 "현재 원(safety)"을 입력해 다음 원을 예측하고,
   실제 다음 원(poison) 중심과의 유클리드 거리(m)를 오차로 측정

좌표 단위는 텔레메트리 기준 cm이므로, 거리(cm)를 100으로 나눠 m로 환산한다.

완료 기준(Day 13): 베이스라인 평균 오차(m) 수치 확보 및 기록.

실행: python ml/evaluate.py
"""
import os
import sys

sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd

from analyze_patterns import build_phase_stats, compute_features
from baseline_model import BaselineModel

BASE = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE, "..", "data", "processed", "zones_dataset.csv")
REPORT_PATH = os.path.join(BASE, "..", "friday", "raw", "memories", "baseline_eval.md")

CM_PER_M = 100.0
TEST_RATIO = 0.2
SEED = 42


def split_matches(df: pd.DataFrame):
    """고유 매치 ID를 섞어 train/test로 나눈다 (매치 단위 분리)."""
    rng = np.random.default_rng(SEED)
    match_ids = df["match_id"].unique()
    rng.shuffle(match_ids)
    n_test = max(1, int(len(match_ids) * TEST_RATIO))
    test_ids = set(match_ids[:n_test])
    train_ids = set(match_ids[n_test:])
    return train_ids, test_ids


def evaluate():
    df = compute_features(pd.read_csv(CSV_PATH))
    train_ids, test_ids = split_matches(df)

    train_df = df[df["match_id"].isin(train_ids)]
    test_df = df[df["match_id"].isin(test_ids)]

    # train 데이터로만 통계 생성 → 모델에 주입
    stats = build_phase_stats(train_df)
    model = BaselineModel.from_stats(stats)

    # test 각 행에 대해 예측하고 실제 poison 중심과의 거리 계산
    errors = []
    per_phase = {}
    for _, row in test_df.iterrows():
        pred = model.predict(row["safety_x"], row["safety_y"],
                             row["safety_radius"], int(row["phase"]))
        dist_cm = np.hypot(pred["x"] - row["poison_x"], pred["y"] - row["poison_y"])
        dist_m = dist_cm / CM_PER_M
        errors.append(dist_m)
        per_phase.setdefault(int(row["phase"]), []).append(dist_m)

    errors = np.array(errors)
    return {
        "n_train_matches": len(train_ids),
        "n_test_matches": len(test_ids),
        "n_test_rows": len(errors),
        "mean_m": float(errors.mean()),
        "median_m": float(np.median(errors)),
        "p90_m": float(np.percentile(errors, 90)),
        "per_phase": {p: float(np.mean(v)) for p, v in sorted(per_phase.items())},
    }


def write_report(r: dict):
    lines = [
        "# 베이스라인 모델 평가 리포트 (Day 13)",
        "",
        f"- 분리: 매치 단위 train/test (test 비율 {TEST_RATIO}, seed {SEED})",
        f"- train 매치 {r['n_train_matches']}개 / test 매치 {r['n_test_matches']}개 "
        f"(test 행 {r['n_test_rows']}개)",
        "",
        "## 오차 (다음 원 중심 예측, 단위 m)",
        f"- 평균: {r['mean_m']:.1f} m",
        f"- 중앙값: {r['median_m']:.1f} m",
        f"- 90퍼센타일: {r['p90_m']:.1f} m",
        "",
        "## 단계별 평균 오차 (m)",
    ]
    for p, e in r["per_phase"].items():
        lines.append(f"- phase {p}: {e:.1f} m")
    lines += [
        "",
        "## 해석 (중요)",
        "- 오차가 전반적으로 매우 작게 나온다(중앙값 <1m). 이는 모델이 뛰어나서가 아니라,",
        "  예측 대상인 다음 원(poison)이 같은 스냅샷에서 현재 원(safety) 기준으로",
        "  이미 게임이 발표한 값이고, 초반 단계는 자기장이 거의 동심원(중심 그대로,",
        "  반경만 축소)이어서 '다음=현재'로 찍어도 오차가 0에 가깝기 때문이다.",
        "- 실제로 의미 있는 이동/오차는 중후반(phase 5~7)에 집중된다.",
        "- 따라서 이 베이스라인은 '미래 예측'보다 '이미 드러난 다음 원 재현'에 가깝다.",
        "  Week 3에서 더 어렵고 유용한 목표(예: 한 단계 더 앞의 원 예측)로",
        "  재구성할 여지가 있다.",
        "- 이 수치는 Week 3 회귀/확률분포 모델과 비교할 기준선(baseline)이다.",
        "- 데이터가 20매치로 적어(test 4매치) 표본이 적은 단계의 수치는 변동이 크다.",
    ]
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    r = evaluate()
    print("=== 베이스라인 평가 결과 ===")
    print(f"train 매치 {r['n_train_matches']} / test 매치 {r['n_test_matches']} "
          f"(test 행 {r['n_test_rows']})")
    print(f"평균 오차:   {r['mean_m']:.1f} m")
    print(f"중앙값 오차: {r['median_m']:.1f} m")
    print(f"90%ile 오차: {r['p90_m']:.1f} m")
    print("\n단계별 평균 오차(m):")
    for p, e in r["per_phase"].items():
        print(f"  phase {p}: {e:.1f} m")

    write_report(r)
    print(f"\n리포트 저장: {REPORT_PATH}")
    print("✅ Day 13 완료 기준 통과: 베이스라인 평균 오차(m) 수치 확보 및 기록")


if __name__ == "__main__":
    main()
