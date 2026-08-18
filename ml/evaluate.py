"""
베이스라인/모델 평가 — 단계 전환(phase N → N+1) 목표 기준.

[재구성] 예측 목표를 단계 전환으로 바꾼 뒤의 평가.
- 매치 단위 train/test 분리(누수 방지)
- train 전환쌍으로 phase별 평균 이동/축소 통계(=베이스라인) 및 RF 학습
- test 전환쌍에서 예측 중심과 실제 다음 원 중심의 거리(m) 측정

실행: python ml/evaluate.py
"""
import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.dataset_pairs import build_transition_pairs

BASE = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE, "..", "data", "processed", "zones_dataset.csv")
REPORT_PATH = os.path.join(BASE, "..", "friday", "raw", "memories", "baseline_eval.md")

CM_PER_M = 100.0
TEST_RATIO = 0.2
SEED = 42
FEATURES = ["map", "safety_x", "safety_y", "safety_radius", "phase"]


def split_matches(pairs):
    rng = np.random.default_rng(SEED)
    ids = pairs["match_id"].unique()
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * TEST_RATIO))
    test_ids = set(ids[:n_test])
    return pairs[~pairs["match_id"].isin(test_ids)], pairs[pairs["match_id"].isin(test_ids)]


def evaluate():
    pairs = build_transition_pairs(pd.read_csv(CSV_PATH))
    train, test = split_matches(pairs)

    # 베이스라인: phase별 평균 이동량 (train)
    means = train.groupby("phase")[["dx", "dy"]].mean()
    gmean = train[["dx", "dy"]].mean()

    def base_delta(ph):
        return (means.loc[ph] if ph in means.index else gmean)

    # RF: 특징 → 이동량
    rf = Pipeline([
        ("pre", ColumnTransformer([
            ("map", OneHotEncoder(handle_unknown="ignore"), ["map"]),
            ("num", "passthrough", ["safety_x", "safety_y", "safety_radius", "phase"]),
        ])),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)),
    ])
    rf.fit(train[FEATURES], train[["dx", "dy"]].values)

    # test 예측/오차
    copy_err, base_err, rf_err, per_phase = [], [], [], {}
    rf_pred = rf.predict(test[FEATURES])
    for i, (_, r) in enumerate(test.iterrows()):
        ax, ay = r["safety_x"] + r["dx"], r["safety_y"] + r["dy"]  # 실제 다음 중심
        # copy: 다음=현재
        copy_err.append(np.hypot(r["safety_x"] - ax, r["safety_y"] - ay) / CM_PER_M)
        # baseline: 현재 + phase 평균 이동
        bd = base_delta(int(r["phase"]))
        base_err.append(np.hypot(r["safety_x"] + bd["dx"] - ax, r["safety_y"] + bd["dy"] - ay) / CM_PER_M)
        # rf
        e = np.hypot(r["safety_x"] + rf_pred[i, 0] - ax, r["safety_y"] + rf_pred[i, 1] - ay) / CM_PER_M
        rf_err.append(e)
        per_phase.setdefault(int(r["phase"]), []).append(e)

    def stat(a):
        a = np.array(a)
        return a.mean(), np.median(a), np.percentile(a, 90)

    return {
        "n_train": len(train), "n_test": len(test),
        "copy": stat(copy_err), "base": stat(base_err), "rf": stat(rf_err),
        "per_phase": {p: float(np.mean(v)) for p, v in sorted(per_phase.items())},
    }


def main():
    r = evaluate()
    print("=== 단계 전환 예측 평가 (다음 원 중심 오차, m) ===")
    print(f"train 전환쌍 {r['n_train']} / test {r['n_test']}\n")
    print(f"{'model':<10}{'mean':>9}{'median':>9}{'p90':>9}")
    for name, key in (("copy", "copy"), ("baseline", "base"), ("RF", "rf")):
        m, md, p9 = r[key]
        print(f"{name:<10}{m:>9.1f}{md:>9.1f}{p9:>9.1f}")
    print("\nRF 단계별 평균 오차(m):")
    for p, e in r["per_phase"].items():
        print(f"  phase {p}: {e:.1f}")

    lines = [
        "# 모델 평가 리포트 (단계 전환 phase N→N+1 목표)",
        "",
        f"- 매치 단위 train/test (seed {SEED}), train 전환쌍 {r['n_train']} / test {r['n_test']}",
        "- 지표: 예측한 다음 원 중심과 실제 다음 원 중심의 거리(m)",
        "",
        "## 모델별 오차(m)",
        "| 모델 | 평균 | 중앙값 | p90 |",
        "|------|-----:|------:|----:|",
        f"| copy(다음=현재) | {r['copy'][0]:.1f} | {r['copy'][1]:.1f} | {r['copy'][2]:.1f} |",
        f"| baseline(단계평균) | {r['base'][0]:.1f} | {r['base'][1]:.1f} | {r['base'][2]:.1f} |",
        f"| RF | {r['rf'][0]:.1f} | {r['rf'][1]:.1f} | {r['rf'][2]:.1f} |",
        "",
        "## 해석",
        "- 이제 copy(다음=현재)는 실제 단계 이동 전체를 오차로 가짐(초반 수백 m).",
        "  즉 예측이 의미를 가지며, baseline/RF가 copy보다 낮아야 학습이 유효함.",
        "- 초반 단계일수록 이동이 커서 예측 난도가 높고 오차도 큼(정상).",
        "- 데이터가 적어 표본 적은 후반 단계는 변동이 큼.",
    ]
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n리포트: {REPORT_PATH}")


if __name__ == "__main__":
    main()
