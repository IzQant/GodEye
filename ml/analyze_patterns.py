"""
Day 11 작업: Phase 2 EDA — 단계별 원 축소 패턴 분석.

각 행은 "현재 안전지대(safety) → 다음 자기장(poison)" 한 쌍이다.
- 이동 벡터: (dx, dy) = (poison - safety)  → 다음 원 중심이 현재 원에서 어디로 이동하는지
- 축소 비율: poison_radius / safety_radius  → 다음 원이 현재 대비 얼마나 작아지는지

이 통계(phase_stats)는 Day 12 베이스라인 모델이 그대로 사용한다:
    next_x = current_x + dx_mean
    next_y = current_y + dy_mean
    next_radius = current_radius * shrink_ratio_mean

산출물:
- ml/figures/movement_vectors.png : 단계별 평균 이동 벡터
- ml/figures/shrink_ratio.png      : 단계별 축소 비율 분포
- ml/phase_stats.json              : 단계별 통계 (Day 12 입력)

실행: python ml/analyze_patterns.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")  # 화면 없는 환경에서도 PNG로 저장 가능하게
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE, "..", "data", "processed", "zones_dataset.csv")
FIG_DIR = os.path.join(BASE, "figures")
STATS_PATH = os.path.join(BASE, "phase_stats.json")


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """이동 벡터(dx, dy), 이동 거리, 축소 비율을 행마다 계산."""
    df = df.copy()
    df["dx"] = df["poison_x"] - df["safety_x"]
    df["dy"] = df["poison_y"] - df["safety_y"]
    df["move_dist"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    # 반경이 0인 이상 행은 축소 비율 계산에서 제외
    df = df[df["safety_radius"] > 0]
    df["shrink_ratio"] = df["poison_radius"] / df["safety_radius"]
    return df


def build_phase_stats(df: pd.DataFrame) -> dict:
    """
    단계별 평균 통계를 dict로 만든다.
    key는 phase(문자열), value는 dx_mean/dy_mean/shrink_ratio_mean + 표준편차/표본수.
    """
    stats = {}
    for phase, g in df.groupby("phase"):
        stats[str(int(phase))] = {
            "dx_mean": float(g["dx"].mean()),
            "dy_mean": float(g["dy"].mean()),
            "shrink_ratio_mean": float(g["shrink_ratio"].mean()),
            "dx_std": float(g["dx"].std(ddof=0)),
            "dy_std": float(g["dy"].std(ddof=0)),
            "move_dist_mean": float(g["move_dist"].mean()),
            "n": int(len(g)),
        }
    return stats


def plot_movement_vectors(stats: dict, path: str):
    """단계별 평균 이동 벡터를 원점 기준 화살표(quiver)로 그린다."""
    phases = sorted(stats.keys(), key=int)
    dxs = [stats[p]["dx_mean"] for p in phases]
    dys = [stats[p]["dy_mean"] for p in phases]

    plt.figure(figsize=(6, 6))
    # 각 단계의 평균 이동을 원점에서 뻗는 화살표로 표시
    plt.quiver(
        [0] * len(phases), [0] * len(phases), dxs, dys,
        angles="xy", scale_units="xy", scale=1,
        color=plt.cm.viridis(np.linspace(0, 1, len(phases))),
    )
    for p, dx, dy in zip(phases, dxs, dys):
        plt.annotate(f"P{p}", (dx, dy))
    plt.axhline(0, color="gray", lw=0.5)
    plt.axvline(0, color="gray", lw=0.5)
    plt.title("Mean movement vector per phase (current -> next circle)")
    plt.xlabel("dx"); plt.ylabel("dy")
    # quiver는 축 범위를 자동으로 넓혀주지 않으므로 데이터 최대치 기준으로 직접 지정
    lim = max(max(abs(v) for v in dxs), max(abs(v) for v in dys), 1.0) * 1.2
    plt.xlim(-lim, lim); plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal", "box")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_shrink_ratio(df: pd.DataFrame, path: str):
    """단계별 축소 비율 분포를 박스플롯으로 그린다."""
    phases = sorted(df["phase"].unique())
    data = [df[df["phase"] == p]["shrink_ratio"].values for p in phases]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=[f"P{int(p)}" for p in phases])
    plt.axhline(1.0, color="red", ls="--", lw=0.8, label="ratio 1.0 (no shrink)")
    plt.title("Shrink ratio distribution per phase (poison_radius / safety_radius)")
    plt.ylabel("shrink ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    df = pd.read_csv(CSV_PATH)
    df = compute_features(df)
    os.makedirs(FIG_DIR, exist_ok=True)

    stats = build_phase_stats(df)

    # 통계 표 출력
    print("=== 단계별 통계 (phase_stats) ===")
    print(f"{'phase':>5} {'n':>4} {'dx_mean':>12} {'dy_mean':>12} {'shrink_mean':>12} {'move_mean':>12}")
    for p in sorted(stats.keys(), key=int):
        s = stats[p]
        print(f"{p:>5} {s['n']:>4} {s['dx_mean']:>12.1f} {s['dy_mean']:>12.1f} "
              f"{s['shrink_ratio_mean']:>12.3f} {s['move_dist_mean']:>12.1f}")

    plot_movement_vectors(stats, os.path.join(FIG_DIR, "movement_vectors.png"))
    plot_shrink_ratio(df, os.path.join(FIG_DIR, "shrink_ratio.png"))

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n그래프 저장: {FIG_DIR}/movement_vectors.png, shrink_ratio.png")
    print(f"통계 저장: {STATS_PATH}")
    print("✅ Day 11 완료 기준: 이동벡터·축소비율 분포 그래프 + phase_stats 확보")


if __name__ == "__main__":
    main()
