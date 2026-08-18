"""
히트맵(분포) 예측 모델.

점(중심 좌표) 예측은 자기장 무작위성 때문에 한계가 있다. 대신
'다음 원 중심이 어디에 있을 확률'을 2D 분포로 예측한다.

방법(소규모 데이터에 적합):
- 각 전환에서 이동 오프셋을 현재 반경으로 정규화: ox=dx/r, oy=dy/r.
  (현재 원 크기와 무관하게 '상대 위치'로 표현 → 단계별로 모아 쓰기 좋음.)
- 단계별로 이 정규화 오프셋들에 가우시안 KDE를 적합.
- 예측 시: 해당 단계 KDE를 현재 반경으로 되돌리고 현재 중심에 얹어
  맵 좌표 위 확률 히트맵을 만든다.

평가:
- Coverage/calibration: 예측 분포의 상위 q% 영역(HPD)에 실제 중심이 들어오는 비율이
  q와 비슷해야 잘 보정된 것.
- 로그가능도(LL): 실제 지점에서의 예측 밀도. 균등분포(현재 원 내부 균등) 기준선과 비교.
"""
import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.stats import gaussian_kde


class HeatmapModel:
    def __init__(self, min_samples=8):
        self.min_samples = min_samples
        self.kde_by_phase = {}
        self.global_kde = None

    def fit(self, pairs):
        d = pairs[pairs["safety_radius"] > 0].copy()
        ox = (d["dx"] / d["safety_radius"]).values
        oy = (d["dy"] / d["safety_radius"]).values
        self.global_kde = gaussian_kde(np.vstack([ox, oy]))
        for phase, g in d.groupby("phase"):
            a = (g["dx"] / g["safety_radius"]).values
            b = (g["dy"] / g["safety_radius"]).values
            if len(a) >= self.min_samples and np.std(a) > 1e-6 and np.std(b) > 1e-6:
                self.kde_by_phase[int(phase)] = gaussian_kde(np.vstack([a, b]))
        return self

    def _kde(self, phase):
        return self.kde_by_phase.get(int(phase), self.global_kde)

    def density_norm(self, phase, ox, oy):
        """정규화 오프셋 공간에서의 밀도(배열 지원)."""
        return self._kde(phase)(np.vstack([np.atleast_1d(ox), np.atleast_1d(oy)]))

    def predict_grid(self, cur_x, cur_y, cur_r, phase, res=120):
        """
        현재 원 주변(±1.2*r) 격자에 대해 맵 좌표 히트맵을 계산.
        반환: (X, Y, density) — X,Y는 맵 좌표 격자, density는 정규화 밀도.
        """
        span = 1.2
        gx = np.linspace(cur_x - span * cur_r, cur_x + span * cur_r, res)
        gy = np.linspace(cur_y - span * cur_r, cur_y + span * cur_r, res)
        X, Y = np.meshgrid(gx, gy)
        ox = (X - cur_x) / cur_r
        oy = (Y - cur_y) / cur_r
        dens = self._kde(phase)(np.vstack([ox.ravel(), oy.ravel()])).reshape(X.shape)
        return X, Y, dens


# ---------- 평가 ----------
def _coverage_and_ll(model, test):
    """test 전환들에 대해 HPD coverage(50/80/90%)와 KDE/균등 LL 비교."""
    qs = [0.5, 0.8, 0.9]
    hits = {q: 0 for q in qs}
    ll_kde, ll_unif = [], []
    n = 0
    # 정규화 오프셋 격자(밀도 적분/HPD 계산용)
    gx = np.linspace(-1.2, 1.2, 121)
    GX, GY = np.meshgrid(gx, gx)
    cell = (gx[1] - gx[0]) ** 2
    unif_density = 1.0 / np.pi  # 단위원 내부 균등분포 밀도

    for _, r in test.iterrows():
        if r["safety_radius"] <= 0:
            continue
        kde = model._kde(int(r["phase"]))
        grid_d = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
        grid_d = grid_d / (grid_d.sum() * cell)  # 정규화
        ax = r["dx"] / r["safety_radius"]
        ay = r["dy"] / r["safety_radius"]
        d_actual = float(kde(np.array([[ax], [ay]]))[0])
        d_actual_n = d_actual / (grid_d.sum() * cell) if False else d_actual
        # HPD: 실제 지점 밀도보다 높은 셀들의 질량 = 1 - 신뢰수준 → 실제가 상위 q영역에 있으면 hit
        mass_above = grid_d[grid_d >= kde(np.array([[ax], [ay]]))[0]].sum() * cell
        for q in qs:
            if mass_above <= q:
                hits[q] += 1
        ll_kde.append(np.log(max(d_actual, 1e-9)))
        ll_unif.append(np.log(unif_density))
        n += 1

    cov = {q: hits[q] / n for q in qs}
    return cov, float(np.mean(ll_kde)), float(np.mean(ll_unif)), n


def main():
    import pandas as pd
    from ml.dataset_pairs import build_transition_pairs

    BASE = os.path.dirname(__file__)
    CSV = os.path.join(BASE, "..", "data", "processed", "zones_dataset.csv")
    pairs = build_transition_pairs(pd.read_csv(CSV))

    rng = np.random.default_rng(42)
    ids = pairs["match_id"].unique(); rng.shuffle(ids)
    n_test = max(1, int(len(ids) * 0.2))
    test_ids = set(ids[:n_test])
    train = pairs[~pairs["match_id"].isin(test_ids)]
    test = pairs[pairs["match_id"].isin(test_ids)]

    model = HeatmapModel().fit(train)
    cov, ll_kde, ll_unif, n = _coverage_and_ll(model, test)

    print(f"train {len(train)} / test {n}")
    print("=== Coverage (예측 상위 q% 영역에 실제 중심이 들어온 비율) ===")
    for q, c in cov.items():
        print(f"  {int(q*100)}% 영역 → 실제 포함 {c*100:.1f}%  (이상적: {int(q*100)}%)")
    print(f"\n로그가능도(높을수록 좋음): KDE {ll_kde:.3f} vs 균등 {ll_unif:.3f} "
          f"→ 개선 {ll_kde-ll_unif:+.3f}")
    print("해석: KDE LL이 균등보다 높으면, 히트맵이 '단순 무작위'보다 정보량이 있다는 뜻.")

    # 샘플 히트맵 렌더
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    row = test[test["phase"] == 3].iloc[0] if (test["phase"] == 3).any() else test.iloc[0]
    X, Y, D = model.predict_grid(row["safety_x"], row["safety_y"], row["safety_radius"], int(row["phase"]))
    plt.figure(figsize=(6, 6))
    plt.imshow(D, origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()], cmap="hot")
    th = np.linspace(0, 2*np.pi, 100)
    plt.plot(row["safety_x"] + row["safety_radius"]*np.cos(th),
             row["safety_y"] + row["safety_radius"]*np.sin(th), "c-", lw=1, label="current zone")
    plt.plot(row["safety_x"] + row["dx"], row["safety_y"] + row["dy"], "g+", ms=14, mew=3, label="actual next")
    plt.legend(); plt.title(f"Next-zone probability heatmap (phase {int(row['phase'])})")
    plt.gca().set_aspect("equal")
    out = os.path.join(BASE, "figures", "heatmap_sample.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()
    print(f"\n샘플 히트맵 저장: {out}")


if __name__ == "__main__":
    main()
