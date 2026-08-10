"""
Day 16 작업: 확률분포(가우시안) 기반 다음 원 예측 + 신뢰구간.

베이스라인은 "점 하나"만 예측했다. 여기서는 다음 원 중심의 이동량
(delta = poison - safety)을 단계별 2D 가우시안 분포로 근사한다.
  - 평균(mean)  : 예측 이동량 → 중심 예측에 사용
  - 표준편차(std): 예측의 불확실성 → 신뢰구간(반경)으로 변환

Day 15의 발견(절대 좌표가 아니라 이동량으로 봐야 공정)을 확률 모델로 구현한 것.
현재 위치를 기본값으로 깔고, 이동량의 분포를 학습한다.

신뢰구간 반경(2D 가우시안):
  중심 오차가 반경 R 안에 들어올 확률이 conf가 되는 R은
  R = sigma * sqrt(chi2_inv(conf, dof=2)).  (예: 95% → sqrt(5.991) ≈ 2.448 * sigma)
  sigma는 x/y 분산의 평균으로 근사한다.

완료 기준(Day 16): 특정 입력에 대해 중심 + 표준편차(신뢰구간) 출력.

실행: python ml/gaussian_model.py
"""
import math
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE, "..", "data", "processed", "zones_dataset.csv")

# 신뢰수준별 2D 가우시안 스케일 계수 (chi-square, dof=2)
#   68% → 1.515, 90% → 2.146, 95% → 2.448
CHI2_SCALE = {0.68: 1.515, 0.90: 2.146, 0.95: 2.448}


class GaussianModel:
    def __init__(self):
        # phase -> {mean_dx, mean_dy, std_x, std_y, mean_shrink, std_shrink, n}
        self.phase_dist: dict[int, dict] = {}
        self.known_phases: list[int] = []

    def fit(self, df: pd.DataFrame) -> "GaussianModel":
        """이동량(delta)과 축소 비율의 단계별 분포(평균·표준편차)를 추정."""
        d = df.copy()
        d["dx"] = d["poison_x"] - d["safety_x"]
        d["dy"] = d["poison_y"] - d["safety_y"]
        d = d[d["safety_radius"] > 0]
        d["shrink"] = d["poison_radius"] / d["safety_radius"]

        for phase, g in d.groupby("phase"):
            self.phase_dist[int(phase)] = {
                "mean_dx": float(g["dx"].mean()),
                "mean_dy": float(g["dy"].mean()),
                "std_x": float(g["dx"].std(ddof=0)),
                "std_y": float(g["dy"].std(ddof=0)),
                "mean_shrink": float(g["shrink"].mean()),
                "std_shrink": float(g["shrink"].std(ddof=0)),
                "n": int(len(g)),
            }
        self.known_phases = sorted(self.phase_dist.keys())
        return self

    def _resolve_phase(self, phase: int) -> int:
        if phase in self.phase_dist:
            return phase
        lower = [p for p in self.known_phases if p <= phase]
        return max(lower) if lower else self.known_phases[0]

    def predict(self, current_x: float, current_y: float,
                current_radius: float, phase: int, conf: float = 0.95) -> dict:
        """
        현재 원 + 단계 → 다음 원 중심(가우시안 평균)과 신뢰구간 반경 반환.
        반환: {x, y, radius, std_x, std_y, confidence, confidence_radius, used_phase}
        """
        used = self._resolve_phase(phase)
        s = self.phase_dist[used]

        # 중심 = 현재 위치 + 평균 이동량 (delta 프레이밍)
        x = current_x + s["mean_dx"]
        y = current_y + s["mean_dy"]
        radius = current_radius * s["mean_shrink"]

        # 신뢰구간 반경: x/y 분산 평균을 sigma로 보고 chi2 스케일 적용
        sigma = math.sqrt((s["std_x"] ** 2 + s["std_y"] ** 2) / 2)
        scale = CHI2_SCALE.get(conf, 2.448)
        confidence_radius = scale * sigma

        return {
            "x": x, "y": y, "radius": radius,
            "std_x": s["std_x"], "std_y": s["std_y"],
            "confidence": conf,
            "confidence_radius": confidence_radius,
            "used_phase": used,
        }


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    model = GaussianModel().fit(df)
    print(f"학습된 단계: {model.known_phases}\n")

    samples = [
        (400000, 400000, 200000, 2),
        (350000, 550000, 100000, 5),
        (320000, 575000, 40000, 6),
    ]
    for cx, cy, cr, ph in samples:
        out = model.predict(cx, cy, cr, ph, conf=0.95)
        print(f"phase={ph} 현재(({cx},{cy}) r={cr})")
        print(f"  → 중심({out['x']:.0f},{out['y']:.0f}) r={out['radius']:.0f}")
        print(f"    std=(x {out['std_x']:.0f}, y {out['std_y']:.0f}), "
              f"95% 신뢰반경 ±{out['confidence_radius']:.0f} [phase {out['used_phase']}]")

    print("\n✅ Day 16 완료 기준 통과: 중심 + 표준편차(신뢰구간) 출력 확인")
