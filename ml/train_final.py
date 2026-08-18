"""
최종 모델 학습 + joblib 직렬화.

[재구성] 학습 목표를 '단계 전환(phase N → phase N+1)'으로 변경.
- 기존: 같은 스냅샷 safety→poison (초반 단계 거의 동일 → 예측 무의미).
- 변경: phase N의 실제 원 → phase N+1의 실제 원 (단계 사이 실제 이동/축소를 예측).
ZonePredictor 구조·인터페이스는 그대로, 학습 데이터만 전환쌍으로 바꾼다.

실행: python ml/train_final.py
"""
import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.services.model_service import ZonePredictor, FEATURE_COLS
from ml.dataset_pairs import build_transition_pairs

BASE = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE, "..", "data", "processed", "zones_dataset.csv")
OUT_PATH = os.path.join(BASE, "models", "predictor.joblib")
SEED = 42


def main():
    df = pd.read_csv(CSV_PATH)
    # 단계 전환쌍(phase N → N+1)으로 학습 데이터 구성
    pairs = build_transition_pairs(df)

    # 1) 중심 이동량(dx, dy) 회귀 — 전환쌍 전체로 학습
    pre = ColumnTransformer([
        ("map", OneHotEncoder(handle_unknown="ignore"), ["map"]),
        ("num", "passthrough", ["safety_x", "safety_y", "safety_radius", "phase"]),
    ])
    rf = Pipeline([("pre", pre),
                   ("rf", RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1))])
    rf.fit(pairs[FEATURE_COLS], pairs[["dx", "dy"]].values)

    # 2) 단계별 반경 축소비율 평균 (다음 단계 반경 예측용)
    phase_shrink = pairs.groupby("phase")["shrink"].mean().to_dict()
    phase_shrink = {int(k): float(v) for k, v in phase_shrink.items()}

    # 3) 단계별 이동량 표준편차 (신뢰구간용)
    phase_sigma = {}
    for phase, g in pairs.groupby("phase"):
        sx = g["dx"].std(ddof=0)
        sy = g["dy"].std(ddof=0)
        phase_sigma[int(phase)] = float(math.sqrt((sx ** 2 + sy ** 2) / 2))

    df = pairs  # 아래 출력용
    predictor = ZonePredictor(rf, phase_shrink, phase_sigma)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    joblib.dump(predictor, OUT_PATH)
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"저장 완료: {OUT_PATH} ({size_kb:.0f} KB)")
    print(f"학습 행 {len(df)}, 단계 {sorted(phase_shrink.keys())}")

    # 저장본 즉시 로드해 1건 예측(직렬화가 온전한지 확인)
    loaded = joblib.load(OUT_PATH)
    out = loaded.predict(350000, 550000, 100000, phase=5, map_name="Erangel")
    print(f"복원 후 예측 샘플: 중심({out['x']:.0f},{out['y']:.0f}) "
          f"r={out['radius']:.0f} ±{out['confidence_radius']:.0f}")
    print("✅ Day 19 완료 기준(모델 직렬화) 통과")


if __name__ == "__main__":
    main()
