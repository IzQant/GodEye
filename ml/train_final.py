"""
Day 19 작업: 최종 모델 학습 + joblib 직렬화.

Day 18에서 선정한 RF(delta)를 전체 데이터로 학습하고,
반경 축소비율/신뢰구간 통계와 함께 ZonePredictor로 묶어
ml/models/predictor.joblib 로 저장한다.

이 파일 하나를 앱이 로드하면 /api/predict 서빙이 가능해진다(Day 20).

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

BASE = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE, "..", "data", "processed", "zones_dataset.csv")
OUT_PATH = os.path.join(BASE, "models", "predictor.joblib")
SEED = 42


def main():
    df = pd.read_csv(CSV_PATH)
    df["dx"] = df["poison_x"] - df["safety_x"]
    df["dy"] = df["poison_y"] - df["safety_y"]
    df = df[df["safety_radius"] > 0].copy()
    df["shrink"] = df["poison_radius"] / df["safety_radius"]

    # 1) 중심 이동량(dx, dy) 회귀 — 전체 데이터로 학습 (배포용 최종본)
    pre = ColumnTransformer([
        ("map", OneHotEncoder(handle_unknown="ignore"), ["map"]),
        ("num", "passthrough", ["safety_x", "safety_y", "safety_radius", "phase"]),
    ])
    rf = Pipeline([("pre", pre),
                   ("rf", RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1))])
    rf.fit(df[FEATURE_COLS], df[["dx", "dy"]].values)

    # 2) 단계별 반경 축소비율 평균 (반경 예측용)
    phase_shrink = df.groupby("phase")["shrink"].mean().to_dict()
    phase_shrink = {int(k): float(v) for k, v in phase_shrink.items()}

    # 3) 단계별 이동량 표준편차 (신뢰구간용): sqrt((var_x+var_y)/2)
    phase_sigma = {}
    for phase, g in df.groupby("phase"):
        sx = g["dx"].std(ddof=0)
        sy = g["dy"].std(ddof=0)
        phase_sigma[int(phase)] = float(math.sqrt((sx ** 2 + sy ** 2) / 2))

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
