"""
알고리즘 & 특징 비교 실험: RF vs LightGBM vs XGBoost, 기본 특징 vs +엔지니어링.

목적:
- 세 트리 모델을 동일 조건(매치 단위 split, 이동량 회귀)에서 비교.
- 엔지니어링 특징(위치·중앙거리·모멘텀 등) 추가 효과 측정.
평가: 예측한 다음 원 중심과 실제 중심의 거리(m). copy(다음=현재)를 기준선으로 병기.

실행: python ml/compare_algos.py
"""
import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from ml.dataset_pairs import build_transition_pairs
from ml.features import add_engineered, ENGINEERED

BASE = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE, "..", "data", "processed", "zones_dataset.csv")
CM_PER_M = 100.0
SEED = 42

BASE_NUM = ["safety_x", "safety_y", "safety_radius", "phase"]
CAT = ["map"]


def split(pairs):
    rng = np.random.default_rng(SEED)
    ids = pairs["match_id"].unique()
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * 0.2))
    test = set(ids[:n_test])
    return pairs[~pairs["match_id"].isin(test)], pairs[pairs["match_id"].isin(test)]


def make(model, num_cols):
    pre = ColumnTransformer([
        ("map", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", "passthrough", num_cols),
    ])
    # XGB/LGBM 다중출력은 MultiOutputRegressor로 감싼다
    return Pipeline([("pre", pre), ("m", MultiOutputRegressor(model))])


def err_m(pred, test):
    ax = test["safety_x"].values + test["dx"].values
    ay = test["safety_y"].values + test["dy"].values
    px = test["safety_x"].values + pred[:, 0]
    py = test["safety_y"].values + pred[:, 1]
    return np.hypot(px - ax, py - ay) / CM_PER_M


def main():
    pairs = add_engineered(build_transition_pairs(pd.read_csv(CSV_PATH)))
    train, test = split(pairs)

    # copy 기준선
    copy_e = np.hypot(test["dx"], test["dy"]).values / CM_PER_M

    models = {
        "RandomForest": lambda: RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
        "LightGBM": lambda: LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                          random_state=SEED, verbose=-1),
        "XGBoost": lambda: XGBRegressor(n_estimators=400, learning_rate=0.05,
                                        random_state=SEED, verbosity=0),
    }
    feature_sets = {"기본": BASE_NUM, "기본+엔지니어링": BASE_NUM + ENGINEERED}

    print(f"train {len(train)} / test {len(test)}  (copy 기준: 평균 {copy_e.mean():.1f}m)\n")
    print(f"{'model':<14}{'features':<16}{'mean(m)':>9}{'median(m)':>11}")
    print("-" * 50)
    for mname, mk in models.items():
        for fname, cols in feature_sets.items():
            pipe = make(mk(), cols)
            pipe.fit(train[CAT + cols], train[["dx", "dy"]].values)
            e = err_m(pipe.predict(test[CAT + cols]), test)
            print(f"{mname:<14}{fname:<16}{e.mean():>9.1f}{np.median(e):>11.1f}")
    print(f"\n{'copy(다음=현재)':<30}{copy_e.mean():>9.1f}{np.median(copy_e):>11.1f}")
    print("\n※ 중심은 무작위성이 커서 copy를 크게 못 이김. 엔지니어링 특징의 효과를 열별로 비교.")


if __name__ == "__main__":
    main()
