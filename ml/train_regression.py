"""
Day 15 작업: scikit-learn 회귀 모델로 다음 원 예측.

베이스라인은 "단계별 평균"만 썼다. 회귀 모델은 여기에
현재 원의 실제 좌표·반경·맵 정보까지 특징(feature)으로 넣어,
입력에 따라 다른 예측을 하도록 학습한다.

특징(X):
- safety_x, safety_y : 현재 원 중심
- safety_radius      : 현재 원 반경
- phase              : 단계
- map                : 맵 이름 (원-핫 인코딩)
목표(y):
- poison_x, poison_y, poison_radius : 다음 원 중심·반경 (동시 예측 = 다중출력)

모델: RandomForestRegressor (다중출력 native 지원, 스케일링 불필요, 튜닝 부담 적음)

완료 기준(Day 15): train/test 분리 후 회귀 모델 학습 및 예측 성공.
(모델 비교/직렬화/서빙은 Day 18~20에서)

실행: python ml/train_regression.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE, "..", "data", "processed", "zones_dataset.csv")

CM_PER_M = 100.0
TEST_RATIO = 0.2
SEED = 42

FEATURES_NUM = ["safety_x", "safety_y", "safety_radius", "phase"]
FEATURES_CAT = ["map"]
TARGETS = ["poison_x", "poison_y", "poison_radius"]


def split_by_match(df: pd.DataFrame):
    """베이스라인 평가와 동일한 방식(매치 단위, seed 42)으로 분리해 비교 가능하게 함."""
    rng = np.random.default_rng(SEED)
    ids = df["match_id"].unique()
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * TEST_RATIO))
    test_ids = set(ids[:n_test])
    return df[~df["match_id"].isin(test_ids)], df[df["match_id"].isin(test_ids)]


def build_model() -> Pipeline:
    """맵은 원-핫, 나머지 수치는 그대로 통과시켜 RandomForest에 넣는 파이프라인."""
    pre = ColumnTransformer(
        transformers=[
            ("map", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
            ("num", "passthrough", FEATURES_NUM),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=SEED,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("rf", model)])


def center_error_m(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """예측/실제의 (x, y) 중심 유클리드 거리(m). pred/actual 열 순서는 TARGETS."""
    dx = pred[:, 0] - actual[:, 0]
    dy = pred[:, 1] - actual[:, 1]
    return np.sqrt(dx ** 2 + dy ** 2) / CM_PER_M


def main():
    df = pd.read_csv(CSV_PATH)
    train_df, test_df = split_by_match(df)

    X_cols = FEATURES_CAT + FEATURES_NUM
    X_train, y_train = train_df[X_cols], train_df[TARGETS].values
    X_test, y_test = test_df[X_cols], test_df[TARGETS].values

    model = build_model()
    model.fit(X_train, y_train)          # 학습
    pred = model.predict(X_test)         # 예측

    err = center_error_m(pred, y_test)
    print(f"train 행 {len(train_df)} / test 행 {len(test_df)} "
          f"(매치 {df['match_id'].nunique()}개)")
    print(f"회귀 모델 중심 오차(m): 평균 {err.mean():.1f} / 중앙값 {np.median(err):.1f} "
          f"/ p90 {np.percentile(err, 90):.1f}")

    print("\n예측 샘플 (앞 5개, 단위 cm):")
    for i in range(min(5, len(pred))):
        print(f"  실제({y_test[i,0]:.0f},{y_test[i,1]:.0f} r={y_test[i,2]:.0f}) "
              f"→ 예측({pred[i,0]:.0f},{pred[i,1]:.0f} r={pred[i,2]:.0f})")

    print("\n✅ Day 15 완료 기준 통과: 회귀 모델 학습 및 예측 성공")


if __name__ == "__main__":
    main()
