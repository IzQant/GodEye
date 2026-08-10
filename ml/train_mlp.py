"""
Day 17 작업(선택): PyTorch 기반 MLP 회귀 실험.

────────────────────────────────────────────────────────────────────────
[이 프로젝트에서 MLP가 하는 일 — 작동 과정 해설]

우리 문제: "현재 자기장 원(safety) + 단계 + 맵" 을 보고
           "다음 원이 현재에서 얼마나 이동하는지(dx, dy)" 를 맞히는 것.
           (Day 15의 발견을 따라 절대 좌표가 아니라 '이동량'을 목표로 한다.)

MLP(다층 퍼셉트론)는 이 입력→출력 관계를 여러 층의 곱셈+비선형으로 근사한다.
한 번의 학습 반복(epoch)에서 벌어지는 일:

  1) 순전파(forward)
     입력 벡터 x = [맵 원-핫..., safety_x, safety_y, safety_radius, phase]
     → Linear(입력→64): x에 가중치 행렬을 곱하고 편향을 더함
     → ReLU: 음수를 0으로 잘라 비선형성 부여(직선만으론 못 맞추는 패턴 학습)
     → Linear(64→32) → ReLU → Linear(32→2)
     → 출력 ŷ = (예측 dx, 예측 dy)

  2) 손실(loss) 계산
     실제 이동량 y=(dx, dy)와 예측 ŷ의 차이를 MSE(평균제곱오차)로 잰다.
     "예측 원 중심이 실제에서 얼마나 벗어났나"를 하나의 숫자로 요약한 것.

  3) 역전파(backward)
     이 손실을 각 가중치로 미분(기울기)해서, "가중치를 어느 방향으로
     조금 바꾸면 손실이 줄어드는지"를 계산한다. (자동미분 autograd)

  4) 최적화(step)
     Adam 옵티마이저가 그 기울기만큼 가중치를 조금씩 갱신한다.
     이 1~4를 수백 번(epoch) 반복하면 예측 이동량이 실제에 가까워진다.

주의: 신경망은 입력 크기 차이에 민감하다. 좌표(수십만) vs phase(한 자리)를
      그대로 넣으면 학습이 불안정하므로, 입력·출력을 표준화(평균0/표준편차1)한 뒤
      학습하고, 예측할 때 원래 단위로 되돌린다.

우리 데이터는 419행으로 작아서 MLP가 트리/평균 기반보다 유리하기 어렵다.
이 실험의 목적은 "성능 비교의 한 축"을 확보하는 것 (Day 18 비교 리포트용).
────────────────────────────────────────────────────────────────────────

완료 기준(Day 17): MLP 학습 후 기존 모델과 오차 비교표 확보.

실행: python ml/train_mlp.py
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder

BASE = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE, "..", "data", "processed", "zones_dataset.csv")

CM_PER_M = 100.0
TEST_RATIO = 0.2
SEED = 42
EPOCHS = 300
LR = 0.01

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_split():
    """CSV 로드 → 이동량(delta) 목표 계산 → 매치 단위 train/test 분리."""
    df = pd.read_csv(CSV_PATH)
    df["dx"] = df["poison_x"] - df["safety_x"]
    df["dy"] = df["poison_y"] - df["safety_y"]

    rng = np.random.default_rng(SEED)
    ids = df["match_id"].unique()
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * TEST_RATIO))
    test_ids = set(ids[:n_test])
    return df[~df["match_id"].isin(test_ids)], df[df["match_id"].isin(test_ids)]


def build_features(train_df, test_df):
    """맵 원-핫 + 수치 특징을 합치고, 표준화(평균0/표준편차1)해서 텐서로 변환."""
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    map_tr = enc.fit_transform(train_df[["map"]])
    map_te = enc.transform(test_df[["map"]])

    num_cols = ["safety_x", "safety_y", "safety_radius", "phase"]
    Xtr = np.hstack([map_tr, train_df[num_cols].values]).astype(np.float32)
    Xte = np.hstack([map_te, test_df[num_cols].values]).astype(np.float32)
    ytr = train_df[["dx", "dy"]].values.astype(np.float32)
    yte = test_df[["dx", "dy"]].values.astype(np.float32)

    # 표준화 통계는 train에서만 계산해 test에 적용(누수 방지)
    x_mean, x_std = Xtr.mean(0), Xtr.std(0) + 1e-8
    y_mean, y_std = ytr.mean(0), ytr.std(0) + 1e-8
    Xtr = (Xtr - x_mean) / x_std
    Xte = (Xte - x_mean) / x_std
    ytr_s = (ytr - y_mean) / y_std

    return (torch.tensor(Xtr), torch.tensor(ytr_s), torch.tensor(Xte),
            yte, y_mean, y_std)


class MLP(nn.Module):
    """입력 → 64 → 32 → 2(dx, dy). ReLU로 비선형성 부여."""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


def main():
    train_df, test_df = load_split()
    Xtr, ytr_s, Xte, yte, y_mean, y_std = build_features(train_df, test_df)

    model = MLP(Xtr.shape[1])
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # ---- 학습 루프: 순전파 → 손실 → 역전파 → 갱신 ----
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()              # 이전 기울기 초기화
        pred_s = model(Xtr)          # 순전파
        loss = loss_fn(pred_s, ytr_s)  # 손실(표준화 공간)
        loss.backward()              # 역전파(기울기 계산)
        opt.step()                   # 가중치 갱신
        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch+1:>3} / {EPOCHS}  loss={loss.item():.4f}")

    # ---- 평가: 예측을 원래 단위(cm)로 되돌려 중심 오차(m) 계산 ----
    model.eval()
    with torch.no_grad():
        pred_delta = model(Xte).numpy() * y_std + y_mean  # 표준화 해제

    # 예측 중심 = 현재 위치 + 예측 이동량
    pred_x = test_df["safety_x"].values + pred_delta[:, 0]
    pred_y = test_df["safety_y"].values + pred_delta[:, 1]
    err = np.sqrt((pred_x - test_df["poison_x"].values) ** 2 +
                  (pred_y - test_df["poison_y"].values) ** 2) / CM_PER_M

    print(f"\ntrain 행 {len(train_df)} / test 행 {len(test_df)}")
    print(f"MLP 중심 오차(m): 평균 {err.mean():.1f} / 중앙값 {np.median(err):.1f} "
          f"/ p90 {np.percentile(err, 90):.1f}")
    print("\n✅ Day 17 완료 기준 통과: MLP 학습·예측 성공 (Day 18 비교표에 반영)")


if __name__ == "__main__":
    main()
