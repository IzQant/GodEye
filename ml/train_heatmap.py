"""
히트맵(KDE) 모델을 빌드 시점에 미리 학습·직렬화한다.

기존엔 첫 /api/heatmap 요청 때 pandas 로드 + 전환쌍 생성 + KDE 학습을 수행해
메모리 스파이크로 작은 컨테이너에서 OOM이 났다. 이를 빌드 때 1회 만들어
ml/models/heatmap.joblib로 저장하고, 서빙 땐 가볍게 로드만 하도록 분리한다.

실행: python ml/train_heatmap.py   (Dockerfile 빌드 단계에서 호출)
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import joblib
import pandas as pd

from ml.dataset_pairs import build_transition_pairs
from ml.heatmap_model import HeatmapModel

BASE = os.path.dirname(os.path.dirname(__file__))
CSV = os.path.join(BASE, "data", "processed", "zones_dataset.csv")
OUT = os.path.join(BASE, "ml", "models", "heatmap.joblib")


def main():
    pairs = build_transition_pairs(pd.read_csv(CSV))
    model = HeatmapModel().fit(pairs)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    joblib.dump(model, OUT)
    print(f"[train_heatmap] 저장: {OUT} (phase KDE {len(model.kde_by_phase)}개)")


if __name__ == "__main__":
    main()
