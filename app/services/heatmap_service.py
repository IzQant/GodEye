"""
히트맵 모델 로더 (임시 연결용).

zones_dataset.csv에서 전환쌍을 만들어 HeatmapModel을 1회 학습하고 캐시한다.
(점 예측기 predictor.joblib와 별개. 배포 정식 반영 전 작동 테스트 목적.)
"""
import os

_MODEL = None
_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "zones_dataset.csv")


def get_heatmap_model():
    global _MODEL
    if _MODEL is None:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
        import pandas as pd
        from ml.dataset_pairs import build_transition_pairs
        from ml.heatmap_model import HeatmapModel
        print("[heatmap_service] 히트맵 모델 학습(1회)")
        pairs = build_transition_pairs(pd.read_csv(_CSV))
        _MODEL = HeatmapModel().fit(pairs)
    return _MODEL
