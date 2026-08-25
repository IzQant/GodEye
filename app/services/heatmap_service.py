"""
히트맵 모델 로더 (임시 연결용).

zones_dataset.csv에서 전환쌍을 만들어 HeatmapModel을 1회 학습하고 캐시한다.
(점 예측기 predictor.joblib와 별개. 배포 정식 반영 전 작동 테스트 목적.)
"""
import os

_MODEL = None
_BASE = os.path.join(os.path.dirname(__file__), "..", "..")
_JOBLIB = os.path.join(_BASE, "ml", "models", "heatmap.joblib")
_CSV = os.path.join(_BASE, "data", "processed", "zones_dataset.csv")


def get_heatmap_model():
    """빌드 시 만든 heatmap.joblib이 있으면 가볍게 로드(메모리 스파이크 회피),
    없으면 요청 시 1회 학습(로컬 개발 폴백)."""
    global _MODEL
    if _MODEL is None:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
        if os.path.exists(_JOBLIB):
            import joblib
            print("[heatmap_service] 히트맵 모델 로드(joblib)", flush=True)
            _MODEL = joblib.load(_JOBLIB)
        else:
            import pandas as pd
            from ml.dataset_pairs import build_transition_pairs
            from ml.heatmap_model import HeatmapModel
            print("[heatmap_service] 히트맵 모델 학습(1회, joblib 없음)", flush=True)
            pairs = build_transition_pairs(pd.read_csv(_CSV))
            _MODEL = HeatmapModel().fit(pairs)
    return _MODEL
