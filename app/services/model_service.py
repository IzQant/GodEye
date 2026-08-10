"""
Day 19 작업: 예측 모델 로딩/추론 서비스.

두 부분으로 구성:
1) ZonePredictor  : 최종 예측기(직렬화 대상). RF(delta)로 중심 이동량을 예측하고,
                    단계별 축소비율로 반경을, 단계별 표준편차로 신뢰구간을 낸다.
2) get_predictor(): 앱 기동 후 '한 번만' joblib에서 로드해 재사용하는 캐시 로더.

joblib 파일은 이 모듈의 ZonePredictor 클래스로 복원되므로,
학습 스크립트(ml/train_final.py)도 반드시 이 클래스를 import해서 저장해야 한다.

[맵별 모델 확장 여지]
지금은 전역 RF 하나(rf_pipeline)로 모든 맵을 처리하되, map을 특징으로 넣어
맵별 차이를 일부 학습한다. 나중에 맵별 전용 모델이 생기면 predict()에서
map_models[map]을 우선 쓰고 없으면 전역 rf로 폴백하도록 확장하면 된다.
(현재는 map_models=None으로 비워둠 — memories/model_upgrade_roadmap.md 참고)
"""
import math
import os

import pandas as pd

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "models", "predictor.joblib")

# 2D 가우시안 신뢰수준 스케일 (chi-square dof=2): 95% → 2.448σ
CHI2_SCALE = {0.68: 1.515, 0.90: 2.146, 0.95: 2.448}

# rf_pipeline이 학습 때 사용한 입력 컬럼 순서(추론 시 동일해야 함)
FEATURE_COLS = ["map", "safety_x", "safety_y", "safety_radius", "phase"]


class ZonePredictor:
    """최종 예측기. joblib으로 통째로 저장/복원된다."""

    def __init__(self, rf_pipeline, phase_shrink: dict, phase_sigma: dict,
                 map_models=None):
        self.rf_pipeline = rf_pipeline          # 이동량(dx, dy) 회귀 파이프라인
        self.phase_shrink = phase_shrink        # {phase: 평균 축소비율}
        self.phase_sigma = phase_sigma          # {phase: 이동량 표준편차(신뢰구간용)}
        self.map_models = map_models or {}      # (확장용) 맵별 전용 모델
        self.known_phases = sorted(phase_shrink.keys())

    def _nearest_phase(self, phase: int) -> int:
        if phase in self.phase_shrink:
            return phase
        lower = [p for p in self.known_phases if p <= phase]
        return max(lower) if lower else self.known_phases[0]

    def predict(self, safety_x: float, safety_y: float, safety_radius: float,
                phase: int, map_name: str = "unknown", conf: float = 0.95) -> dict:
        """
        현재 원(safety) + 단계 + 맵 → 다음 원 중심/반경/신뢰구간 예측.
        중심 = 현재 + RF가 예측한 이동량. 반경 = 현재 * 단계 평균 축소비율.
        """
        # RF 입력은 학습 때와 동일한 컬럼의 1행 DataFrame으로 구성
        row = pd.DataFrame([[map_name, safety_x, safety_y, safety_radius, phase]],
                           columns=FEATURE_COLS)
        dx, dy = self.rf_pipeline.predict(row)[0]

        x = safety_x + dx
        y = safety_y + dy

        used = self._nearest_phase(phase)
        radius = safety_radius * self.phase_shrink[used]

        sigma = self.phase_sigma[used]
        confidence_radius = CHI2_SCALE.get(conf, 2.448) * sigma

        return {
            "x": float(x), "y": float(y), "radius": float(radius),
            "confidence": conf,
            "confidence_radius": float(confidence_radius),
            "used_phase": used,
        }


# ---------- 캐시 로더: 앱 생애주기 동안 1회만 로드 ----------
_PREDICTOR: ZonePredictor | None = None


def get_predictor(path: str = DEFAULT_MODEL_PATH) -> ZonePredictor:
    """
    최초 호출 때만 joblib에서 모델을 읽고, 이후에는 메모리의 동일 객체를 재사용한다.
    (FastAPI 라우터가 요청마다 이 함수를 불러도 디스크 로드는 한 번만 일어난다.)
    """
    global _PREDICTOR
    if _PREDICTOR is None:
        import joblib
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} 없음. 먼저 ml/train_final.py로 모델을 학습·저장하세요."
            )
        print(f"[model_service] 모델 로드: {path}")
        _PREDICTOR = joblib.load(path)
    return _PREDICTOR
