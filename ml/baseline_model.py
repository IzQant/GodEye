"""
Day 12 작업: 규칙 기반 베이스라인 예측 모델.

아이디어(가장 단순한 방법):
  다음 원 중심 = 현재 원 중심 + (그 단계의 평균 이동 벡터)
  다음 원 반경 = 현재 원 반경 * (그 단계의 평균 축소 비율)

여기서 "그 단계의 평균 통계"는 Day 11이 만든 ml/phase_stats.json을 그대로 쓴다.
학습(train)이랄 것도 없이, 데이터에서 뽑은 평균을 규칙으로 적용하는 방식이라
'베이스라인'이다. Week 3에서 회귀/확률분포 모델과 성능을 비교할 기준점이 된다.

완료 기준(Day 12): 임의 입력(현재 원)에 대해 다음 원 좌표를 추정하는 함수 동작.
"""
import json
import os

BASE = os.path.dirname(__file__)
DEFAULT_STATS_PATH = os.path.join(BASE, "phase_stats.json")


class BaselineModel:
    def __init__(self, stats_path: str = DEFAULT_STATS_PATH):
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"{stats_path} 없음. 먼저 ml/analyze_patterns.py를 실행하세요."
            )
        with open(stats_path, encoding="utf-8") as f:
            # JSON의 key는 문자열이므로 int로 바꿔 다루기 쉽게 저장
            self.phase_stats = {int(k): v for k, v in json.load(f).items()}
        # 통계가 있는 단계 목록(정렬)
        self.known_phases = sorted(self.phase_stats.keys())

    @classmethod
    def from_stats(cls, stats: dict) -> "BaselineModel":
        """
        파일 대신 통계 dict로 모델을 만든다.
        평가(evaluate.py)에서 train 데이터로만 통계를 만들어 주입할 때 사용
        (데이터 누수 방지 — test 매치는 통계 계산에 넣지 않는다).
        """
        obj = cls.__new__(cls)  # __init__(파일 로드)를 건너뛴다
        obj.phase_stats = {int(k): v for k, v in stats.items()}
        obj.known_phases = sorted(obj.phase_stats.keys())
        return obj

    def _resolve_phase(self, phase: int) -> int:
        """
        요청한 phase의 통계가 없을 때의 대비책(fallback).
        - 정확히 있으면 그대로 사용
        - 없으면 '그 이하의 가장 가까운 단계'를 사용 (없으면 가장 작은 단계)
        표본이 부족한 단계(예: P9)나 범위를 벗어난 요청을 안전하게 처리한다.
        """
        if phase in self.phase_stats:
            return phase
        lower = [p for p in self.known_phases if p <= phase]
        if lower:
            return max(lower)
        return self.known_phases[0]

    def predict(self, current_x: float, current_y: float,
                current_radius: float, phase: int) -> dict:
        """
        현재 원(중심 x,y + 반경)과 단계를 받아 다음 원을 추정한다.
        반환: {x, y, radius, confidence_radius, used_phase}
        - confidence_radius: 이동 벡터의 표준편차 크기를 불확실성의 근사로 제공
        """
        used = self._resolve_phase(phase)
        s = self.phase_stats[used]

        next_x = current_x + s["dx_mean"]
        next_y = current_y + s["dy_mean"]
        next_radius = current_radius * s["shrink_ratio_mean"]

        # 이동의 흩어짐(dx_std, dy_std)을 하나의 반경으로 합쳐 불확실성 근사값 제공
        confidence_radius = (s.get("dx_std", 0.0) ** 2 + s.get("dy_std", 0.0) ** 2) ** 0.5

        return {
            "x": next_x,
            "y": next_y,
            "radius": next_radius,
            "confidence_radius": confidence_radius,
            "used_phase": used,
        }


if __name__ == "__main__":
    # 완료 기준 확인: 임의 입력에 대해 다음 원 추정이 동작하는지 데모
    model = BaselineModel()
    print(f"통계 보유 단계: {model.known_phases}\n")

    # 임의의 현재 원 몇 개로 테스트 (좌표 단위는 텔레메트리 기준 cm)
    samples = [
        # (current_x, current_y, current_radius, phase)
        (400000, 400000, 200000, 2),
        (350000, 550000, 100000, 5),
        (320000, 575000, 40000, 6),
        (300000, 300000, 20000, 99),  # 없는 단계 → fallback 확인
    ]
    for cx, cy, cr, ph in samples:
        out = model.predict(cx, cy, cr, ph)
        print(f"입력 phase={ph:>2} 현재(({cx},{cy}) r={cr}) → "
              f"예측(({out['x']:.0f},{out['y']:.0f}) r={out['radius']:.0f}) "
              f"±{out['confidence_radius']:.0f} [used_phase={out['used_phase']}]")

    print("\n✅ Day 12 완료 기준 통과: 임의 입력에 대한 다음 원 추정 동작 확인")
